# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import asyncio
import json
import logging
import os
import re
from typing import Any, Optional
from uuid import uuid4

from arft.agent_flow.agent_flow import AgentFlowBase, AgentFlowOutput, AgentFlowStep, register
from verl.experimental.agent_loop.tool_parser import FunctionCall, ToolParser
from verl.tools.schemas import ToolResponse
from verl.utils.profiler import simple_timer
from verl.utils.rollout_trace import rollout_trace_op

from recipe.time_series_forecast.prompts import *
from recipe.time_series_forecast.prompts import build_runtime_user_prompt
from recipe.time_series_forecast.task_protocol import parse_task_prompt
from recipe.time_series_forecast.utils import (
    parse_time_series_string,
    parse_time_series_to_dataframe,
    format_prediction_tool_output,
    format_predictions_to_string,
    get_last_timestamp,
    predict_time_series_async,
    extract_basic_statistics,
    format_basic_statistics,
    extract_within_channel_dynamics,
    format_within_channel_dynamics,
    extract_forecast_residuals,
    format_forecast_residuals,
    extract_data_quality,
    format_data_quality,
    extract_event_summary,
    format_event_summary,
)
from recipe.time_series_forecast.reward import (
    compute_score,
    extract_values_from_time_series_string,
    extract_ground_truth_values,
    normalize_for_reward,
)
import numpy as np

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


@register("time_series_forecast_agent")
class TimeSeriesForecastAgentFlow(AgentFlowBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.history_analysis = []
        self.raw_prompt_text = ""
        self.time_series_data = ""
        self.steps = []
        
        # Parsed data cache
        self.timestamps = None
        self.values = None
        self.prediction_results = None
        self.prediction_tool_output = None
        self.prediction_model_used = None
        self.final_answer = None
        self.final_answer_reject_reason = None
        self.data_source = "ETTh1"
        self.target_column = "OT"
        self.parse_error_message = None

        # Feature extraction caches
        self.basic_statistics = None
        self.within_channel_dynamics = None
        self.forecast_residuals = None
        self.data_quality = None
        self.event_summary = None
        self.parse_error_message = None
        self.io_log_path = os.getenv(
            "TS_FORECAST_IO_JSONL_PATH",
            os.path.join(os.path.dirname(__file__), "time_series_forecast_io.jsonl"),
        )

    @classmethod
    def init_class(cls, config, tokenizer, processor, **kwargs):
        if cls._class_initialized:
            return
        cls._class_initialized = True
        print("Performing class-level TimeSeriesForecastAgentFlow initialization")

        # Initialize tools from config file
        cls.tokenizer = tokenizer
        cls.processor = processor
        cls.max_steps = kwargs.get("max_steps", 5)
        cls.max_parallel_calls = kwargs.get("max_parallel_calls", 5)
        cls.lookback_window = kwargs.get("lookback_window", 96)
        cls.forecast_horizon = kwargs.get("forecast_horizon", 96)
        cls.tool_parser = ToolParser.get_tool_parser(config.actor_rollout_ref.rollout.multi_turn.format, cls.tokenizer)
        cls.prompt_length = config.actor_rollout_ref.rollout.prompt_length
        cls.response_length = config.actor_rollout_ref.rollout.response_length
        cls.tool_schemas = TIMESERIES_TOOL_SCHEMAS

    @rollout_trace_op
    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentFlowOutput:
        self.history_analysis = []
        self.steps = []
        self.timestamps = None
        self.values = None
        self.prediction_results = None
        self.prediction_tool_output = None
        self.prediction_model_used = None
        self.final_answer = None
        self.final_answer_reject_reason = None
        self.basic_statistics = None
        self.within_channel_dynamics = None
        self.forecast_residuals = None
        self.data_quality = None
        self.event_summary = None

        raw_prompt = list(kwargs["raw_prompt"])
        self.raw_prompt_text = raw_prompt[0]["content"]

        task_spec = parse_task_prompt(self.raw_prompt_text, data_source=kwargs.get("data_source"))
        self.time_series_data = task_spec.historical_data or self.raw_prompt_text
        self.data_source = task_spec.data_source or self.data_source
        self.target_column = task_spec.target_column or self.target_column
        if task_spec.lookback_window is not None:
            self.lookback_window = task_spec.lookback_window
        if task_spec.forecast_horizon is not None:
            self.forecast_horizon = task_spec.forecast_horizon
        
        # Get ground_truth from reward_model field
        reward_model = kwargs.get("reward_model", {})
        ground_truth = reward_model.get("ground_truth", "") if isinstance(reward_model, dict) else ""
        
        # Parse the input data once at the beginning
        try:
            self.timestamps, self.values = parse_time_series_string(
                self.time_series_data,
                target_column=self.target_column,
            )
        except Exception as e:
            sample_index = kwargs.get("index")
            self.parse_error_message = f"{type(e).__name__}: {e}"
            logger.error(
                "Error parsing time series data for sample index=%s: %s",
                sample_index,
                self.parse_error_message,
            )
            self.timestamps, self.values = [], []

        metrics = {}
        request_id = uuid4().hex[:8]
        system_prompt = build_timeseries_system_prompt(
            data_source=self.data_source,
            target_column=self.target_column,
        )

        if self.parse_error_message is not None:
            return await self._build_parse_failure_output(system_prompt, **kwargs)
        
        orig_mse = np.nan
        orig_mae = np.nan
        norm_mse = np.nan
        norm_mae = np.nan
        pred_len = 0
        expected_len = len(extract_ground_truth_values(ground_truth)) if ground_truth else int(self.forecast_horizon or 96)
        generation_stop_reason = ""
        io_records: list[dict[str, Any]] = []

        num_steps = 0
        while num_steps < self.max_steps:
            num_steps += 1
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": self._build_user_prompt()},
            ]

            apply_chat_template_kwargs = {
                "tools": self.tool_schemas,
                "add_generation_prompt": True,
                "tokenize": True,
                # Keep runtime behavior aligned with the SFT data. Allowing free-form
                # reasoning before tool calls makes the prompt balloon across turns.
                "enable_thinking": False,
            }

            prompt_ids = await self.loop.run_in_executor(
                None,
                lambda: self.tokenizer.apply_chat_template(messages, **apply_chat_template_kwargs),
            )
            current_sampling_params = self._prepare_sampling_params(sampling_params)

            with simple_timer("generate_sequences", metrics):
                output = await self.server_manager.generate(
                    request_id=request_id, prompt_ids=prompt_ids, sampling_params=current_sampling_params
                )
            response_ids = output.token_ids[: self.response_length]
            generation_stop_reason = getattr(output, "stop_reason", "")
            
            # Decode response to check for final answer
            response_text = await self.loop.run_in_executor(None, self.tokenizer.decode, response_ids)

            # io_records.append(
            #     {
            #         "step": num_steps,
            #         "input": messages[1]["content"],
            #         "output": response_text,
            #     }
            # )

            final_answer, format_penalty = self._extract_final_answer(response_text)

            if final_answer:
                # Final answer detected - but first validate the workflow was followed
                workflow_valid, workflow_penalty, workflow_msg = self._validate_workflow_completion(final_answer)
                
                if not workflow_valid:
                    # Workflow not completed - apply penalty (reward hacking prevention)
                    reward_score = workflow_penalty
                    logger.warning(f"Workflow violation detected: {workflow_msg}. Penalty: {reward_score}")
                    # Don't accept this as final answer - force model to continue
                    # Set final_answer to None so the loop continues
                    final_answer = None
                else:
                    # Workflow completed - compute reward based on prediction accuracy
                    self.final_answer = final_answer
                    reward_score = self._compute_final_reward(final_answer, ground_truth) + format_penalty
                    logger.info(f"Final answer detected. Reward score: {reward_score}")
                    # if reward_score > 0.5:
                    #     for record in io_records:
                    #         record["request_id"] = request_id
                    #         record["reward_score"] = reward_score
                    #     await self._append_jsonl_records(self.io_log_path, io_records)
            else:
                # No final answer yet - process tool calls
                assistant_content, tool_calls = await self.tool_parser.extract_tool_calls(response_ids)
                tool_calls = tool_calls[:self.max_parallel_calls]
                tool_call_payloads = self._build_tool_call_payloads(tool_calls, step_index=num_steps)

                # Use compact state as memory. Do not carry long assistant prose
                # across turns; only executed tool effects update state.
                for tool_call, tool_payload in zip(tool_calls, tool_call_payloads):
                    tool_output = await self._execute_tool_call(tool_call, **kwargs)

                # Small reward for making progress (using tools correctly)
                reward_score = 0.01 if tool_calls else 0.0

            step = AgentFlowStep(
                prompt_ids=prompt_ids,
                response_ids=response_ids,
                response_logprobs=output.log_probs[: self.response_length] if output.log_probs else None,
                reward_score=reward_score,
            )
            step = await self._postprocess(step, **kwargs)
            reward_extra_info = dict(step.extra_fields.get("reward_extra_info", {}) or {})
            selected_model = self.prediction_model_used or reward_extra_info.get("selected_model") or "unknown"
            reward_extra_info.update(
                {
                    "MSE": orig_mse,
                    "MAE": orig_mae,
                    "orig_mse": orig_mse,
                    "orig_mae": orig_mae,
                    "norm_mse": norm_mse,
                    "norm_mae": norm_mae,
                    "pred_len": pred_len,
                    "expected_len": expected_len,
                    "final_answer_reject_reason": self.final_answer_reject_reason,
                    "prediction_model_used": self.prediction_model_used,
                    "output_source": self.prediction_model_used,
                    "selected_model": selected_model,
                    "generation_stop_reason": generation_stop_reason,
                }
            )
            step.extra_fields["reward_extra_info"] = reward_extra_info
            self.steps.append(step)
            
            # If final answer is detected, we can stop
            if final_answer:
                break

        
        if self.final_answer and ground_truth:
            try:
                pred_values = extract_values_from_time_series_string(self.final_answer)
                gt_values = extract_ground_truth_values(ground_truth)
                pred_len = len(pred_values)
                expected_len = len(gt_values)

                if pred_values and gt_values and pred_len == expected_len:
                    pred_arr = np.array(pred_values)
                    gt_arr = np.array(gt_values)
                    orig_mse = float(np.mean((pred_arr - gt_arr) ** 2))
                    orig_mae = float(np.mean(np.abs(pred_arr - gt_arr)))
                    norm_pred, norm_gt = normalize_for_reward(pred_values, gt_values)
                    norm_pred_arr = np.array(norm_pred)
                    norm_gt_arr = np.array(norm_gt)
                    norm_mse = float(np.mean((norm_pred_arr - norm_gt_arr) ** 2))
                    norm_mae = float(np.mean(np.abs(norm_pred_arr - norm_gt_arr)))

                    logger.info(
                        "Metrics - orig_mse: %.4f, orig_mae: %.4f, norm_mse: %.4f, norm_mae: %.4f",
                        orig_mse,
                        orig_mae,
                        norm_mse,
                        norm_mae,
                    )
            except Exception as e:
                logger.error(f"Error calculating metrics: {e}")
        
        final_reward_extra_info = dict(self.steps[-1].extra_fields.get("reward_extra_info", {}) or {})
        final_selected_model = self.prediction_model_used or final_reward_extra_info.get("selected_model") or "unknown"
        final_reward_extra_info.update(
            {
                "MSE": orig_mse,
                "MAE": orig_mae,
                "orig_mse": orig_mse,
                "orig_mae": orig_mae,
                "norm_mse": norm_mse,
                "norm_mae": norm_mae,
                "pred_len": pred_len,
                "expected_len": expected_len,
                "final_answer_reject_reason": self.final_answer_reject_reason,
                "prediction_model_used": self.prediction_model_used,
                "output_source": self.prediction_model_used,
                "selected_model": final_selected_model,
                "generation_stop_reason": generation_stop_reason,
            }
        )
        self.steps[-1].extra_fields["reward_extra_info"] = final_reward_extra_info

        return AgentFlowOutput(steps=self.steps, metrics=metrics)

    def _validate_workflow_completion(self, final_answer: str) -> tuple[bool, float, str]:
        """
        Validate that the workflow was properly completed before accepting final answer.
        Prevents reward hacking where model skips tools and copies input data.
        
        Returns:
            Tuple of (is_valid, penalty_score, message)
        """
        # Check 1: Must have called at least one feature extraction tool
        if not self._has_any_feature_analysis():
            return False, -0.5, "No feature extraction tools were called. You must analyze the data first."
        
        # Check 2: Must have called predict_time_series
        if self.prediction_results is None:
            return False, -0.5, "predict_time_series was not called. You must get model predictions first."
        
        # Check 3: Check if answer is just copying the input data (reward hacking detection)
        if self._is_copying_input(final_answer):
            return False, -1.0, "Answer appears to copy input data. Predictions must be for FUTURE timestamps."
        
        return True, 0.0, "Workflow completed correctly"
    
    def _is_copying_input(self, final_answer: str) -> bool:
        """
        Detect if the model is copying input data instead of providing predictions.
        Checks if timestamps in answer overlap significantly with input timestamps.
        """
        if self.timestamps is None or len(self.timestamps) == 0:
            return False
        
        # Extract timestamps from the answer
        answer_timestamps = []
        lines = final_answer.strip().split('\n')
        for line in lines:
            line = line.strip()
            if not line:
                continue
            # Try to extract timestamp (format: "2018-05-19 00:00:00 value")
            match = re.match(r'(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})', line)
            if match:
                answer_timestamps.append(match.group(1))
        
        if not answer_timestamps:
            return False
        
        # Convert input timestamps to strings for comparison
        input_ts_strings = set()
        for ts in self.timestamps:
            if hasattr(ts, 'strftime'):
                input_ts_strings.add(ts.strftime('%Y-%m-%d %H:%M:%S'))
            else:
                input_ts_strings.add(str(ts))
        
        # Count how many answer timestamps are in the input
        overlap_count = sum(1 for ts in answer_timestamps if ts in input_ts_strings)
        overlap_ratio = overlap_count / len(answer_timestamps) if answer_timestamps else 0
        
        # If more than 50% of answer timestamps are from input, it's likely copying
        if overlap_ratio > 0.5:
            logger.warning(f"Detected input copying: {overlap_ratio:.1%} of answer timestamps match input")
            return True
        
        return False

    def _expected_prediction_count(self) -> int:
        return int(self.forecast_horizon or 96)

    def _final_answer_max_tokens(self) -> int:
        expected = self._expected_prediction_count()
        # Keep enough headroom for 96 numeric lines plus tags, but avoid letting
        # malformed final outputs run all the way to the global response cap.
        return min(int(self.response_length or 0), max(256, expected * 10 + 64))

    def _prepare_sampling_params(self, sampling_params: dict[str, Any]) -> dict[str, Any]:
        params = dict(sampling_params)
        if self.prediction_results:
            params["stop"] = ["</answer>"]
            params["include_stop_str_in_output"] = True
            existing_max_tokens = params.get("max_tokens", params.get("max_new_tokens"))
            final_turn_max_tokens = self._final_answer_max_tokens()
            if existing_max_tokens is None:
                params["max_tokens"] = final_turn_max_tokens
            else:
                params["max_tokens"] = min(int(existing_max_tokens), final_turn_max_tokens)
                params.pop("max_new_tokens", None)
        return params

    def _extract_forecast_block(self, text: str) -> Optional[str]:
        cleaned = (
            text.replace("<|im_end|>", "\n")
            .replace("<answer>", "\n")
            .replace("</answer>", "\n")
            .strip()
        )
        lines = cleaned.splitlines()
        collected: list[str] = []
        started = False
        for raw_line in lines:
            line = raw_line.strip()
            if not line:
                continue

            is_timestamp_value = re.match(r"^\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\s+-?\d+\.?\d*$", line) is not None
            is_numeric_value = re.match(r"^-?\d+\.?\d*$", line) is not None
            if not started:
                if is_timestamp_value or is_numeric_value:
                    started = True
                    collected.append(line)
                continue

            if is_timestamp_value or is_numeric_value:
                collected.append(line)
                continue

            break

        if not collected:
            return None
        return "\n".join(collected)

    def _looks_like_forecast_answer(self, answer_text: Optional[str]) -> bool:
        if not answer_text:
            return False
        expected = self._expected_prediction_count()
        lines = [line.strip() for line in answer_text.splitlines() if line.strip()]
        values = extract_values_from_time_series_string(answer_text)
        if len(lines) != expected or len(values) != expected:
            return False
        for line in lines:
            if re.fullmatch(r"-?\d+(?:\.\d+)?", line) is None:
                return False
        return True

    def _infer_final_answer_reject_reason(self, answer_text: str) -> str:
        expected = self._expected_prediction_count()
        lines = [line.strip() for line in answer_text.splitlines() if line.strip()]
        values = extract_values_from_time_series_string(answer_text)
        if not lines:
            return "empty_answer_block"
        if len(lines) != expected:
            return f"invalid_answer_shape:lines={len(lines)},expected={expected}"
        if len(values) != expected:
            return f"invalid_answer_shape:values={len(values)},expected={expected}"
        for line in lines:
            if re.fullmatch(r"-?\d+(?:\.\d+)?", line) is None:
                return "invalid_answer_shape:non_numeric_line"
        return "invalid_answer_shape:unknown"

    def _extract_final_answer(self, response_text: str) -> tuple[Optional[str], float]:
        """
        Extract final answer from response text.

        Args:
            response_text: The model's response text
            
        Returns:
            Tuple of (answer_text, format_penalty).
        """
        self.final_answer_reject_reason = None
        match = re.search(r'<answer>(.*?)</answer>', response_text, re.DOTALL)
        if match:
            candidate = match.group(1).strip()
            if self._looks_like_forecast_answer(candidate):
                return candidate, 0.0
            self.final_answer_reject_reason = self._infer_final_answer_reject_reason(candidate)
            return None, 0.0

        if "<answer>" in response_text and "</answer>" not in response_text:
            self.final_answer_reject_reason = "missing_answer_close_tag"
        elif "</answer>" in response_text and "<answer>" not in response_text:
            self.final_answer_reject_reason = "missing_answer_open_tag"
        else:
            self.final_answer_reject_reason = "missing_answer_block"

        return None, 0.0

    async def _build_parse_failure_output(self, system_prompt: str, **kwargs) -> AgentFlowOutput:
        user_prompt = self._build_user_prompt()
        prompt_ids = await self.loop.run_in_executor(
            None,
            lambda: self.tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                tools=self.tool_schemas,
                add_generation_prompt=True,
                tokenize=True,
                enable_thinking=False,
            ),
        )
        response_ids = self.tokenizer.encode("PARSE_FAILURE", add_special_tokens=False)
        if not response_ids:
            fallback_token = self.tokenizer.eos_token_id
            if fallback_token is None:
                fallback_token = self.tokenizer.pad_token_id or 0
            response_ids = [fallback_token]

        step = AgentFlowStep(
            prompt_ids=prompt_ids,
            response_ids=response_ids,
            response_logprobs=None,
            reward_score=-1.0,
            extra_fields={
                "parse_error": self.parse_error_message,
                "sample_index": kwargs.get("index"),
                "reward_extra_info": {
                    "parse_failed": True,
                    "parse_error": self.parse_error_message,
                    "sample_index": kwargs.get("index"),
                },
            },
        )
        step = await self._postprocess(step, **kwargs)
        return AgentFlowOutput(steps=[step], metrics={})

    def _compute_final_reward(self, final_answer: str, ground_truth: str) -> float:
        """
        Compute reward score based on final prediction and ground truth.
        
        Args:
            final_answer: The predicted values as string
            ground_truth: The ground truth values as string
            
        Returns:
            Reward score
        """
        if not ground_truth:
            return 0.0
        
        try:
            # Use the compute_score function from reward.py
            result = compute_score(
                data_source="time_series",
                solution_str=f"<answer>{final_answer}</answer>",
                ground_truth=ground_truth
            )
            return float(result["score"] if isinstance(result, dict) else result)
        except Exception as e:
            logger.error(f"Error computing final reward: {e}")
            return -0.5

    async def _append_jsonl_records(self, path: str, records: list[dict[str, Any]]) -> None:
        def _write_records() -> None:
            if not records:
                return
            dir_path = os.path.dirname(path)
            if dir_path:
                os.makedirs(dir_path, exist_ok=True)
            with open(path, "a", encoding="utf-8") as handle:
                for record in records:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")

        await self.loop.run_in_executor(None, _write_records)

    def _build_user_prompt(self) -> str:
        return build_runtime_user_prompt(
            data_source=self.data_source,
            target_column=self.target_column,
            lookback_window=self.lookback_window,
            forecast_horizon=self.forecast_horizon,
            time_series_data=self.time_series_data,
            history_analysis=self.history_analysis,
            prediction_results=self.prediction_tool_output or self.prediction_results,
            prediction_model_used=self.prediction_model_used,
            conversation_has_tool_history=bool(self.history_analysis or self.prediction_tool_output or self.prediction_results),
        )

    def _build_tool_call_payloads(self, tool_calls: list[FunctionCall], step_index: int) -> list[dict[str, Any]]:
        payloads: list[dict[str, Any]] = []
        for call_index, tool_call in enumerate(tool_calls, start=1):
            arguments = tool_call.arguments
            if not isinstance(arguments, str):
                arguments = json.dumps(arguments, ensure_ascii=False, separators=(",", ":"))
            payloads.append(
                {
                    "id": f"call_{step_index}_{call_index}_{tool_call.name}",
                    "type": "function",
                    "function": {
                        "name": tool_call.name,
                        "arguments": arguments,
                    },
                }
            )
        return payloads

    async def _execute_tool_call(self, tool_call: FunctionCall, **kwargs) -> Optional[str]:
        if tool_call.name == "predict_time_series":
            if not self._has_any_feature_analysis():
                logger.warning("predict_time_series called without feature analysis, rejected")
                return None

            model_name = None
            if hasattr(tool_call, "arguments") and tool_call.arguments:
                args = tool_call.arguments
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {}
                if isinstance(args, dict):
                    model_name = args.get("model_name")
            if model_name not in {"chronos2", "arima", "patchtst", "itransformer"}:
                model_name = "chronos2"
            return await self._run_prediction_tool(model_name=model_name)

        if tool_call.name == "extract_basic_statistics":
            return self._run_basic_statistics_tool()
        if tool_call.name == "extract_within_channel_dynamics":
            return self._run_within_channel_dynamics_tool()
        if tool_call.name == "extract_forecast_residuals":
            return self._run_forecast_residuals_tool()
        if tool_call.name == "extract_data_quality":
            return self._run_data_quality_tool()
        if tool_call.name == "extract_event_summary":
            return self._run_event_summary_tool()
        return None

    def _has_any_feature_analysis(self) -> bool:
        """Check if any feature analysis has been performed."""
        return any([
            self.basic_statistics is not None,
            self.within_channel_dynamics is not None,
            self.forecast_residuals is not None,
            self.data_quality is not None,
            self.event_summary is not None,
        ])

    async def predict(self, model_name: str = "chronos2", **kwargs) -> float:
        """
        Generate predictions using the specified model.

        Args:
            model_name: Name of the model to use ("chronos2" or "arima")

        Returns:
            Reward score (0.0 for tool execution, actual reward at final step)
        """
        await self._run_prediction_tool(model_name=model_name)
        return 0.0

    async def _run_prediction_tool(self, model_name: str = "chronos2") -> Optional[str]:
        try:
            if not self.values or len(self.values) < 2:
                logger.warning("Insufficient data for prediction")
                return None

            # Convert to DataFrame for prediction
            context_df = parse_time_series_to_dataframe(
                self.time_series_data,
                series_id=self.data_source or "series_0",
                target_column=self.target_column,
            )

            last_ts = get_last_timestamp(self.time_series_data)

            pred_df = await predict_time_series_async(
                context_df,
                prediction_length=self.forecast_horizon,
                model_name=model_name,
            )

            self.prediction_model_used = model_name
            self.prediction_results = format_predictions_to_string(pred_df, last_ts)
            self.prediction_tool_output = format_prediction_tool_output(
                pred_df,
                last_timestamp=last_ts,
                model_name=model_name,
            )

            self.history_analysis.append(
                f"Model Prediction: Generated {self.forecast_horizon}-step forecast using {model_name.upper()} model"
            )

            logger.info(f"Prediction completed using {model_name} model")

            return self.prediction_tool_output

        except Exception as e:
            logger.error(f"Error in predict with {model_name}: {e}")
            self.prediction_results = None
            self.prediction_tool_output = None
            self.history_analysis.append(f"Prediction failed ({model_name}): {str(e)}")
            return None

    async def extract_basic_statistics(self, **kwargs) -> float:
        """Extract core statistical features from time series data."""
        self._run_basic_statistics_tool()
        return 0.0

    def _run_basic_statistics_tool(self) -> Optional[str]:
        try:
            if not self.values or len(self.values) < 2:
                logger.warning("Insufficient data for basic statistics extraction")
                return None

            features = extract_basic_statistics(data=self.values)
            self.basic_statistics = features

            analysis_record = format_basic_statistics(features)
            self.history_analysis.append(analysis_record)

            logger.info("Basic statistics extraction completed")
            return analysis_record
        except Exception as e:
            logger.error(f"Error in extract_basic_statistics: {e}")
            return None

    async def extract_within_channel_dynamics(self, **kwargs) -> float:
        """Extract within-channel dynamics features from time series data."""
        self._run_within_channel_dynamics_tool()
        return 0.0

    def _run_within_channel_dynamics_tool(self) -> Optional[str]:
        try:
            if not self.values or len(self.values) < 2:
                logger.warning("Insufficient data for within-channel dynamics extraction")
                return None

            features = extract_within_channel_dynamics(data=self.values)
            self.within_channel_dynamics = features

            analysis_record = format_within_channel_dynamics(features)
            self.history_analysis.append(analysis_record)

            logger.info("Within-channel dynamics extraction completed")
            return analysis_record
        except Exception as e:
            logger.error(f"Error in extract_within_channel_dynamics: {e}")
            return None

    async def extract_forecast_residuals(self, **kwargs) -> float:
        """Extract forecast residual features from time series data."""
        self._run_forecast_residuals_tool()
        return 0.0

    def _run_forecast_residuals_tool(self) -> Optional[str]:
        try:
            if not self.values or len(self.values) < 2:
                logger.warning("Insufficient data for forecast residuals extraction")
                return None

            features = extract_forecast_residuals(data=self.values)
            self.forecast_residuals = features

            analysis_record = format_forecast_residuals(features)
            self.history_analysis.append(analysis_record)

            logger.info("Forecast residuals extraction completed")
            return analysis_record
        except Exception as e:
            logger.error(f"Error in extract_forecast_residuals: {e}")
            return None

    async def extract_data_quality(self, **kwargs) -> float:
        """Extract data quality features from time series data."""
        self._run_data_quality_tool()
        return 0.0

    def _run_data_quality_tool(self) -> Optional[str]:
        try:
            if not self.values or len(self.values) < 2:
                logger.warning("Insufficient data for data quality extraction")
                return None

            features = extract_data_quality(data=self.values)
            self.data_quality = features

            analysis_record = format_data_quality(features)
            self.history_analysis.append(analysis_record)

            logger.info("Data quality extraction completed")
            return analysis_record
        except Exception as e:
            logger.error(f"Error in extract_data_quality: {e}")
            return None

    async def extract_event_summary(self, **kwargs) -> float:
        """Extract event summary features from time series data."""
        self._run_event_summary_tool()
        return 0.0

    def _run_event_summary_tool(self) -> Optional[str]:
        try:
            if not self.values or len(self.values) < 2:
                logger.warning("Insufficient data for event summary extraction")
                return None

            features = extract_event_summary(data=self.values)
            self.event_summary = features

            analysis_record = format_event_summary(features)
            self.history_analysis.append(analysis_record)

            logger.info("Event summary extraction completed")
            return analysis_record
        except Exception as e:
            logger.error(f"Error in extract_event_summary: {e}")
            return None
