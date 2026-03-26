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
from typing import Any, Optional
from uuid import uuid4

import numpy as np

from arft.agent_flow.agent_flow import AgentFlowBase, AgentFlowOutput, AgentFlowStep, register
from verl.experimental.agent_loop.tool_parser import FunctionCall, ToolParser
from verl.utils.chain_debug import append_chain_debug
from verl.utils.profiler import simple_timer
from verl.utils.rollout_trace import rollout_trace_op

from recipe.time_series_forecast.agent_flow_feature_tools import FEATURE_TOOL_SPECS
from recipe.time_series_forecast.agent_flow_support import (
    analysis_state_signature,
    build_prediction_tool_debug_payload,
    build_turn_debug_payload,
    collect_refinement_metrics,
    compute_series_metrics,
    current_turn_stage,
    expected_prediction_count,
    feature_tool_signature,
    required_step_budget,
    sample_uid_text,
    shared_reward_tracking_fields,
)
from recipe.time_series_forecast.prompts import (
    FEATURE_TOOL_SCHEMAS,
    PREDICT_TIMESERIES_TOOL_SCHEMA,
    TIMESERIES_TOOL_SCHEMAS,
    build_runtime_user_prompt,
    build_timeseries_system_prompt,
)
from recipe.time_series_forecast.task_protocol import parse_task_prompt
from recipe.time_series_forecast.utils import (
    format_prediction_tool_output,
    format_predictions_to_string,
    get_last_timestamp,
    parse_time_series_string,
    parse_time_series_to_dataframe,
    predict_time_series_async,
)
from recipe.time_series_forecast.reward import (
    compute_score,
    extract_values_from_time_series_string,
    extract_ground_truth_values,
    normalize_for_reward,
    parse_final_answer_protocol,
)
from recipe.time_series_forecast.reward_protocol import extract_forecast_block, normalized_nonempty_lines
from recipe.time_series_forecast.diagnostic_policy import (
    FEATURE_TOOL_ORDER,
)

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


@register("time_series_forecast_agent")
class TimeSeriesForecastAgentFlow(AgentFlowBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.raw_prompt_text = ""
        self.time_series_data = ""
        self.data_source = "ETTh1"
        self.target_column = "OT"
        self.io_log_path = os.getenv(
            "TS_FORECAST_IO_JSONL_PATH",
            os.path.join(os.path.dirname(__file__), "time_series_forecast_io.jsonl"),
        )
        self._reset_episode_state()

    def _reset_prediction_state(self) -> None:
        self.timestamps = None
        self.values = None
        self.prediction_results = None
        self.prediction_tool_output = None
        self.prediction_model_used = None
        self.prediction_requested_model = None
        self.prediction_model_defaulted = False
        self.prediction_tool_error = None
        self.prediction_step_index = None
        self.prediction_turn_stage = None
        self.prediction_attempt_count = 0
        self.prediction_call_count = 0
        self.illegal_turn3_tool_call_count = 0
        self.feature_tool_sequence = []

    def _reset_feature_state(self) -> None:
        self.basic_statistics = None
        self.within_channel_dynamics = None
        self.forecast_residuals = None
        self.data_quality = None
        self.event_summary = None

    def _reset_episode_state(self) -> None:
        self.history_analysis = []
        self.steps = []
        self.final_answer = None
        self.final_answer_reject_reason = None
        self.final_answer_parse_mode = None
        self.final_answer_step_index = None
        self.parse_error_message = None
        self._reset_prediction_state()
        self._reset_feature_state()

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
        cls.max_prediction_attempts = max(1, int(kwargs.get("max_prediction_attempts", 2) or 2))
        cls.max_parallel_calls = kwargs.get("max_parallel_calls", 5)
        cls.lookback_window = kwargs.get("lookback_window", 96)
        cls.forecast_horizon = kwargs.get("forecast_horizon", 96)
        cls.tool_parser = ToolParser.get_tool_parser(config.actor_rollout_ref.rollout.multi_turn.format, cls.tokenizer)
        cls.prompt_length = config.actor_rollout_ref.rollout.prompt_length
        cls.response_length = config.actor_rollout_ref.rollout.response_length
        cls.tool_schemas = TIMESERIES_TOOL_SCHEMAS

    @rollout_trace_op
    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentFlowOutput:
        self._reset_episode_state()

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
        generation_finish_reason = ""
        io_records: list[dict[str, Any]] = []

        num_steps = 0
        configured_max_steps = int(self.max_steps or 0)
        while True:
            effective_max_steps = max(configured_max_steps, self._required_step_budget())
            if num_steps >= effective_max_steps:
                break
            num_steps += 1
            turn_stage = self._current_turn_stage()
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": self._build_user_prompt()},
            ]

            apply_chat_template_kwargs = {
                "add_generation_prompt": True,
                "tokenize": True,
            }
            tool_schemas = self._tool_schemas_for_turn(turn_stage)
            if tool_schemas:
                apply_chat_template_kwargs["tools"] = tool_schemas

            prompt_ids = await self.loop.run_in_executor(
                None,
                lambda: self.tokenizer.apply_chat_template(messages, **apply_chat_template_kwargs),
            )
            current_sampling_params = self._prepare_sampling_params(sampling_params, turn_stage=turn_stage)

            with simple_timer("generate_sequences", metrics):
                output = await self.server_manager.generate(
                    request_id=request_id, prompt_ids=prompt_ids, sampling_params=current_sampling_params
                )
            response_ids = output.token_ids[: self.response_length]
            generation_stop_reason = getattr(output, "stop_reason", "")
            output_extra_info = getattr(output, "extra_info", {}) or {}
            if isinstance(output_extra_info, dict):
                generation_finish_reason = str(output_extra_info.get("finish_reason") or "")
            else:
                generation_finish_reason = ""
            if not generation_finish_reason:
                generation_finish_reason = generation_stop_reason
            
            # Decode response to check for final answer
            response_text = await self.loop.run_in_executor(None, self.tokenizer.decode, response_ids)

            # io_records.append(
            #     {
            #         "step": num_steps,
            #         "input": messages[1]["content"],
            #         "output": response_text,
            #     }
            # )

            final_answer = None
            format_penalty = 0.0
            workflow_status = "not_attempted"
            workflow_message = ""
            tool_call_names: list[str] = []
            tool_calls: list[FunctionCall] = []
            executed_tool_names: list[str] = []
            terminate_after_step = False

            should_parse_tool_calls = (
                turn_stage != "refinement" or "<tool_call>" in response_text or "</tool_call>" in response_text
            )
            if should_parse_tool_calls:
                assistant_content, tool_calls = await self.tool_parser.extract_tool_calls(response_ids)
                tool_calls = tool_calls[:self.max_parallel_calls]
                tool_call_names = [tool_call.name for tool_call in tool_calls]

            if turn_stage == "refinement" or not tool_calls:
                final_answer, format_penalty = self._extract_final_answer(response_text)
            else:
                # On tool-use turns, prioritize structured tool execution over any
                # trailing answer-like garbage that the model may append.
                self.final_answer_reject_reason = None
                self.final_answer_parse_mode = "tool_call_response"

            if final_answer and tool_calls:
                for tool_call in tool_calls:
                    tool_output = await self._execute_tool_call(tool_call, turn_stage=turn_stage, **kwargs)
                    if tool_output is not None:
                        executed_tool_names.append(tool_call.name)

            if final_answer:
                # Final answer detected - but first validate the workflow was followed
                workflow_valid, workflow_penalty, workflow_msg = self._validate_workflow_completion(final_answer)
                workflow_status = "accepted" if workflow_valid else "rejected"
                workflow_message = workflow_msg
                
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
                    self.final_answer_step_index = num_steps
                    reward_score = self._compute_final_reward(response_text, ground_truth) + format_penalty
                    logger.info(f"Final answer detected. Reward score: {reward_score}")
                    # if reward_score > 0.5:
                    #     for record in io_records:
                    #         record["request_id"] = request_id
                    #         record["reward_score"] = reward_score
                    #     await self._append_jsonl_records(self.io_log_path, io_records)
            else:
                if turn_stage == "refinement":
                    for tool_call in tool_calls:
                        tool_output = await self._execute_tool_call(tool_call, turn_stage=turn_stage, **kwargs)
                        if tool_output is not None:
                            executed_tool_names.append(tool_call.name)
                    reject_reason = self.final_answer_reject_reason or (
                        "illegal_tool_call_in_refinement" if tool_calls else "missing_valid_final_answer"
                    )
                    self.final_answer_reject_reason = reject_reason
                    workflow_status = "rejected"
                    if tool_calls:
                        workflow_message = "Refinement turn must output the final answer and may not call tools."
                    else:
                        workflow_message = f"Refinement turn ended without a valid final answer: {reject_reason}."
                    reward_score = -1.0
                    terminate_after_step = True
                else:
                    # Use compact state as memory. Do not carry long assistant prose
                    # across turns; only executed tool effects update state.
                    for tool_call in tool_calls:
                        tool_output = await self._execute_tool_call(tool_call, turn_stage=turn_stage, **kwargs)
                        if tool_output is not None:
                            executed_tool_names.append(tool_call.name)

                    reward_score = self._compute_intermediate_reward(
                        turn_stage=turn_stage,
                        executed_tool_names=executed_tool_names,
                        ground_truth=ground_truth,
                    )

            step = AgentFlowStep(
                prompt_ids=prompt_ids,
                response_ids=response_ids,
                response_logprobs=output.log_probs[: self.response_length] if output.log_probs else None,
                reward_score=reward_score,
            )
            step = await self._postprocess(step, **kwargs)
            reward_extra_info = dict(step.extra_fields.get("reward_extra_info", {}) or {})
            selected_model = self.prediction_model_used or reward_extra_info.get("selected_model") or "unknown"
            refinement_metrics = self._collect_refinement_metrics(ground_truth)
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
                    "final_answer_parse_mode": self.final_answer_parse_mode,
                    "prediction_model_used": self.prediction_model_used,
                    "output_source": self.prediction_model_used,
                    "selected_model": selected_model,
                    "generation_stop_reason": generation_stop_reason,
                    "generation_finish_reason": generation_finish_reason,
                    **self._shared_reward_tracking_fields(sample_uid=kwargs.get("uid")),
                    "tool_call_count": int(len(tool_call_names)),
                    "tool_call_sequence": "->".join(tool_call_names) if tool_call_names else "",
                    "executed_tool_count": int(len(executed_tool_names)),
                    "executed_tool_sequence": "->".join(executed_tool_names) if executed_tool_names else "",
                    "rejected_tool_call_count": int(max(len(tool_call_names) - len(executed_tool_names), 0)),
                    "turn_stage": turn_stage,
                    "workflow_status": workflow_status,
                    "workflow_violation_reason": workflow_message,
                    "required_step_budget": int(self._required_step_budget()),
                    "prompt_char_len": int(len(messages[1]["content"])),
                    "response_char_len": int(len(response_text)),
                    "response_token_len": int(len(response_ids)),
                    **refinement_metrics,
                }
            )
            step.extra_fields["reward_extra_info"] = reward_extra_info
            self.steps.append(step)
            self._append_turn_debug(
                step_index=num_steps,
                request_id=request_id,
                sample_index=kwargs.get("index"),
                sample_uid=kwargs.get("uid"),
                turn_stage=turn_stage,
                prompt_text=messages[1]["content"],
                response_text=response_text,
                generation_stop_reason=generation_stop_reason,
                generation_finish_reason=generation_finish_reason,
                tool_call_names=tool_call_names,
                workflow_status=workflow_status,
                workflow_message=workflow_message,
                reward_extra_info=reward_extra_info,
            )
            
            # If final answer is detected, we can stop
            if final_answer or terminate_after_step:
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
        final_refinement_metrics = self._collect_refinement_metrics(ground_truth)
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
                "final_answer_parse_mode": self.final_answer_parse_mode,
                "prediction_model_used": self.prediction_model_used,
                "output_source": self.prediction_model_used,
                "selected_model": final_selected_model,
                "generation_stop_reason": generation_stop_reason,
                "generation_finish_reason": generation_finish_reason,
                **self._shared_reward_tracking_fields(sample_uid=kwargs.get("uid")),
                **final_refinement_metrics,
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
        # Check 1: The agent must complete some diagnostic analysis before routing.
        if not self._has_diagnostic_analysis():
            return False, -0.5, "At least one diagnostic feature tool must be executed before routing."
        
        # Check 2: Must have called predict_time_series
        if self.prediction_results is None:
            return False, -0.5, "predict_time_series was not called. You must get model predictions first."

        # Check 3: Routing must contain exactly one forecasting-tool invocation.
        if self.prediction_call_count != 1:
            return (
                False,
                -0.5,
                f"predict_time_series must be called exactly once before final output. Got {self.prediction_call_count}.",
            )

        if str(self.prediction_turn_stage or "").strip().lower() != "routing":
            return (
                False,
                -0.5,
                "predict_time_series must be called during the routing stage after diagnostics are complete.",
            )

        # Check 4: Refinement must not invoke any tools after prediction.
        if self.illegal_turn3_tool_call_count > 0:
            return (
                False,
                -0.5,
                f"Refinement may not call tools after prediction results are available. Illegal calls: {self.illegal_turn3_tool_call_count}.",
            )
        
        # Check 5: Check if answer is just copying the input data (reward hacking detection)
        if self._is_copying_input(final_answer):
            return False, -1.0, "Answer appears to copy input data. Predictions must be for FUTURE timestamps."
        
        return True, 0.0, "Workflow completed correctly"
    
    def _is_copying_input(self, final_answer: str) -> bool:
        """
        Detect if the model is copying the historical input window instead of
        forecasting future values.
        """
        if not self.values:
            return False

        answer_values = extract_values_from_time_series_string(final_answer or "")
        expected = self._expected_prediction_count()
        if len(answer_values) != expected or len(self.values) < expected:
            return False

        historical_tail = [float(value) for value in self.values[-expected:]]
        answer_arr = np.asarray(answer_values, dtype=float)
        historical_arr = np.asarray(historical_tail, dtype=float)
        exact_copy = np.allclose(answer_arr, historical_arr, atol=1e-8, rtol=0.0)
        if exact_copy:
            logger.warning("Detected input copying: final answer matches the last historical window")
            return True
        return False

    def _expected_prediction_count(self) -> int:
        return expected_prediction_count(self.forecast_horizon)

    def _current_turn_stage(self) -> str:
        return current_turn_stage(
            prediction_results=self.prediction_results,
            executed_feature_tool_names=self._executed_feature_tool_names(),
        )

    def _max_prediction_attempts(self) -> int:
        try:
            configured = int(getattr(self, "max_prediction_attempts", getattr(type(self), "max_prediction_attempts", 2)) or 2)
        except (TypeError, ValueError):
            configured = 2
        return max(configured, 1)

    def _required_step_budget(self) -> int:
        return required_step_budget(
            absolute_step_budget=getattr(self, "absolute_step_budget", None),
            configured_max_steps=getattr(self, "max_steps", getattr(type(self), "max_steps", 0)),
            max_prediction_attempts=self._max_prediction_attempts(),
        )

    def _available_diagnostic_tool_names(self) -> list[str]:
        return list(FEATURE_TOOL_ORDER)

    def _analysis_state_signature(self) -> str:
        return analysis_state_signature(self._executed_feature_tool_names())

    @staticmethod
    def _sample_uid_text(sample_uid: Any) -> str:
        return sample_uid_text(sample_uid)

    def _executed_feature_tool_names(self) -> list[str]:
        feature_state = {
            name: getattr(self, spec.state_attr) is not None
            for name, spec in FEATURE_TOOL_SPECS.items()
        }
        return [name for name in FEATURE_TOOL_ORDER if feature_state.get(name)]

    def _has_diagnostic_analysis(self) -> bool:
        return bool(self._executed_feature_tool_names())

    def _feature_tool_signature(self) -> str:
        return feature_tool_signature(self.feature_tool_sequence)

    def _shared_reward_tracking_fields(self, *, sample_uid: Any) -> dict[str, Any]:
        return shared_reward_tracking_fields(
            sample_uid=sample_uid,
            prediction_attempt_count=int(self.prediction_attempt_count),
            prediction_call_count=int(self.prediction_call_count),
            illegal_turn3_tool_call_count=int(self.illegal_turn3_tool_call_count),
            prediction_requested_model=self.prediction_requested_model or "",
            prediction_model_defaulted=bool(self.prediction_model_defaulted),
            prediction_tool_error=self.prediction_tool_error or "",
            prediction_step_index=self.prediction_step_index,
            prediction_turn_stage=self.prediction_turn_stage or "",
            final_answer_step_index=self.final_answer_step_index,
            feature_tool_sequence=list(self.feature_tool_sequence),
            required_feature_tools=[],
            executed_feature_tool_names=self._executed_feature_tool_names(),
            history_analysis=list(self.history_analysis),
            required_step_budget=int(self._required_step_budget()),
        )

    def _append_turn_debug(
        self,
        *,
        step_index: int,
        request_id: str,
        sample_index: Any,
        sample_uid: Any,
        turn_stage: str,
        prompt_text: str,
        response_text: str,
        generation_stop_reason: str,
        generation_finish_reason: str,
        tool_call_names: list[str],
        workflow_status: str,
        workflow_message: str,
        reward_extra_info: dict[str, Any],
    ) -> None:
        append_chain_debug(
            "agent_turn_summary",
            build_turn_debug_payload(
                request_id=request_id,
                sample_index=sample_index,
                sample_uid=sample_uid,
                step_index=step_index,
                turn_stage=turn_stage,
                tool_call_names=tool_call_names,
                prompt_text=prompt_text,
                response_text=response_text,
                generation_stop_reason=generation_stop_reason,
                generation_finish_reason=generation_finish_reason,
                workflow_status=workflow_status,
                workflow_message=workflow_message,
                reward_extra_info=reward_extra_info,
                feature_tool_sequence=list(self.feature_tool_sequence),
                required_feature_tools=[],
                executed_feature_tool_names=self._executed_feature_tool_names(),
                history_analysis=list(self.history_analysis),
                prediction_requested_model=self.prediction_requested_model or "",
                prediction_model_used=self.prediction_model_used or "",
                prediction_model_defaulted=bool(self.prediction_model_defaulted),
                prediction_tool_error=self.prediction_tool_error or "",
                prediction_attempt_count=int(self.prediction_attempt_count),
                prediction_call_count=int(self.prediction_call_count),
                prediction_step_index=self.prediction_step_index,
                prediction_turn_stage=self.prediction_turn_stage or "",
                final_answer_step_index=self.final_answer_step_index,
                illegal_turn3_tool_call_count=int(self.illegal_turn3_tool_call_count),
                final_answer_reject_reason=self.final_answer_reject_reason or "",
                final_answer_parse_mode=self.final_answer_parse_mode or "",
                required_step_budget=int(self._required_step_budget()),
            ),
        )

    def _final_answer_max_tokens(self) -> int:
        expected = self._expected_prediction_count()
        response_cap = int(self.response_length or 0) or 2048
        # Paper-style timestamp-value answers are much longer than the old
        # numeric-only format. On the rebuilt paper-aligned ETTh1 SFT set,
        # 96-step refinement responses cluster around ~2.7k tokens with Qwen3
        # tokenization, so the old `expected * 12 + 256` cap truncates valid
        # answers before `</answer>`.
        return min(response_cap, max(768, expected * 28 + 256))

    def _tool_turn_max_tokens(self) -> int:
        response_cap = int(self.response_length or 0) or 2048
        max_parallel_calls = int(getattr(self, "max_parallel_calls", 1) or 1)
        return min(response_cap, max(256, max_parallel_calls * 128))

    @staticmethod
    def _merge_stop_strings(existing_stops: Any, new_stops: list[str]) -> list[str]:
        merged: list[str] = []
        if isinstance(existing_stops, str):
            existing_iterable = [existing_stops]
        else:
            existing_iterable = list(existing_stops or [])
        for stop in existing_iterable + list(new_stops):
            if isinstance(stop, str) and stop not in merged:
                merged.append(stop)
        return merged

    def _prepare_sampling_params(
        self,
        sampling_params: dict[str, Any],
        *,
        turn_stage: Optional[str] = None,
    ) -> dict[str, Any]:
        params = dict(sampling_params)
        stage = turn_stage or self._current_turn_stage()
        existing_max_tokens = params.get("max_tokens", params.get("max_new_tokens"))

        if stage == "refinement":
            params["stop"] = self._merge_stop_strings(params.get("stop"), ["</answer>"])
            params["include_stop_str_in_output"] = True
            final_turn_max_tokens = self._final_answer_max_tokens()
            if existing_max_tokens is None:
                params["max_tokens"] = final_turn_max_tokens
            else:
                params["max_tokens"] = min(int(existing_max_tokens), final_turn_max_tokens)
                params.pop("max_new_tokens", None)
            return params

        params["stop"] = self._merge_stop_strings(params.get("stop"), ["<answer>"])
        params.pop("include_stop_str_in_output", None)
        tool_turn_max_tokens = self._tool_turn_max_tokens()
        if existing_max_tokens is None:
            params["max_tokens"] = tool_turn_max_tokens
        else:
            params["max_tokens"] = min(int(existing_max_tokens), tool_turn_max_tokens)
            params.pop("max_new_tokens", None)
        return params

    def _tool_schemas_for_turn(self, turn_stage: str) -> list[dict[str, Any]]:
        if turn_stage == "diagnostic":
            return list(FEATURE_TOOL_SCHEMAS)
        if turn_stage == "routing":
            return [PREDICT_TIMESERIES_TOOL_SCHEMA]
        return []

    def _extract_final_answer(self, response_text: str) -> tuple[Optional[str], float]:
        """
        Extract final answer from response text.

        Args:
            response_text: The model's response text

        Returns:
            Tuple of (answer_text, format_penalty).
        """
        self.final_answer_reject_reason = None
        self.final_answer_parse_mode = None

        final_answer, parse_mode, reject_reason = parse_final_answer_protocol(
            response_text,
            self._expected_prediction_count(),
            allow_recovery=False,
        )
        if final_answer is not None:
            if parse_mode.startswith("recovered_"):
                logger.info("Recovered malformed final answer via %s.", parse_mode)
            self.final_answer_parse_mode = parse_mode
            return final_answer, 0.0

        reject_reason = reject_reason or "unknown_parse_failure"
        self.final_answer_reject_reason = reject_reason
        self.final_answer_parse_mode = parse_mode or f"rejected_{reject_reason}"
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

    def _compute_final_reward(self, final_response: str, ground_truth: str) -> float:
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
            result = compute_score(
                data_source="time_series",
                solution_str=final_response,
                ground_truth=ground_truth,
                allow_recovery=False,
            )
            return float(result["score"] if isinstance(result, dict) else result)
        except Exception as e:
            logger.error(f"Error computing final reward: {e}")
            return -0.5

    def _canonical_answer_from_prediction_text(self, prediction_text: str) -> str:
        expected = self._expected_prediction_count()
        forecast_block = extract_forecast_block(prediction_text or "")
        candidate_text = forecast_block if forecast_block is not None else str(prediction_text or "")
        lines = normalized_nonempty_lines(candidate_text)
        if len(lines) < expected:
            raise ValueError(f"prediction_text has {len(lines)} forecast lines, expected at least {expected}")
        return "\n".join(lines[:expected])

    def _compute_selected_forecast_reward(self, ground_truth: str) -> float:
        if not self.prediction_results or not ground_truth:
            return 0.0
        try:
            selected_answer = self._canonical_answer_from_prediction_text(self.prediction_results)
            result = compute_score(
                data_source="time_series",
                solution_str=self._wrap_final_protocol(
                    selected_answer,
                    "I keep the selected forecast as the final answer because it is already consistent.",
                ),
                ground_truth=ground_truth,
            )
            score = float(result["score"] if isinstance(result, dict) else result)
            return 0.25 * score
        except Exception as exc:
            logger.warning("Error computing selected forecast reward: %s", exc)
            return 0.0

    @staticmethod
    def _wrap_final_protocol(answer_text: str, reflection_text: str) -> str:
        return f"<think>\n{reflection_text.strip()}\n</think>\n<answer>\n{answer_text}\n</answer>"

    def _compute_intermediate_reward(
        self,
        *,
        turn_stage: str,
        executed_tool_names: list[str],
        ground_truth: str,
    ) -> float:
        return 0.0

    def _compute_series_metrics(
        self,
        candidate_values: list[float],
        reference_values: list[float],
    ) -> tuple[float, float, float, float]:
        return compute_series_metrics(
            candidate_values,
            reference_values,
            normalize_for_reward_fn=normalize_for_reward,
        )

    def _collect_refinement_metrics(self, ground_truth: str) -> dict[str, Any]:
        return collect_refinement_metrics(
            ground_truth=ground_truth,
            prediction_results=self.prediction_results,
            final_answer=self.final_answer,
            extract_values_fn=extract_values_from_time_series_string,
            extract_ground_truth_values_fn=extract_ground_truth_values,
            normalize_for_reward_fn=normalize_for_reward,
        )

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
        turn_stage = self._current_turn_stage()
        prediction_payload = getattr(self, "prediction_tool_output", None) or self.prediction_results
        if turn_stage == "refinement" and self.prediction_results:
            try:
                # Present a clean horizon-length base forecast in Turn 3 so the
                # model can copy/refine the selected prediction without parsing
                # tool headers or timestamps again.
                prediction_payload = self._canonical_answer_from_prediction_text(self.prediction_results)
            except ValueError:
                prediction_payload = self.prediction_results

        return build_runtime_user_prompt(
            data_source=self.data_source,
            target_column=self.target_column,
            lookback_window=self.lookback_window,
            forecast_horizon=self.forecast_horizon,
            time_series_data=self.time_series_data,
            history_analysis=self.history_analysis,
            prediction_results=prediction_payload,
            prediction_model_used=getattr(self, "prediction_model_used", None),
            available_feature_tools=self._available_diagnostic_tool_names(),
            completed_feature_tools=self._executed_feature_tool_names(),
            turn_stage=turn_stage,
        )

    async def _execute_tool_call(self, tool_call: FunctionCall, **kwargs) -> Optional[str]:
        turn_stage = str(kwargs.get("turn_stage") or self._current_turn_stage()).strip().lower()
        if self.prediction_results is not None:
            self.illegal_turn3_tool_call_count += 1
            logger.warning("Tool call %s attempted after prediction stage; rejected.", tool_call.name)
            return None

        if tool_call.name == "predict_time_series":
            requested_model_name = "__missing__"
            if hasattr(tool_call, "arguments") and tool_call.arguments:
                args = tool_call.arguments
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {}
                if isinstance(args, dict):
                    requested_model_name = str(args.get("model_name") or "__missing__").strip().lower()

            self.prediction_requested_model = requested_model_name
            if turn_stage == "diagnostic":
                logger.warning("predict_time_series called during diagnostic turn, rejected until next turn")
                return None

            if not self._has_diagnostic_analysis():
                logger.warning(
                    "predict_time_series called before any diagnostic feature tool was completed",
                )
                return None

            model_name = requested_model_name
            valid_model_names = {"chronos2", "arima", "patchtst", "itransformer"}
            if model_name not in valid_model_names:
                self.prediction_model_defaulted = False
                self.prediction_tool_error = (
                    f"Invalid model_name: {requested_model_name}. "
                    f"Supported models: {sorted(valid_model_names)}"
                )
                logger.warning("predict_time_series called with invalid model_name=%s", requested_model_name)
                append_chain_debug(
                    "prediction_tool_result",
                    build_prediction_tool_debug_payload(
                        model_name=model_name,
                        prediction_requested_model=self.prediction_requested_model or "",
                        prediction_model_defaulted=False,
                        prediction_attempt_count=int(self.prediction_attempt_count),
                        prediction_step_index=self.prediction_step_index,
                        prediction_call_count=int(self.prediction_call_count),
                        analysis_state_signature_value=self._analysis_state_signature(),
                        feature_tool_signature_value=self._feature_tool_signature(),
                        forecast_horizon=self.forecast_horizon,
                        prediction_results="",
                        extract_values_fn=extract_values_from_time_series_string,
                        success=False,
                        error=self.prediction_tool_error,
                    ),
                )
                return None
            self.prediction_model_defaulted = False
            if int(self.prediction_attempt_count) >= self._max_prediction_attempts():
                logger.warning(
                    "predict_time_series attempted after retry budget exhausted; max_prediction_attempts=%s",
                    self._max_prediction_attempts(),
                )
                return None
            self.prediction_attempt_count += 1
            self.prediction_call_count += 1
            self.prediction_step_index = len(self.steps) + 1
            self.prediction_turn_stage = turn_stage
            tool_output = await self._run_prediction_tool(model_name=model_name)
            if tool_output is None:
                self.prediction_call_count = max(0, int(self.prediction_call_count) - 1)
                self.prediction_step_index = None
                self.prediction_turn_stage = None
                return None
            return tool_output

        if turn_stage == "routing":
            logger.warning("Feature tool %s attempted during routing turn; predict_time_series required instead.", tool_call.name)
            return None

        if self._feature_tool_already_executed(tool_call.name):
            logger.warning("Feature tool %s attempted after it was already executed; rejected.", tool_call.name)
            return None

        if tool_call.name in FEATURE_TOOL_SPECS:
            return self._run_feature_tool(tool_call.name)
        return None

    def _feature_tool_already_executed(self, tool_name: str) -> bool:
        spec = FEATURE_TOOL_SPECS.get(tool_name)
        if spec is None:
            return False
        return getattr(self, spec.state_attr) is not None

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
                include_covariates=True,
            )

            last_ts = get_last_timestamp(self.time_series_data)

            pred_df = await predict_time_series_async(
                context_df,
                prediction_length=self.forecast_horizon,
                model_name=model_name,
            )

            self.prediction_model_used = model_name
            self.prediction_tool_error = None
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
            append_chain_debug(
                "prediction_tool_result",
                build_prediction_tool_debug_payload(
                    model_name=model_name,
                    prediction_requested_model=self.prediction_requested_model or "",
                    prediction_model_defaulted=bool(self.prediction_model_defaulted),
                    prediction_attempt_count=int(self.prediction_attempt_count),
                    prediction_step_index=self.prediction_step_index,
                    prediction_call_count=int(self.prediction_call_count),
                    analysis_state_signature_value=self._analysis_state_signature(),
                    feature_tool_signature_value=self._feature_tool_signature(),
                    forecast_horizon=self.forecast_horizon,
                    prediction_results=self.prediction_results or "",
                    extract_values_fn=extract_values_from_time_series_string,
                    success=True,
                    error="",
                ),
            )

            return self.prediction_tool_output

        except Exception as e:
            logger.error(f"Error in predict with {model_name}: {e}")
            self.prediction_tool_error = f"{type(e).__name__}: {e}"
            self.prediction_results = None
            self.prediction_tool_output = None
            self.history_analysis.append(f"Prediction failed ({model_name}): {str(e)}")
            append_chain_debug(
                "prediction_tool_result",
                build_prediction_tool_debug_payload(
                    model_name=model_name,
                    prediction_requested_model=self.prediction_requested_model or "",
                    prediction_model_defaulted=bool(self.prediction_model_defaulted),
                    prediction_attempt_count=int(self.prediction_attempt_count),
                    prediction_step_index=self.prediction_step_index,
                    prediction_call_count=int(self.prediction_call_count),
                    analysis_state_signature_value=self._analysis_state_signature(),
                    feature_tool_signature_value=self._feature_tool_signature(),
                    forecast_horizon=self.forecast_horizon,
                    prediction_results="",
                    extract_values_fn=extract_values_from_time_series_string,
                    success=False,
                    error=self.prediction_tool_error,
                ),
            )
            return None

    async def extract_basic_statistics(self, **kwargs) -> float:
        """Extract core statistical features from time series data."""
        self._run_feature_tool("extract_basic_statistics")
        return 0.0

    def _run_basic_statistics_tool(self) -> Optional[str]:
        return self._run_feature_tool("extract_basic_statistics")

    async def extract_within_channel_dynamics(self, **kwargs) -> float:
        """Extract within-channel dynamics features from time series data."""
        self._run_feature_tool("extract_within_channel_dynamics")
        return 0.0

    def _run_within_channel_dynamics_tool(self) -> Optional[str]:
        return self._run_feature_tool("extract_within_channel_dynamics")

    async def extract_forecast_residuals(self, **kwargs) -> float:
        """Extract forecast residual features from time series data."""
        self._run_feature_tool("extract_forecast_residuals")
        return 0.0

    def _run_forecast_residuals_tool(self) -> Optional[str]:
        return self._run_feature_tool("extract_forecast_residuals")

    async def extract_data_quality(self, **kwargs) -> float:
        """Extract data quality features from time series data."""
        self._run_feature_tool("extract_data_quality")
        return 0.0

    def _run_data_quality_tool(self) -> Optional[str]:
        return self._run_feature_tool("extract_data_quality")

    async def extract_event_summary(self, **kwargs) -> float:
        """Extract event summary features from time series data."""
        self._run_feature_tool("extract_event_summary")
        return 0.0

    def _run_event_summary_tool(self) -> Optional[str]:
        return self._run_feature_tool("extract_event_summary")

    def _run_feature_tool(self, tool_name: str) -> Optional[str]:
        spec = FEATURE_TOOL_SPECS.get(tool_name)
        if spec is None:
            logger.warning("Unsupported feature tool %s", tool_name)
            return None
        try:
            if not self.values or len(self.values) < 2:
                logger.warning("Insufficient data for %s extraction", tool_name)
                return None

            features = spec.extractor(data=self.values)
            setattr(self, spec.state_attr, features)
            analysis_record = spec.formatter(features)
            self.history_analysis.append(analysis_record)
            self.feature_tool_sequence.append(tool_name)
            logger.info(spec.success_log)
            return analysis_record
        except Exception as e:
            logger.error("Error in %s: %s", tool_name, e)
            return None
