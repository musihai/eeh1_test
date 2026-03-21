# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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
"""
PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import gc
import json
import os
import random
import re
import uuid
from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
import math
from functools import reduce
from pprint import pprint
from typing import Optional

import numpy as np
import ray
import torch
from omegaconf import OmegaConf, open_dict
from torch.utils.data import Dataset, Sampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm

from verl import DataProto
from verl.experimental.dataset.sampler import AbstractCurriculumSampler
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.config import AlgoConfig
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.core_algos import AdvantageEstimator, agg_loss
from arft.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    process_validation_metrics,
)
from verl.trainer.ppo.reward import extract_reward
from verl.trainer.ppo.utils import Role, WorkerType, need_critic, need_reference_policy, need_reward_model
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path, should_save_ckpt_esi
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.debug import marked_timer
from verl.utils.metric import reduce_metrics
from verl.utils.rollout_skip import RolloutSkip
from verl.utils.seqlen_balancing import calculate_workload, get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.utils.torch_functional import masked_mean
from verl.utils.tracking import ValidationGenerationsLogger
from verl.utils.ray_utils import auto_await
from verl.utils.chain_debug import append_chain_debug
from verl.trainer.ppo.ray_trainer import (
    RayPPOTrainer,
    RayWorkerGroup,
    ResourcePoolManager,
    Role,
    WorkerType,
    apply_kl_penalty,
    compute_data_metrics,
    compute_response_mask,
    compute_timing_metrics,
    marked_timer,
    reduce_metrics,
)


def evaluate_validation_reward_manager(val_reward_fn, batch: DataProto) -> dict[str, torch.Tensor | dict[str, list]]:
    """Run validation reward for both legacy callable and new reward-loop managers."""
    if val_reward_fn is None:
        raise ValueError("val_reward_fn must be provided for validation.")

    if callable(val_reward_fn):
        return val_reward_fn(batch, return_dict=True)

    run_single = getattr(val_reward_fn, "run_single", None)
    if run_single is None:
        raise TypeError(f"Unsupported val_reward_fn type: {type(val_reward_fn)!r}")

    run_single_sync = auto_await(run_single)
    reward_tensor = torch.zeros_like(batch.batch["responses"], dtype=torch.float32)
    reward_extra_info: dict[str, list] = defaultdict(list)

    for sample_idx in range(len(batch)):
        sample_batch = batch.select_idxs([sample_idx])
        result = run_single_sync(sample_batch)
        if not isinstance(result, dict) or "reward_score" not in result:
            raise TypeError(
                "Reward manager run_single must return a dict containing 'reward_score'. "
                f"Got: {type(result)!r}"
            )

        reward_score = float(result["reward_score"])
        response_length = sample_batch.batch["responses"].shape[-1]
        valid_response_length = response_length

        if "attention_mask" in sample_batch.batch.keys():
            attention_mask = sample_batch.batch["attention_mask"][0]
            if "prompts" in sample_batch.batch.keys():
                prompt_length = sample_batch.batch["prompts"].shape[-1]
                valid_response_length = int(attention_mask[prompt_length:].sum().item())
            else:
                valid_response_length = int(attention_mask[-response_length:].sum().item())

        valid_response_length = max(1, min(valid_response_length, response_length))
        reward_tensor[sample_idx, valid_response_length - 1] = reward_score

        sample_extra = result.get("reward_extra_info", {}) or {}
        for key in list(reward_extra_info.keys()):
            if key not in sample_extra:
                reward_extra_info[key].append(None)
        for key, value in sample_extra.items():
            if key not in reward_extra_info:
                reward_extra_info[key] = [None] * sample_idx
            reward_extra_info[key].append(value)

    return {
        "reward_tensor": reward_tensor,
        "reward_extra_info": reward_extra_info,
    }


def compute_advantage(
    data: DataProto,
    adv_estimator: AdvantageEstimator,
    gamma: float = 1.0,
    lam: float = 1.0,
    num_repeat: int = 1,
    norm_adv_by_std_in_grpo: bool = True,
    config: Optional[AlgoConfig] = None,
) -> DataProto:
    """Compute advantage estimates for policy optimization.

    This function computes advantage estimates using various estimators like GAE, GRPO, REINFORCE++, etc.
    The advantage estimates are used to guide policy optimization in RL algorithms.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        adv_estimator (AdvantageEstimator): The advantage estimator to use (e.g., GAE, GRPO, REINFORCE++).
        gamma (float, optional): Discount factor for future rewards. Defaults to 1.0.
        lam (float, optional): Lambda parameter for GAE. Defaults to 1.0.
        num_repeat (int, optional): Number of times to repeat the computation. Defaults to 1.
        norm_adv_by_std_in_grpo (bool, optional): Whether to normalize advantages by standard deviation in
            GRPO. Defaults to True.
        config (dict, optional): Configuration dictionary for algorithm settings. Defaults to None.

    Returns:
        DataProto: The updated data with computed advantages and returns.
    """
    # Back-compatible with trainers that do not compute response mask in fit
    if "response_mask" not in data.batch.keys():
        data.batch["response_mask"] = compute_response_mask(data)
    advantages = torch.zeros_like(data.batch["token_level_rewards"])
    returns = torch.zeros_like(data.batch["token_level_rewards"])
    is_pad = data.non_tensor_batch.get("is_pad", None)
    if is_pad is not None:
        valid_mask = ~is_pad
        valid_data = data.select_idxs(valid_mask)
    else:
        valid_data = data
    # prepare response group
    if adv_estimator == AdvantageEstimator.GAE:
        # Compute advantages and returns using Generalized Advantage Estimation (GAE)
        from arft.core_algos import compute_gae_advantage_return
        valid_advantages, valid_returns = compute_gae_advantage_return(
            token_level_rewards=valid_data.batch["token_level_rewards"],
            values=valid_data.batch["values"],
            response_mask=valid_data.batch["response_mask"],
            trajectory_uids=valid_data.non_tensor_batch["trajectory_uids"],
            step_indices=valid_data.non_tensor_batch["step_indices"],
            gamma=gamma,
            lam=lam,
        )
        advantages[valid_mask] = valid_advantages
        returns[valid_mask] = valid_returns
    elif adv_estimator == AdvantageEstimator.GRPO:
        # Call compute_grpo_outcome_advantage with parameters matching its definition
        from arft.core_algos import compute_grpo_outcome_advantage
        valid_advantages, valid_returns = compute_grpo_outcome_advantage(
            token_level_rewards=valid_data.batch["token_level_rewards"],
            response_mask=valid_data.batch["response_mask"],
            index=valid_data.non_tensor_batch["uid"],
            trajectory_uids=valid_data.non_tensor_batch["trajectory_uids"],
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        )
        advantages[valid_mask] = valid_advantages
        returns[valid_mask] = valid_returns

    data.batch["advantages"] = advantages
    data.batch["returns"] = returns
    return data


class RayAgentTrainer(RayPPOTrainer):
    """Distributed PPO trainer using Ray for scalable reinforcement learning.

    This trainer orchestrates distributed PPO training across multiple nodes and GPUs,
    managing actor rollouts, critic training, and reward computation with Ray backend.
    Supports various model architectures including FSDP, Megatron, vLLM, and SGLang integration.
    """

    def __init__(self, *args, reward_fn=None, val_reward_fn=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

    @staticmethod
    def _to_float_list(values, n: int, default: float = float("nan")) -> list[float]:
        if values is None:
            return [default] * n
        out: list[float] = []
        for i in range(n):
            try:
                out.append(float(values[i]))
            except Exception:
                out.append(default)
        return out

    @staticmethod
    def _to_bool_list(values, n: int, default: bool = False) -> list[bool]:
        if values is None:
            return [default] * n
        out: list[bool] = []
        for i in range(n):
            try:
                value = values[i]
                if isinstance(value, str):
                    out.append(value.strip().lower() in {"1", "true", "yes", "on"})
                else:
                    out.append(bool(value))
            except Exception:
                out.append(default)
        return out

    @staticmethod
    def _to_str_list(values, n: int, default: str = "") -> list[str]:
        if values is None:
            return [default] * n
        out: list[str] = []
        for i in range(n):
            try:
                value = values[i]
                if value is None:
                    out.append(default)
                else:
                    out.append(str(value))
            except Exception:
                out.append(default)
        return out

    @staticmethod
    def _percentile(values: list[float], q: float) -> float:
        finite = [float(v) for v in values if isinstance(v, (int, float)) and np.isfinite(v)]
        if not finite:
            return float("nan")
        return float(np.percentile(np.asarray(finite, dtype=np.float64), q))

    @staticmethod
    def _extract_values_from_text(text: str) -> list[float]:
        values: list[float] = []
        if not text:
            return values
        answer_match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
        body = answer_match.group(1) if answer_match else text
        for line in str(body).splitlines():
            line = line.strip()
            if not line:
                continue
            match = re.search(r"(-?\d+\.?\d*)$", line)
            if not match:
                continue
            try:
                values.append(float(match.group(1)))
            except Exception:
                continue
        return values

    @staticmethod
    def _tail_lines(text: str, max_lines: int = 10) -> list[str]:
        if not text:
            return []
        return str(text).splitlines()[-max_lines:]

    @staticmethod
    def _normalized_mse_mae(pred_values: list[float], gt_values: list[float]) -> tuple[float, float]:
        if not pred_values or not gt_values:
            return float("nan"), float("nan")
        n = min(len(pred_values), len(gt_values))
        if n <= 0:
            return float("nan"), float("nan")
        pred = np.asarray(pred_values[:n], dtype=np.float64)
        gt = np.asarray(gt_values[:n], dtype=np.float64)
        mu = float(np.nanmean(gt))
        std = max(float(np.nanstd(gt)), 1e-8)
        pred_n = (pred - mu) / std
        gt_n = (gt - mu) / std
        diff = pred_n - gt_n
        mse = float(np.mean(diff ** 2))
        mae = float(np.mean(np.abs(diff)))
        return mse, mae

    @staticmethod
    def _orig_mse_mae(pred_values: list[float], gt_values: list[float]) -> tuple[float, float]:
        if not pred_values or not gt_values:
            return float("nan"), float("nan")
        n = min(len(pred_values), len(gt_values))
        if n <= 0:
            return float("nan"), float("nan")
        pred = np.asarray(pred_values[:n], dtype=np.float64)
        gt = np.asarray(gt_values[:n], dtype=np.float64)
        diff = pred - gt
        mse = float(np.mean(diff ** 2))
        mae = float(np.mean(np.abs(diff)))
        return mse, mae

    def _write_min_eval_debug_files(
        self,
        *,
        sample_uids: list,
        sample_outputs: list,
        sample_gts: list,
        sample_scores: list[float],
        reward_extra_infos_dict: dict[str, list],
    ) -> None:
        n = len(sample_scores)
        if n <= 0:
            return

        pred_len = self._to_float_list(reward_extra_infos_dict.get("pred_len"), n)
        expected_len = self._to_float_list(
            reward_extra_infos_dict.get("expected_len") or reward_extra_infos_dict.get("gt_len"),
            n,
        )
        orig_mse = self._to_float_list(
            reward_extra_infos_dict.get("orig_mse") or reward_extra_infos_dict.get("raw_mse"),
            n,
        )
        orig_mae = self._to_float_list(
            reward_extra_infos_dict.get("orig_mae") or reward_extra_infos_dict.get("raw_mae"),
            n,
        )
        norm_mse = self._to_float_list(reward_extra_infos_dict.get("norm_mse"), n)
        norm_mae = self._to_float_list(reward_extra_infos_dict.get("norm_mae"), n)
        has_answer_tag = self._to_bool_list(reward_extra_infos_dict.get("has_answer_tag"), n)
        has_answer_close = self._to_bool_list(reward_extra_infos_dict.get("has_answer_close"), n)
        was_clipped = self._to_bool_list(reward_extra_infos_dict.get("was_clipped"), n)
        format_failure_reason = self._to_str_list(reward_extra_infos_dict.get("format_failure_reason"), n)
        final_answer_reject_reason = self._to_str_list(
            reward_extra_infos_dict.get("final_answer_reject_reason"),
            n,
        )
        length_hard_fail = self._to_float_list(reward_extra_infos_dict.get("length_hard_fail"), n, default=0.0)
        strict_length_match = self._to_bool_list(reward_extra_infos_dict.get("strict_length_match"), n)
        trainer_seq_score = self._to_float_list(
            reward_extra_infos_dict.get("trainer_seq_score") or reward_extra_infos_dict.get("score"),
            n,
            default=float("nan"),
        )
        generation_stop_reason = self._to_str_list(reward_extra_infos_dict.get("generation_stop_reason"), n)
        selected_model = self._to_str_list(
            reward_extra_infos_dict.get("selected_model")
            or reward_extra_infos_dict.get("prediction_model_used")
            or reward_extra_infos_dict.get("output_source"),
            n,
            default="unknown",
        )
        reward_main_scale = self._to_str_list(reward_extra_infos_dict.get("reward_main_scale"), n)
        selected_forecast_orig_mse = self._to_float_list(
            reward_extra_infos_dict.get("selected_forecast_orig_mse"),
            n,
            default=float("nan"),
        )
        selected_forecast_len_match = self._to_bool_list(
            reward_extra_infos_dict.get("selected_forecast_len_match"),
            n,
        )
        selected_forecast_exact_copy = self._to_bool_list(
            reward_extra_infos_dict.get("selected_forecast_exact_copy"),
            n,
        )
        final_vs_selected_mse = self._to_float_list(
            reward_extra_infos_dict.get("final_vs_selected_mse"),
            n,
            default=float("nan"),
        )
        refinement_delta_orig_mse = self._to_float_list(
            reward_extra_infos_dict.get("refinement_delta_orig_mse"),
            n,
            default=float("nan"),
        )
        refinement_compare_len = self._to_float_list(
            reward_extra_infos_dict.get("refinement_compare_len"),
            n,
            default=float("nan"),
        )
        refinement_changed_value_count = self._to_float_list(
            reward_extra_infos_dict.get("refinement_changed_value_count"),
            n,
            default=float("nan"),
        )
        refinement_first_changed_index = self._to_float_list(
            reward_extra_infos_dict.get("refinement_first_changed_index"),
            n,
            default=float("nan"),
        )
        refinement_change_mean_abs = self._to_float_list(
            reward_extra_infos_dict.get("refinement_change_mean_abs"),
            n,
            default=float("nan"),
        )
        refinement_change_max_abs = self._to_float_list(
            reward_extra_infos_dict.get("refinement_change_max_abs"),
            n,
            default=float("nan"),
        )
        refinement_changed = self._to_bool_list(reward_extra_infos_dict.get("refinement_changed"), n)
        refinement_improved = self._to_bool_list(reward_extra_infos_dict.get("refinement_improved"), n)
        refinement_degraded = self._to_bool_list(reward_extra_infos_dict.get("refinement_degraded"), n)
        analysis_coverage_ratio = self._to_float_list(
            reward_extra_infos_dict.get("analysis_coverage_ratio"),
            n,
            default=float("nan"),
        )
        feature_tool_count = self._to_float_list(
            reward_extra_infos_dict.get("feature_tool_count"),
            n,
            default=float("nan"),
        )
        prediction_call_count = self._to_float_list(
            reward_extra_infos_dict.get("prediction_call_count"),
            n,
            default=float("nan"),
        )
        tool_call_count = self._to_float_list(
            reward_extra_infos_dict.get("tool_call_count"),
            n,
            default=float("nan"),
        )
        history_analysis_count = self._to_float_list(
            reward_extra_infos_dict.get("history_analysis_count"),
            n,
            default=float("nan"),
        )
        illegal_turn3_tool_call_count = self._to_float_list(
            reward_extra_infos_dict.get("illegal_turn3_tool_call_count"),
            n,
            default=float("nan"),
        )
        prediction_model_defaulted = self._to_bool_list(
            reward_extra_infos_dict.get("prediction_model_defaulted"),
            n,
        )
        prediction_requested_model = self._to_str_list(
            reward_extra_infos_dict.get("prediction_requested_model"),
            n,
        )
        feature_tool_signature = self._to_str_list(
            reward_extra_infos_dict.get("feature_tool_signature"),
            n,
            default="none",
        )
        tool_call_sequence = self._to_str_list(
            reward_extra_infos_dict.get("tool_call_sequence"),
            n,
            default="none",
        )
        analysis_state_signature = self._to_str_list(
            reward_extra_infos_dict.get("analysis_state_signature"),
            n,
            default="none",
        )
        workflow_status = self._to_str_list(
            reward_extra_infos_dict.get("workflow_status"),
            n,
        )
        turn_stage = self._to_str_list(
            reward_extra_infos_dict.get("turn_stage"),
            n,
        )
        prediction_tool_error = self._to_str_list(
            reward_extra_infos_dict.get("prediction_tool_error"),
            n,
        )
        selected_forecast_preview = self._to_str_list(
            reward_extra_infos_dict.get("selected_forecast_preview"),
            n,
        )
        final_answer_preview = self._to_str_list(
            reward_extra_infos_dict.get("final_answer_preview"),
            n,
        )

        pred_len_arr = np.asarray(pred_len, dtype=np.float64)
        expected_len_arr = np.asarray(expected_len, dtype=np.float64)
        valid_pred_len = np.isfinite(pred_len_arr)
        valid_expected_len = np.isfinite(expected_len_arr)

        is_96 = (pred_len_arr == 96)
        is_94_95 = np.logical_or(pred_len_arr == 94, pred_len_arr == 95)
        is_lt_96 = pred_len_arr < 96
        is_gt_96 = pred_len_arr > 96
        missing_close = np.asarray([reason == "missing_answer_close_tag" for reason in format_failure_reason], dtype=bool)
        invalid_answer_shape = np.asarray(
            [reason.startswith("invalid_answer_shape") for reason in final_answer_reject_reason],
            dtype=bool,
        )
        strict_length_match_arr = np.asarray(strict_length_match, dtype=bool)
        exact_expected_match = np.logical_and.reduce(
            [
                valid_pred_len,
                valid_expected_len,
                pred_len_arr == expected_len_arr,
            ]
        )
        final_answer_accept = np.logical_and.reduce(
            [
                np.asarray(has_answer_tag, dtype=bool),
                np.asarray(has_answer_close, dtype=bool),
                exact_expected_match,
                np.asarray([reason == "" for reason in final_answer_reject_reason], dtype=bool),
            ]
        )

        success_mask = np.logical_and.reduce(
            [
                final_answer_accept,
                np.asarray([np.isfinite(v) for v in orig_mse], dtype=bool),
                np.asarray([np.isfinite(v) for v in orig_mae], dtype=bool),
            ]
        )

        success_orig_mse_values = [orig_mse[i] for i in range(n) if success_mask[i] and np.isfinite(orig_mse[i])]
        success_orig_mae_values = [orig_mae[i] for i in range(n) if success_mask[i] and np.isfinite(orig_mae[i])]
        success_norm_mse_values = [norm_mse[i] for i in range(n) if success_mask[i] and np.isfinite(norm_mse[i])]
        success_norm_mae_values = [norm_mae[i] for i in range(n) if success_mask[i] and np.isfinite(norm_mae[i])]
        reward_values = [float(v) for v in sample_scores if isinstance(v, (int, float)) and np.isfinite(v)]
        length_hard_fail_values = [float(v) for v in length_hard_fail if np.isfinite(v)]
        trainer_seq_values = [float(v) for v in trainer_seq_score if np.isfinite(v)]
        final_answer_reject_counter = Counter(reason for reason in final_answer_reject_reason if reason)
        format_failure_reason_counter = Counter(reason for reason in format_failure_reason if reason)
        generation_stop_reason_counter = Counter(reason for reason in generation_stop_reason if reason)
        selected_model_counter = Counter(model for model in selected_model if model)
        reward_main_scale_counter = Counter(scale for scale in reward_main_scale if scale)
        feature_tool_signature_counter = Counter(signature for signature in feature_tool_signature if signature)
        tool_call_sequence_counter = Counter((signature if signature else "none") for signature in tool_call_sequence)
        analysis_state_signature_counter = Counter(signature for signature in analysis_state_signature if signature)
        prediction_requested_model_counter = Counter(model for model in prediction_requested_model if model)
        workflow_status_counter = Counter(status for status in workflow_status if status)
        turn_stage_counter = Counter(stage for stage in turn_stage if stage)
        prediction_tool_error_count = int(sum(1 for value in prediction_tool_error if value))

        selected_forecast_orig_mse_values = [
            float(v) for v in selected_forecast_orig_mse if isinstance(v, (int, float)) and np.isfinite(v)
        ]
        final_vs_selected_mse_values = [
            float(v) for v in final_vs_selected_mse if isinstance(v, (int, float)) and np.isfinite(v)
        ]
        refinement_delta_orig_mse_values = [
            float(v) for v in refinement_delta_orig_mse if isinstance(v, (int, float)) and np.isfinite(v)
        ]
        refinement_compare_len_values = [
            float(v) for v in refinement_compare_len if isinstance(v, (int, float)) and np.isfinite(v)
        ]
        refinement_changed_value_count_values = [
            float(v) for v in refinement_changed_value_count if isinstance(v, (int, float)) and np.isfinite(v)
        ]
        refinement_first_changed_index_values = [
            float(v) for v in refinement_first_changed_index if isinstance(v, (int, float)) and np.isfinite(v) and v >= 0
        ]
        refinement_change_mean_abs_values = [
            float(v) for v in refinement_change_mean_abs if isinstance(v, (int, float)) and np.isfinite(v)
        ]
        refinement_change_max_abs_values = [
            float(v) for v in refinement_change_max_abs if isinstance(v, (int, float)) and np.isfinite(v)
        ]
        analysis_coverage_ratio_values = [
            float(v) for v in analysis_coverage_ratio if isinstance(v, (int, float)) and np.isfinite(v)
        ]
        feature_tool_count_values = [
            float(v) for v in feature_tool_count if isinstance(v, (int, float)) and np.isfinite(v)
        ]
        prediction_call_count_values = [
            float(v) for v in prediction_call_count if isinstance(v, (int, float)) and np.isfinite(v)
        ]
        tool_call_count_values = [
            float(v) for v in tool_call_count if isinstance(v, (int, float)) and np.isfinite(v)
        ]
        history_analysis_count_values = [
            float(v) for v in history_analysis_count if isinstance(v, (int, float)) and np.isfinite(v)
        ]
        illegal_turn3_tool_call_count_values = [
            float(v) for v in illegal_turn3_tool_call_count if isinstance(v, (int, float)) and np.isfinite(v)
        ]

        total = float(n)
        known_model_shares = {
            "patchtst_share": float(selected_model_counter.get("patchtst", 0) / total),
            "itransformer_share": float(selected_model_counter.get("itransformer", 0) / total),
            "arima_share": float(selected_model_counter.get("arima", 0) / total),
            "chronos2_share": float(selected_model_counter.get("chronos2", 0) / total),
            "unknown_model_share": float(selected_model_counter.get("unknown", 0) / total),
        }
        agg_row = {
            "step": int(self.global_steps),
            "total_samples": int(n),
            "exact_96_ratio": float(np.sum(is_96) / total),
            "pred_len_96_ratio": float(np.sum(is_96) / total),
            "pred_len_94_95_ratio": float(np.sum(is_94_95) / total),
            "pred_len_lt_96_ratio": float(np.sum(np.logical_and(is_lt_96, valid_pred_len)) / total),
            "pred_len_gt_96_ratio": float(np.sum(np.logical_and(is_gt_96, valid_pred_len)) / total),
            "has_answer_tag_ratio": float(np.mean(np.asarray(has_answer_tag, dtype=np.float64))),
            "has_answer_close_ratio": float(np.mean(np.asarray(has_answer_close, dtype=np.float64))),
            "final_answer_accept_ratio": float(np.mean(final_answer_accept.astype(np.float64))),
            "strict_length_match_ratio": float(np.mean(strict_length_match_arr.astype(np.float64))),
            "missing_answer_close_tag_count": int(np.sum(missing_close)),
            "invalid_answer_shape_count": int(np.sum(invalid_answer_shape)),
            "was_clipped_count": int(np.sum(np.asarray(was_clipped, dtype=bool))),
            "orig_mse_mean": float(np.mean(success_orig_mse_values)) if success_orig_mse_values else float("nan"),
            "orig_mse_p50": self._percentile(success_orig_mse_values, 50),
            "orig_mse_p90": self._percentile(success_orig_mse_values, 90),
            "orig_mae_mean": float(np.mean(success_orig_mae_values)) if success_orig_mae_values else float("nan"),
            "orig_mae_p50": self._percentile(success_orig_mae_values, 50),
            "orig_mae_p90": self._percentile(success_orig_mae_values, 90),
            "norm_mse_mean": float(np.mean(success_norm_mse_values)) if success_norm_mse_values else float("nan"),
            "norm_mse_p50": self._percentile(success_norm_mse_values, 50),
            "norm_mse_p90": self._percentile(success_norm_mse_values, 90),
            "norm_mae_mean": float(np.mean(success_norm_mae_values)) if success_norm_mae_values else float("nan"),
            "norm_mae_p50": self._percentile(success_norm_mae_values, 50),
            "norm_mae_p90": self._percentile(success_norm_mae_values, 90),
            "success_raw_mse_mean": float(np.mean(success_orig_mse_values)) if success_orig_mse_values else float("nan"),
            "success_raw_mse_p50": self._percentile(success_orig_mse_values, 50),
            "success_raw_mse_p90": self._percentile(success_orig_mse_values, 90),
            "success_raw_mae_mean": float(np.mean(success_orig_mae_values)) if success_orig_mae_values else float("nan"),
            "success_raw_mae_p50": self._percentile(success_orig_mae_values, 50),
            "success_raw_mae_p90": self._percentile(success_orig_mae_values, 90),
            "validation_reward_mean": float(np.mean(reward_values)) if reward_values else float("nan"),
            "length_hard_fail_ratio": float(np.mean(length_hard_fail_values)) if length_hard_fail_values else float("nan"),
            "length_hard_fail_mean": float(np.mean(length_hard_fail_values)) if length_hard_fail_values else float("nan"),
            "trainer_seq_score_mean": float(np.mean(trainer_seq_values)) if trainer_seq_values else float("nan"),
            "selected_forecast_orig_mse_mean": float(np.mean(selected_forecast_orig_mse_values))
            if selected_forecast_orig_mse_values
            else float("nan"),
            "selected_forecast_len_match_ratio": float(
                np.mean(np.asarray(selected_forecast_len_match, dtype=np.float64))
            ),
            "selected_forecast_exact_copy_ratio": float(
                np.mean(np.asarray(selected_forecast_exact_copy, dtype=np.float64))
            ),
            "final_vs_selected_mse_mean": float(np.mean(final_vs_selected_mse_values))
            if final_vs_selected_mse_values
            else float("nan"),
            "refinement_delta_orig_mse_mean": float(np.mean(refinement_delta_orig_mse_values))
            if refinement_delta_orig_mse_values
            else float("nan"),
            "refinement_compare_len_mean": float(np.mean(refinement_compare_len_values))
            if refinement_compare_len_values
            else float("nan"),
            "refinement_changed_value_count_mean": float(np.mean(refinement_changed_value_count_values))
            if refinement_changed_value_count_values
            else float("nan"),
            "refinement_first_changed_index_mean": float(np.mean(refinement_first_changed_index_values))
            if refinement_first_changed_index_values
            else float("nan"),
            "refinement_change_mean_abs_mean": float(np.mean(refinement_change_mean_abs_values))
            if refinement_change_mean_abs_values
            else float("nan"),
            "refinement_change_max_abs_mean": float(np.mean(refinement_change_max_abs_values))
            if refinement_change_max_abs_values
            else float("nan"),
            "refinement_changed_ratio": float(np.mean(np.asarray(refinement_changed, dtype=np.float64))),
            "refinement_improved_ratio": float(np.mean(np.asarray(refinement_improved, dtype=np.float64))),
            "refinement_degraded_ratio": float(np.mean(np.asarray(refinement_degraded, dtype=np.float64))),
            "analysis_coverage_ratio_mean": float(np.mean(analysis_coverage_ratio_values))
            if analysis_coverage_ratio_values
            else float("nan"),
            "feature_tool_count_mean": float(np.mean(feature_tool_count_values))
            if feature_tool_count_values
            else float("nan"),
            "prediction_call_count_mean": float(np.mean(prediction_call_count_values))
            if prediction_call_count_values
            else float("nan"),
            "tool_call_count_mean": float(np.mean(tool_call_count_values)) if tool_call_count_values else float("nan"),
            "history_analysis_count_mean": float(np.mean(history_analysis_count_values))
            if history_analysis_count_values
            else float("nan"),
            "no_tool_call_ratio": float(
                np.mean((np.asarray(tool_call_count_values, dtype=np.float64) <= 0.0).astype(np.float64))
            )
            if tool_call_count_values
            else float("nan"),
            "no_history_analysis_ratio": float(
                np.mean((np.asarray(history_analysis_count_values, dtype=np.float64) <= 0.0).astype(np.float64))
            )
            if history_analysis_count_values
            else float("nan"),
            "illegal_turn3_tool_call_ratio": float(
                np.mean((np.asarray(illegal_turn3_tool_call_count_values, dtype=np.float64) > 0).astype(np.float64))
            )
            if illegal_turn3_tool_call_count_values
            else float("nan"),
            "prediction_model_defaulted_ratio": float(
                np.mean(np.asarray(prediction_model_defaulted, dtype=np.float64))
            ),
            "prediction_tool_error_count": prediction_tool_error_count,
            "selected_model_distribution": {str(k): int(v) for k, v in sorted(selected_model_counter.items())},
            "format_failure_reason_distribution": {str(k): int(v) for k, v in sorted(format_failure_reason_counter.items())},
            "final_answer_reject_reason_distribution": {
                str(k): int(v) for k, v in sorted(final_answer_reject_counter.items())
            },
            "generation_stop_reason_distribution": {
                str(k): int(v) for k, v in sorted(generation_stop_reason_counter.items())
            },
            "reward_main_scale_distribution": {str(k): int(v) for k, v in sorted(reward_main_scale_counter.items())},
            "feature_tool_signature_distribution": {
                str(k): int(v) for k, v in sorted(feature_tool_signature_counter.items())
            },
            "tool_call_sequence_distribution": {
                str(k): int(v) for k, v in sorted(tool_call_sequence_counter.items())
            },
            "analysis_state_signature_distribution": {
                str(k): int(v) for k, v in sorted(analysis_state_signature_counter.items())
            },
            "prediction_requested_model_distribution": {
                str(k): int(v) for k, v in sorted(prediction_requested_model_counter.items())
            },
            "workflow_status_distribution": {str(k): int(v) for k, v in sorted(workflow_status_counter.items())},
            "turn_stage_distribution": {str(k): int(v) for k, v in sorted(turn_stage_counter.items())},
        }
        agg_row.update(known_model_shares)

        all_indices = list(range(n))
        success_indices = [i for i in all_indices if success_mask[i]]
        near_miss_indices = [i for i in all_indices if is_94_95[i]]
        failure_indices = [i for i in all_indices if i not in set(success_indices)]

        rng = random.Random(int(self.global_steps) + 20260318)

        def _sample(indices: list[int], k: int) -> list[int]:
            if len(indices) <= k:
                return indices
            return rng.sample(indices, k)

        pick_success = _sample(success_indices, 10)
        pick_failure = _sample(failure_indices, 10)
        pick_nearmiss = _sample(near_miss_indices, 10)

        sample_rows: list[dict] = []

        def _build_base_row(i: int, category: str) -> dict:
            output_text = str(sample_outputs[i]) if i < len(sample_outputs) else ""
            return {
                "step": int(self.global_steps),
                "category": category,
                "sample_id": str(sample_uids[i]) if i < len(sample_uids) else f"sample_{i}",
                "selected_model": selected_model[i],
                "pred_len": int(pred_len_arr[i]) if np.isfinite(pred_len_arr[i]) else -1,
                "expected_len": int(expected_len_arr[i]) if np.isfinite(expected_len_arr[i]) else -1,
                "orig_mse": float(orig_mse[i]) if np.isfinite(orig_mse[i]) else float("nan"),
                "orig_mae": float(orig_mae[i]) if np.isfinite(orig_mae[i]) else float("nan"),
                "norm_mse": float(norm_mse[i]) if np.isfinite(norm_mse[i]) else float("nan"),
                "norm_mae": float(norm_mae[i]) if np.isfinite(norm_mae[i]) else float("nan"),
                "raw_mse": float(orig_mse[i]) if np.isfinite(orig_mse[i]) else float("nan"),
                "raw_mae": float(orig_mae[i]) if np.isfinite(orig_mae[i]) else float("nan"),
                "has_answer_tag": bool(has_answer_tag[i]),
                "has_answer_close": bool(has_answer_close[i]),
                "failure_reason": format_failure_reason[i] if format_failure_reason[i] else "",
                "final_answer_reject_reason": final_answer_reject_reason[i] if final_answer_reject_reason[i] else "",
                "generation_stop_reason": generation_stop_reason[i] if generation_stop_reason[i] else "",
                "strict_length_match": bool(strict_length_match_arr[i]),
                "length_hard_fail": bool(length_hard_fail[i]),
                "trainer_seq_score": float(trainer_seq_score[i]) if np.isfinite(trainer_seq_score[i]) else float("nan"),
                "selected_forecast_orig_mse": float(selected_forecast_orig_mse[i])
                if np.isfinite(selected_forecast_orig_mse[i])
                else float("nan"),
                "selected_forecast_len_match": bool(selected_forecast_len_match[i]),
                "selected_forecast_exact_copy": bool(selected_forecast_exact_copy[i]),
                "final_vs_selected_mse": float(final_vs_selected_mse[i])
                if np.isfinite(final_vs_selected_mse[i])
                else float("nan"),
                "refinement_delta_orig_mse": float(refinement_delta_orig_mse[i])
                if np.isfinite(refinement_delta_orig_mse[i])
                else float("nan"),
                "refinement_compare_len": int(refinement_compare_len[i])
                if np.isfinite(refinement_compare_len[i])
                else -1,
                "refinement_changed_value_count": int(refinement_changed_value_count[i])
                if np.isfinite(refinement_changed_value_count[i])
                else -1,
                "refinement_first_changed_index": int(refinement_first_changed_index[i])
                if np.isfinite(refinement_first_changed_index[i])
                else -1,
                "refinement_change_mean_abs": float(refinement_change_mean_abs[i])
                if np.isfinite(refinement_change_mean_abs[i])
                else float("nan"),
                "refinement_change_max_abs": float(refinement_change_max_abs[i])
                if np.isfinite(refinement_change_max_abs[i])
                else float("nan"),
                "refinement_changed": bool(refinement_changed[i]),
                "refinement_improved": bool(refinement_improved[i]),
                "refinement_degraded": bool(refinement_degraded[i]),
                "analysis_coverage_ratio": float(analysis_coverage_ratio[i])
                if np.isfinite(analysis_coverage_ratio[i])
                else float("nan"),
                "feature_tool_count": int(feature_tool_count[i]) if np.isfinite(feature_tool_count[i]) else -1,
                "prediction_call_count": int(prediction_call_count[i]) if np.isfinite(prediction_call_count[i]) else -1,
                "tool_call_count": int(tool_call_count[i]) if np.isfinite(tool_call_count[i]) else -1,
                "history_analysis_count": int(history_analysis_count[i])
                if np.isfinite(history_analysis_count[i])
                else -1,
                "illegal_turn3_tool_call_count": int(illegal_turn3_tool_call_count[i])
                if np.isfinite(illegal_turn3_tool_call_count[i])
                else -1,
                "prediction_requested_model": prediction_requested_model[i] if prediction_requested_model[i] else "",
                "prediction_model_defaulted": bool(prediction_model_defaulted[i]),
                "feature_tool_signature": feature_tool_signature[i] if feature_tool_signature[i] else "",
                "tool_call_sequence": tool_call_sequence[i] if tool_call_sequence[i] else "none",
                "analysis_state_signature": analysis_state_signature[i] if analysis_state_signature[i] else "",
                "workflow_status": workflow_status[i] if workflow_status[i] else "",
                "turn_stage": turn_stage[i] if turn_stage[i] else "",
                "prediction_tool_error": prediction_tool_error[i] if prediction_tool_error[i] else "",
                "selected_forecast_preview": selected_forecast_preview[i] if selected_forecast_preview[i] else "",
                "final_answer_preview": final_answer_preview[i] if final_answer_preview[i] else "",
                "raw_model_output_tail": self._tail_lines(output_text, 10),
            }

        for i in pick_success:
            row = _build_base_row(i, "success")
            output_text = str(sample_outputs[i]) if i < len(sample_outputs) else ""
            gt_text = str(sample_gts[i]) if i < len(sample_gts) else ""
            row["pred_values"] = self._extract_values_from_text(output_text)
            row["gt_values"] = self._extract_values_from_text(gt_text)
            sample_rows.append(row)

        for i in pick_failure:
            row = _build_base_row(i, "failure")
            sample_rows.append(row)

        for i in pick_nearmiss:
            row = _build_base_row(i, "near_miss_94_95")
            output_text = str(sample_outputs[i]) if i < len(sample_outputs) else ""
            gt_text = str(sample_gts[i]) if i < len(sample_gts) else ""
            pred_values = self._extract_values_from_text(output_text)
            gt_values = self._extract_values_from_text(gt_text)
            if pred_values and gt_values and len(pred_values) < len(gt_values):
                filled_pred = list(pred_values)
                filled_pred.extend([filled_pred[-1]] * (len(gt_values) - len(filled_pred)))
                filled_orig_mse, filled_orig_mae = self._orig_mse_mae(filled_pred, gt_values)
                filled_norm_mse, filled_norm_mae = self._normalized_mse_mae(filled_pred, gt_values)
            else:
                filled_orig_mse, filled_orig_mae = float("nan"), float("nan")
                filled_norm_mse, filled_norm_mae = float("nan"), float("nan")
            row["filled_orig_mse"] = float(filled_orig_mse)
            row["filled_orig_mae"] = float(filled_orig_mae)
            row["filled_norm_mse"] = float(filled_norm_mse)
            row["filled_norm_mae"] = float(filled_norm_mae)
            row["filled_raw_mse"] = float(filled_orig_mse)
            row["filled_raw_mae"] = float(filled_orig_mae)
            sample_rows.append(row)

        debug_dir = os.getenv("TS_MIN_DEBUG_DIR", os.path.join(os.getcwd(), "logs", "debug"))
        os.makedirs(debug_dir, exist_ok=True)
        agg_file = os.getenv("TS_MIN_EVAL_AGG_FILE", os.path.join(debug_dir, "eval_step_aggregate.jsonl"))
        sample_file = os.getenv("TS_MIN_EVAL_SAMPLE_FILE", os.path.join(debug_dir, "eval_step_samples.jsonl"))

        with open(agg_file, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(agg_row, ensure_ascii=False) + "\n")

        with open(sample_file, "a", encoding="utf-8") as handle:
            for row in sample_rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    def _dump_generations(self, inputs, outputs, gts, scores, reward_extra_infos_dict, dump_path):
        """Dump rollout/validation samples as JSONL."""
        os.makedirs(dump_path, exist_ok=True)
        filename = os.path.join(dump_path, f"{self.global_steps}.jsonl")

        n = len(inputs)
        base_data = {
            "input": inputs,
            "output": outputs,
            "gts": gts,
            "score": scores,
            "step": [self.global_steps] * n,
        }

        for k, v in reward_extra_infos_dict.items():
            if len(v) == n:
                base_data[k] = v

        lines = []
        for i in range(n):
            entry = {k: v[i] for k, v in base_data.items()}
            lines.append(json.dumps(entry, ensure_ascii=False))

        with open(filename, "w") as f:
            f.write("\n".join(lines) + "\n")

        print(f"Dumped generations to {filename}")

    def _log_rollout_data(
        self, batch: DataProto, reward_extra_infos_dict: dict, timing_raw: dict, rollout_data_dir: str
    ):
        """Log rollout data to disk.
        Args:
            batch (DataProto): The batch containing rollout data
            reward_extra_infos_dict (dict): Additional reward information to log
            timing_raw (dict): Timing information for profiling
            rollout_data_dir (str): Directory path to save the rollout data
        """
        with marked_timer("dump_rollout_generations", timing_raw, color="green"):
            inputs = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
            outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
            scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()
            sample_gts = [item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None) for item in batch]

            reward_extra_infos_to_dump = reward_extra_infos_dict.copy()
            if "request_id" in batch.non_tensor_batch:
                reward_extra_infos_dict.setdefault(
                    "request_id",
                    batch.non_tensor_batch["request_id"].tolist(),
                )

            self._dump_generations(
                inputs=inputs,
                outputs=outputs,
                gts=sample_gts,
                scores=scores,
                reward_extra_infos_dict=reward_extra_infos_to_dump,
                dump_path=rollout_data_dir,
            )

    def _maybe_log_val_generations(self, inputs, outputs, scores):
        """Log a table of validation samples to the configured logger (wandb or swanlab)"""

        generations_to_log = self.config.trainer.log_val_generations

        if generations_to_log == 0:
            return

        import numpy as np

        # Create tuples of (input, output, score) and sort by input text
        samples = list(zip(inputs, outputs, scores, strict=True))
        samples.sort(key=lambda x: x[0])  # Sort by input text

        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        # Take first N samples after shuffling
        samples = samples[:generations_to_log]

        # Log to each configured logger
        self.validation_generations_logger.log(self.config.trainer.logger, samples, self.global_steps)

    def _validate(self):
        data_source_lst = []
        reward_extra_infos_dict: dict[str, list] = defaultdict(list)

        # Lists to collect samples for the table
        sample_inputs = []
        sample_outputs = []
        sample_gts = []
        sample_scores = []
        sample_turns = []
        sample_uids = []

        def _extract_input_texts(batch: DataProto) -> list[str]:
            if "input_ids" in batch.batch.keys():
                input_ids = batch.batch["input_ids"]
                return [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            if "prompts" in batch.batch.keys():
                prompt_ids = batch.batch["prompts"]
                return [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in prompt_ids]

            raw_prompts = batch.non_tensor_batch.get("raw_prompt")
            if raw_prompts is not None:
                texts: list[str] = []
                for raw_prompt in raw_prompts:
                    if isinstance(raw_prompt, (list, tuple)):
                        texts.append(
                            "\n".join(
                                message.get("content", "")
                                for message in raw_prompt
                                if isinstance(message, dict)
                            ).strip()
                        )
                    else:
                        texts.append(str(raw_prompt))
                return texts

            return [""] * len(batch)

        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)

            if "uid" not in test_batch.non_tensor_batch:
                test_batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(test_batch.batch))], dtype=object
                )

            # repeat test batch
            test_batch = test_batch.repeat(
                repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n, interleave=True
            )

            # we only do validation on rule-based rm
            if self.config.reward_model.enable and test_batch[0].non_tensor_batch["reward_model"]["style"] == "model":
                return {}

            # Store original inputs
            input_texts = _extract_input_texts(test_batch)
            sample_inputs.extend(input_texts)
            sample_uids.extend(test_batch.non_tensor_batch["uid"])

            ground_truths = [
                item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None) for item in test_batch
            ]
            sample_gts.extend(ground_truths)

            test_gen_batch = self._get_gen_batch(test_batch)
            test_gen_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                "validate": True,
                "global_steps": self.global_steps,
            }
            print(f"test_gen_batch meta info: {test_gen_batch.meta_info}")

            test_output_gen_batch = self.async_rollout_manager.generate_sequences(test_gen_batch)

            # get indice for last step in each request
            # num_steps: [3,2,3] -> last_step_indice: [2,4,7]
            if "num_steps" in test_output_gen_batch.meta_info:
                num_steps = test_output_gen_batch.meta_info.pop("num_steps")
                last_step_indice = np.array(num_steps).cumsum() - 1
            else:
                last_step_indice = np.arange(len(test_output_gen_batch))

            # for validation, we only need the last step of each trajectory
            test_output_gen_batch = test_output_gen_batch.select_idxs(last_step_indice)

            print("validation generation end")

            # Store generated outputs
            output_ids = test_output_gen_batch.batch["responses"]
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            sample_outputs.extend(output_texts)

            test_batch = test_batch.union(test_output_gen_batch)
            test_batch.meta_info["validate"] = True

            # evaluate using reward_function
            result = evaluate_validation_reward_manager(self.val_reward_fn, test_batch)
            reward_tensor = result["reward_tensor"]
            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)

            reward_extra_infos_dict["reward"].extend(scores)

            if "reward_extra_info" in result:
                for key, lst in result["reward_extra_info"].items():
                    reward_extra_infos_dict[key].extend(lst)

            # collect num_turns of each prompt
            if "__num_turns__" in test_batch.non_tensor_batch:
                sample_turns.append(test_batch.non_tensor_batch["__num_turns__"])

            data_source_lst.append(test_batch.non_tensor_batch.get("data_source", ["unknown"] * reward_tensor.shape[0]))

        self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)

        # dump generations
        val_data_dir = self.config.trainer.get("validation_data_dir", None)
        if val_data_dir:
            self._dump_generations(
                inputs=sample_inputs,
                outputs=sample_outputs,
                gts=sample_gts,
                scores=sample_scores,
                reward_extra_infos_dict=reward_extra_infos_dict,
                dump_path=val_data_dir,
            )

        for key_info, lst in reward_extra_infos_dict.items():
            assert len(lst) == 0 or len(lst) == len(sample_scores), f"{key_info}: {len(lst)=}, {len(sample_scores)=}"

        data_sources = np.concatenate(data_source_lst, axis=0)

        data_src2var2metric2val = process_validation_metrics(data_sources, sample_uids, reward_extra_infos_dict)
        metric_dict = {}
        for data_source, var2metric2val in data_src2var2metric2val.items():
            core_var = "acc" if "acc" in var2metric2val else "reward"
            for var_name, metric2val in var2metric2val.items():
                n_max = max([int(name.split("@")[-1].split("/")[0]) for name in metric2val.keys()])
                for metric_name, metric_val in metric2val.items():
                    if (
                        (var_name == core_var)
                        and any(metric_name.startswith(pfx) for pfx in ["mean", "maj", "best"])
                        and (f"@{n_max}" in metric_name)
                    ):
                        metric_sec = "val-core"
                    else:
                        metric_sec = "val-aux"
                    pfx = f"{metric_sec}/{data_source}/{var_name}/{metric_name}"
                    metric_dict[pfx] = metric_val

        # Record validation metrics to chain debug
        validation_debug_info = {
            "global_step": int(self.global_steps),
            "num_validation_samples": len(sample_scores),
            "reward_min": float(np.min(sample_scores)) if sample_scores else float("nan"),
            "reward_max": float(np.max(sample_scores)) if sample_scores else float("nan"),
            "reward_mean": float(np.mean(sample_scores)) if sample_scores else float("nan"),
            "reward_std": float(np.std(sample_scores)) if len(sample_scores) > 1 else 0.0,
        }
        
        # Add extra metrics from reward_extra_infos_dict if available
        for key, values in reward_extra_infos_dict.items():
            if key == "reward":  # skip reward as we already added it
                continue
            if values and len(values) > 0 and isinstance(values[0], (int, float)):
                try:
                    validation_debug_info[f"{key}_min"] = float(np.min(values))
                    validation_debug_info[f"{key}_max"] = float(np.max(values))
                    validation_debug_info[f"{key}_mean"] = float(np.mean(values))
                except (TypeError, ValueError):
                    pass  # Skip non-numeric values
        
        append_chain_debug("validation_metrics", validation_debug_info)

        try:
            self._write_min_eval_debug_files(
                sample_uids=sample_uids,
                sample_outputs=sample_outputs,
                sample_gts=sample_gts,
                sample_scores=sample_scores,
                reward_extra_infos_dict=reward_extra_infos_dict,
            )
        except Exception as exc:
            append_chain_debug(
                "validation_min_debug_error",
                {
                    "global_step": int(self.global_steps),
                    "error": f"{type(exc).__name__}: {exc}",
                },
            )

        # if len(sample_turns) > 0:
        #     sample_turns = np.concatenate(sample_turns)
        #     metric_dict["val-aux/num_turns/min"] = sample_turns.min()
        #     metric_dict["val-aux/num_turns/max"] = sample_turns.max()
        #     metric_dict["val-aux/num_turns/mean"] = sample_turns.mean()

        return metric_dict

    def init_workers(self):
        """Initialize distributed training workers using Ray backend.

        Creates:
        1. Ray resource pools from configuration
        2. Worker groups for each role (actor, critic, etc.)
        """
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        actor_role = Role.ActorRolloutRef if Role.ActorRolloutRef in self.role_worker_mapping else Role.ActorRollout
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(actor_role)
            actor_rollout_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[actor_role],
                config=self.config.actor_rollout_ref,
                role=str(actor_role),
            )
            self.resource_pool_to_cls[resource_pool][str(actor_role)] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cfg = omega_conf_to_dataclass(self.config.critic)
            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=critic_cfg)
            self.resource_pool_to_cls[resource_pool][str(Role.Critic)] = critic_cls

        # create reference policy if needed
        if self.use_reference_policy and Role.RefPolicy in self.role_worker_mapping:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(
                self.role_worker_mapping[Role.RefPolicy],
                config=self.config.actor_rollout_ref,
                role=str(Role.RefPolicy),
            )
            self.resource_pool_to_cls[resource_pool][str(Role.RefPolicy)] = ref_policy_cls

        # create a reward model if reward_fn is None
        if self.use_rm:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model)
            self.resource_pool_to_cls[resource_pool][str(Role.RewardModel)] = rm_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`.
        # Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        wg_kwargs = {}  # Setting up kwargs for RayWorkerGroup
        if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
            wg_kwargs["ray_wait_register_center_timeout"] = self.config.trainer.ray_wait_register_center_timeout
        if OmegaConf.select(self.config.global_profiler, "steps") is not None:
            wg_kwargs["profile_steps"] = OmegaConf.select(self.config.global_profiler, "steps")
            # Only require nsight worker options when tool is nsys
            if OmegaConf.select(self.config.global_profiler, "tool") == "nsys":
                assert (
                    OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                    is not None
                ), "worker_nsight_options must be set when using nsys with profile_steps"
                wg_kwargs["worker_nsight_options"] = OmegaConf.to_container(
                    OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                )
        wg_kwargs["device_name"] = self.device_name

        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(
                resource_pool=resource_pool,
                ray_cls_with_init=worker_dict_cls,
                **wg_kwargs,
            )
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)

        if self.use_critic:
            self.critic_wg = all_wg[str(Role.Critic)]
            self.critic_wg.init_model()

        if self.use_reference_policy and not self.ref_in_actor:
            if str(Role.RefPolicy) in all_wg:
                self.ref_policy_wg = all_wg[str(Role.RefPolicy)]
                self.ref_policy_wg.init_model()
            else:
                # Model engine: ActorRolloutRefWorker
                assert str(Role.ActorRolloutRef) in all_wg, f"{all_wg.keys()=}"
                self.ref_policy_wg = all_wg[str(Role.ActorRolloutRef)]

        self.rm_wg = None
        # initalization of rm_wg will be deprecated in the future
        if self.use_rm:
            self.rm_wg = all_wg[str(Role.RewardModel)]
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg[str(actor_role)]
        self.actor_rollout_wg.init_model()

        # create async rollout manager and request scheduler
        self.async_rollout_mode = False
        if self.config.actor_rollout_ref.rollout.mode == "async":
            from arft.agent_flow import AgentFlowManager

            self.async_rollout_mode = True
            self.async_rollout_mode = True
            if self.config.reward_model.enable and self.config.reward_model.enable_resource_pool:
                rm_resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            else:
                rm_resource_pool = None
            
            self.async_rollout_manager = AgentFlowManager(
                config=self.config,
                worker_group=self.actor_rollout_wg,
                rm_resource_pool=rm_resource_pool,
            )

        assert self.async_rollout_mode == True, "async_rollout_mode must be True"

    def _cleanup_after_fit(self):
        """Best-effort cleanup for rollout services and dataloaders at trainer exit."""
        if getattr(self, "_fit_cleanup_done", False):
            return
        self._fit_cleanup_done = True

        if hasattr(self.actor_rollout_wg, "async_calls_finalize_fn_exec"):
            try:
                self.actor_rollout_wg.async_calls_finalize_fn_exec(blocking=True)
            except Exception as exc:  # pragma: no cover - cleanup best effort only
                print(f"Warning: failed to finalize pending async actor calls during cleanup: {exc}")

        if getattr(self, "async_rollout_manager", None) is not None:
            try:
                self.async_rollout_manager.clear_kv_cache()
            except Exception as exc:  # pragma: no cover - cleanup best effort only
                print(f"Warning: failed to clear rollout KV cache during cleanup: {exc}")
            try:
                self.async_rollout_manager.sleep()
            except Exception as exc:  # pragma: no cover - cleanup best effort only
                print(f"Warning: failed to put rollout replicas to sleep during cleanup: {exc}")

        for attr_name in ("train_dataloader", "val_dataloader"):
            if hasattr(self, attr_name):
                setattr(self, attr_name, None)

        gc.collect()

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        current_epoch = self.global_steps // len(self.train_dataloader)

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        if self.config.actor_rollout_ref.rollout.get("skip_rollout", False):
            rollout_skip = RolloutSkip(self.config, self.actor_rollout_wg)
            rollout_skip.wrap_generate_sequences()

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None
        self.max_steps_duration = 0

        prev_step_profile = False
        curr_step_profile = (
            self.global_steps in self.config.global_profiler.steps
            if self.config.global_profiler.steps is not None
            else False
        )
        next_step_profile = False

        for epoch in range(current_epoch, self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                if hasattr(self.actor_rollout_wg, "async_calls_finalize_fn_exec"):
                    self.actor_rollout_wg.async_calls_finalize_fn_exec(blocking=False)
                metrics = {}
                timing_raw = {}

                with marked_timer("start_profile", timing_raw):
                    self._start_profiling(
                        not prev_step_profile and curr_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                batch: DataProto = DataProto.from_single_dict(batch_dict)
                batch.meta_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature

                # add uid to batch
                batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
                )

                gen_batch = self._get_gen_batch(batch)

                # pass global_steps to trace
                gen_batch.meta_info["global_steps"] = self.global_steps
                gen_batch_output = gen_batch.repeat(
                    repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True
                )

                is_last_step = self.global_steps >= self.total_training_steps
                with marked_timer("step", timing_raw):
                    # generate a batch
                    with marked_timer("gen", timing_raw, color="red"):
                        gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch_output)

                        timing_raw.update(gen_batch_output.meta_info["timing"])
                        gen_batch_output.meta_info.pop("timing", None)

                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        if self.reward_fn is None:
                            raise ValueError("A reward_fn is required for REMAX advantage estimation.")
                        # TODO: implement REMAX advantage estimation for agent flow.
                        raise NotImplementedError("REMAX advantage estimation is not supported for agent flow.")
                    
                    # repeat to align with repeated responses in rollout
                    batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    num_steps = gen_batch_output.meta_info.pop("num_steps")
                    batch = batch.sample_level_repeat(num_steps)
                    batch = batch.union(gen_batch_output)

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()
                    
                    if "response_mask" not in batch.batch.keys():
                        batch.batch["response_mask"] = compute_response_mask(batch)

                    # batch needs to be padded to divisor of world size, we will pad with everything masked out
                    batch = self._pad_dataproto_to_world_size(batch)

                    # Balance the number of valid tokens across DP ranks.
                    # NOTE: This usually changes the order of data in the `batch`,
                    # which won't affect the advantage calculation (since it's based on uid),
                    # but might affect the loss calculation (due to the change of mini-batching).
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    with marked_timer("reward", timing_raw, color="yellow"):
                        # compute reward model score
                        if self.use_rm and "rm_scores" not in batch.batch.keys():
                            reward_tensor = self.rm_wg.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)

                        reward_tensor, reward_extra_infos_dict = extract_reward(batch)
                        reward_seq = reward_tensor.sum(-1)
                        append_chain_debug(
                            "trainer_reward_extract",
                            {
                                "global_step": int(self.global_steps),
                                "token_score_min": float(reward_tensor.min().item()),
                                "token_score_max": float(reward_tensor.max().item()),
                                "token_score_mean": float(reward_tensor.mean().item()),
                                "seq_score_min": float(reward_seq.min().item()),
                                "seq_score_max": float(reward_seq.max().item()),
                                "seq_score_mean": float(reward_seq.mean().item()),
                                "seq_negative_count": int((reward_seq < 0).sum().item()),
                                "seq_scores_head": reward_seq[:20].cpu().tolist(),
                            },
                        )

                    # Operating Mode Selection:
                    # - Bypass mode: Sets old_log_probs = rollout_log_probs (2 policies: π_rollout, π_θ)
                    # - Decoupled mode: Recomputes old_log_probs as proximal anchor (3 policies: π_rollout, π_old, π_θ)
                    #   Note: π_old computed once per data batch, serves as stable reference during mini-batch updates
                    rollout_corr_config = self.config.algorithm.get("rollout_correction", None)
                    bypass_recomputing_logprobs = rollout_corr_config and rollout_corr_config.get("bypass_mode", False)
                    if bypass_recomputing_logprobs:  # Use `rollout_log_probs`
                        from verl.trainer.ppo.rollout_corr_helper import apply_rollout_correction

                        apply_rollout_correction(
                            batch=batch,
                            rollout_corr_config=rollout_corr_config,
                            policy_loss_config=self.config.actor_rollout_ref.actor.policy_loss,
                        )
                    else:  # Recompute old_log_probs
                        with marked_timer("old_log_prob", timing_raw, color="blue"):
                            old_log_prob, old_log_prob_mfu = self._compute_old_log_prob(batch)
                            actor_config = self.config.actor_rollout_ref.actor
                            old_log_prob_metrics = {
                                "perf/mfu/actor_infer": old_log_prob_mfu,
                            }
                            entropys = old_log_prob.batch.pop("entropys", None)
                            if entropys is not None:
                                response_masks = batch.batch["response_mask"]
                                entropy_agg = agg_loss(
                                    loss_mat=entropys,
                                    loss_mask=response_masks,
                                    loss_agg_mode=actor_config.loss_agg_mode,
                                    loss_scale_factor=actor_config.loss_scale_factor,
                                )
                                old_log_prob_metrics["actor/entropy"] = entropy_agg.detach().item()
                            metrics.update(old_log_prob_metrics)
                            batch = batch.union(old_log_prob)
                            if "rollout_log_probs" in batch.batch.keys():
                                # TODO: we may want to add diff of probs too.
                                from verl.utils.debug.metrics import calculate_debug_metrics

                                metrics.update(calculate_debug_metrics(batch))

                    assert "old_log_probs" in batch.batch, f'"old_log_prob" not in {batch.batch.keys()=}'

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with marked_timer(str(Role.RefPolicy), timing_raw, color="olive"):
                            if not self.ref_in_actor:
                                ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            else:
                                ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with marked_timer("values", timing_raw, color="cyan"):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with marked_timer("adv", timing_raw, color="brown"):
                        # we combine with rule-based rm
                        batch.batch["token_level_scores"] = reward_tensor
                        score_min = float(batch.batch["token_level_scores"].min().item())
                        score_max = float(batch.batch["token_level_scores"].max().item())
                        score_mean = float(batch.batch["token_level_scores"].mean().item())
                        score_seq = batch.batch["token_level_scores"].sum(-1)

                        if reward_extra_infos_dict:
                            batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                        # compute rewards. apply_kl_penalty if available
                        if self.config.algorithm.use_kl_in_reward:
                            batch, kl_metrics = apply_kl_penalty(
                                batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                            )
                            metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                        reward_min = float(batch.batch["token_level_rewards"].min().item())
                        reward_max = float(batch.batch["token_level_rewards"].max().item())
                        reward_mean = float(batch.batch["token_level_rewards"].mean().item())
                        reward_seq = batch.batch["token_level_rewards"].sum(-1)
                        append_chain_debug(
                            "trainer_reward_tensor",
                            {
                                "global_step": int(self.global_steps),
                                "score_min": score_min,
                                "score_max": score_max,
                                "score_mean": score_mean,
                                "score_seq_min": float(score_seq.min().item()),
                                "score_seq_max": float(score_seq.max().item()),
                                "score_seq_mean": float(score_seq.mean().item()),
                                "score_seq_negative_count": int((score_seq < 0).sum().item()),
                                "reward_min": reward_min,
                                "reward_max": reward_max,
                                "reward_mean": reward_mean,
                                "reward_seq_min": float(reward_seq.min().item()),
                                "reward_seq_max": float(reward_seq.max().item()),
                                "reward_seq_mean": float(reward_seq.mean().item()),
                                "reward_seq_negative_count": int((reward_seq < 0).sum().item()),
                                "reward_equals_score": bool(
                                    torch.allclose(
                                        batch.batch["token_level_rewards"],
                                        batch.batch["token_level_scores"],
                                    )
                                ),
                                "use_kl_in_reward": bool(self.config.algorithm.use_kl_in_reward),
                            },
                        )

                        # Compute rollout correction: IS weights, rejection sampling, and metrics
                        # Only runs in decoupled mode (computes once per batch using stable π_old)
                        # In bypass mode, this is skipped - actor computes metrics from evolving π_θ vs π_rollout
                        if (
                            rollout_corr_config is not None
                            and "rollout_log_probs" in batch.batch
                            and not bypass_recomputing_logprobs  # Only in decoupled mode
                        ):
                            from verl.trainer.ppo.rollout_corr_helper import compute_rollout_correction_and_add_to_batch

                            # Compute IS weights, apply rejection sampling, compute metrics
                            batch, is_metrics = compute_rollout_correction_and_add_to_batch(batch, rollout_corr_config)
                            # IS and off-policy metrics already have rollout_corr/ prefix
                            metrics.update(is_metrics)

                        # compute advantages, executed on the driver process
                        norm_adv_by_std_in_grpo = self.config.algorithm.get(
                            "norm_adv_by_std_in_grpo", True
                        )  # GRPO adv normalization factor

                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                            config=self.config.algorithm,
                        )

                    # update critic
                    if self.use_critic:
                        with marked_timer("update_critic", timing_raw, color="pink"):
                            # Temporarily replace response_mask for critic
                            response_mask = batch.batch["response_mask"]
                            response_length = response_mask.sum(dim=1).unsqueeze(1) - 1
                            value_mask = torch.zeros_like(response_mask)
                            value_mask.scatter_(1, response_length, 1)
                            batch.batch["response_mask"] = value_mask
                            
                            # update critic
                            critic_output = self.critic_wg.update_critic(batch)
                            
                            # restore response_mask
                            batch.batch["response_mask"] = response_mask
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with marked_timer("update_actor", timing_raw, color="red"):
                            rollout_config = self.config.actor_rollout_ref.rollout
                            batch.meta_info["multi_turn"] = rollout_config.multi_turn.enable
                            # TODO: Make "temperature" single source of truth from generation.
                            batch.meta_info["temperature"] = rollout_config.temperature
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                    # Log rollout generations if enabled
                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        self._log_rollout_data(batch, reward_extra_infos_dict, timing_raw, rollout_data_dir)

                # validate
                if (
                    self.val_reward_fn is not None
                    and self.config.trainer.test_freq > 0
                    and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
                ):
                    with marked_timer("testing", timing_raw, color="green"):
                        val_metrics: dict = self._validate()
                        if is_last_step:
                            last_val_metrics = val_metrics
                    metrics.update(val_metrics)

                # Check if the ESI (Elastic Server Instance)/training plan is close to expiration.
                esi_close_to_expiration = should_save_ckpt_esi(
                    max_steps_duration=self.max_steps_duration,
                    redundant_time=self.config.trainer.esi_redundant_time,
                )
                # Check if the conditions for saving a checkpoint are met.
                # The conditions include a mandatory condition (1) and
                # one of the following optional conditions (2/3/4):
                # 1. The save frequency is set to a positive value.
                # 2. It's the last training step.
                # 3. The current step number is a multiple of the save frequency.
                # 4. The ESI(Elastic Server Instance)/training plan is close to expiration.
                if self.config.trainer.save_freq > 0 and (
                    is_last_step or self.global_steps % self.config.trainer.save_freq == 0 or esi_close_to_expiration
                ):
                    if esi_close_to_expiration:
                        print("Force saving checkpoint: ESI instance expiration approaching.")
                    with marked_timer("save_checkpoint", timing_raw, color="green"):
                        self._save_checkpoint()

                with marked_timer("stop_profile", timing_raw):
                    next_step_profile = (
                        self.global_steps + 1 in self.config.global_profiler.steps
                        if self.config.global_profiler.steps is not None
                        else False
                    )
                    self._stop_profiling(
                        curr_step_profile and not next_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                    prev_step_profile = curr_step_profile
                    curr_step_profile = next_step_profile

                steps_duration = timing_raw["step"]
                self.max_steps_duration = max(self.max_steps_duration, steps_duration)

                # training metrics
                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                    }
                )
                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))
                # Note: mismatch metrics (KL, PPL, etc.) are collected at line 1179 after advantage computation

                # this is experimental and may be changed/removed in the future in favor of a general-purpose one
                if isinstance(self.train_dataloader.sampler, AbstractCurriculumSampler):
                    self.train_dataloader.sampler.update(batch=batch)

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                progress_bar.update(1)
                self.global_steps += 1

                if (
                    hasattr(self.config.actor_rollout_ref.actor, "profiler")
                    and self.config.actor_rollout_ref.actor.profiler.tool == "torch_memory"
                ):
                    self.actor_rollout_wg.dump_memory_snapshot(
                        tag=f"post_update_step{self.global_steps}", sub_dir=f"step{self.global_steps}"
                    )

                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    self._cleanup_after_fit()
                    return

                # this is experimental and may be changed/removed in the future
                # in favor of a general-purpose data buffer pool
                if hasattr(self.train_dataset, "on_batch_end"):
                    # The dataset may be changed after each training batch
                    self.train_dataset.on_batch_end(batch=batch)

    def _pad_dataproto_to_world_size(self, batch):
        world_sizes = []
        if self.use_critic and self.critic_wg.world_size != 0:
            world_sizes.append(self.critic_wg.world_size)
        if self.use_reference_policy and self.ref_policy_wg.world_size != 0:
            world_sizes.append(self.ref_policy_wg.world_size)
        if self.use_rm and self.rm_wg.world_size != 0:
            world_sizes.append(self.rm_wg.world_size)
        if self.hybrid_engine:
            if self.actor_rollout_wg.world_size != 0:
                world_sizes.append(self.actor_rollout_wg.world_size)
        else:
            if self.actor_wg.world_size != 0:
                world_sizes.append(self.actor_wg.world_size)
            if self.rollout_wg.world_size != 0:
                world_sizes.append(self.rollout_wg.world_size)
        if not world_sizes:
            return batch

        world_size = reduce(math.lcm, world_sizes)

        original_batch_size = batch.batch["prompts"].shape[0]
        batch, pad_size = pad_dataproto_to_divisor(batch, world_size)
        batch.non_tensor_batch["is_pad"] = np.array([False] * original_batch_size + [True] * pad_size)

        return batch
