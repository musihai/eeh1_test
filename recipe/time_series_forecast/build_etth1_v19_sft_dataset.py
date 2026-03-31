from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from recipe.time_series_forecast.build_etth1_sft_dataset import (
    ETTH1_COVARIATE_COLUMNS,
    ETTH1_FEATURE_COLUMNS,
    FEATURE_TOOL_SCHEMA_BY_NAME,
    _filter_records_for_stage_mode,
    _make_stage_record,
    _make_tool_call,
    _summarize_paper_turn3_protocol,
    _validate_paper_turn3_protocol,
    build_feature_tool_results,
)
from recipe.time_series_forecast.candidate_selection_support import compute_candidate_visible_metrics
from recipe.time_series_forecast.dataset_file_utils import load_jsonl_records, write_metadata_file
from recipe.time_series_forecast.diagnostic_policy import (
    build_diagnostic_plan,
    plan_diagnostic_tool_batches,
    select_feature_tool_names,
)
from recipe.time_series_forecast.prompts import (
    RISK_GATE_TIMESERIES_TOOL_SCHEMA,
    build_runtime_user_prompt,
    build_timeseries_system_prompt,
    build_v19_final_select_prompt,
    build_v19_risk_gate_prompt,
)
from recipe.time_series_forecast.task_protocol import parse_task_prompt
from recipe.time_series_forecast.time_series_io import parse_time_series_string


MODE_FINAL_SELECT_ONLY = "final_select_only"
MODE_FULL_STEPWISE_V19 = "full_stepwise_v19"
SUPPORTED_MODES = {MODE_FINAL_SELECT_ONLY, MODE_FULL_STEPWISE_V19}
DEFAULT_INPUT_DIR = Path("dataset/ett_v19_candidate_bank")
DEFAULT_OUTPUT_DIR = Path("dataset/ett_sft_etth1_v19_final_select_only")
DEFAULT_TURN2_POLICY = "fixed_expand"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the v19 ETTh1 SFT datasets.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--mode", choices=sorted(SUPPORTED_MODES), default=MODE_FINAL_SELECT_ONLY)
    parser.add_argument("--turn2-policy", choices=["fixed_expand", "learned_risk_gate"], default=DEFAULT_TURN2_POLICY)
    parser.add_argument("--balance-final-select-train", action="store_true")
    parser.add_argument("--balance-final-select-train-seed", type=int, default=19)
    parser.add_argument("--shuffle-final-candidates", action="store_true")
    parser.add_argument("--candidate-shuffle-seed", type=int, default=19)
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    return parser.parse_args()


def _series_stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {
            "start": 0.0,
            "end": 0.0,
            "mean": 0.0,
            "std": 0.0,
            "net_change": 0.0,
            "mean_abs_step": 0.0,
        }
    array = np.asarray(values, dtype=float)
    return {
        "start": float(array[0]),
        "end": float(array[-1]),
        "mean": float(array.mean()),
        "std": float(array.std()),
        "net_change": float(array[-1] - array[0]),
        "mean_abs_step": float(np.mean(np.abs(np.diff(array)))) if len(array) >= 2 else 0.0,
    }


def _history_stats(historical_data: str, *, target_column: str) -> dict[str, dict[str, float]]:
    _, values = parse_time_series_string(historical_data, target_column=target_column)
    recent = values[-24:] if len(values) >= 24 else values
    return {
        "full": _series_stats(values),
        "recent": _series_stats(recent),
        "last_value": float(values[-1]) if values else 0.0,
    }


def _candidate_stats(candidate: dict[str, Any]) -> dict[str, float]:
    prediction_text = str(candidate.get("prediction_text") or candidate.get("compact_prediction_text") or "")
    _, values = parse_time_series_string(prediction_text)
    return _series_stats(values)


def _candidate_gap_report(reference: dict[str, float], candidate: dict[str, float]) -> dict[str, float]:
    return {
        "level_gap": abs(candidate["start"] - reference["end"]),
        "trend_gap": abs(candidate["net_change"] - reference["net_change"]),
        "step_gap": abs(candidate["mean_abs_step"] - reference["mean_abs_step"]),
        "std_gap": abs(candidate["std"] - reference["std"]),
    }


def _best_alternative_candidate(
    *,
    row: dict[str, Any],
    candidates: list[dict[str, Any]],
    default_candidate_id: str,
) -> dict[str, Any] | None:
    alternatives = [candidate for candidate in candidates if str(candidate.get("candidate_id") or "") != default_candidate_id]
    if not alternatives:
        return None
    score_details = row.get("candidate_score_details") or {}
    ranked = sorted(
        alternatives,
        key=lambda candidate: (
            float((score_details.get(str(candidate.get("candidate_id") or "")) or {}).get("orig_mse", float("inf"))),
            str(candidate.get("candidate_id") or ""),
        ),
    )
    return ranked[0]


def _metric_sentence(
    *,
    metric_name: str,
    selected_id: str,
    selected_stats: dict[str, float],
    compare_id: str,
    compare_stats: dict[str, float],
    recent_stats: dict[str, float],
) -> str:
    if metric_name == "level_gap":
        return (
            f"{selected_id} starts closer to the latest observed level "
            f"({abs(selected_stats['start'] - recent_stats['end']):.2f} vs "
            f"{abs(compare_stats['start'] - recent_stats['end']):.2f} for {compare_id})."
        )
    if metric_name == "trend_gap":
        return (
            f"Its forecast net change stays closer to the recent direction "
            f"({selected_stats['net_change']:.2f} vs recent {recent_stats['net_change']:.2f}) "
            f"than {compare_id} ({compare_stats['net_change']:.2f})."
        )
    if metric_name == "step_gap":
        return (
            f"Its step-to-step movement better matches the recent local pace "
            f"({selected_stats['mean_abs_step']:.2f} vs recent {recent_stats['mean_abs_step']:.2f}) "
            f"than {compare_id} ({compare_stats['mean_abs_step']:.2f})."
        )
    return (
        f"Its overall spread stays closer to the recent window "
        f"({selected_stats['std']:.2f} vs recent {recent_stats['std']:.2f}) "
        f"than {compare_id} ({compare_stats['std']:.2f})."
    )


def _build_final_select_reasoning(
    *,
    row: dict[str, Any],
    historical_data: str,
    target_column: str,
    candidates: list[dict[str, Any]],
) -> str:
    candidate_by_id = {str(candidate.get("candidate_id") or ""): candidate for candidate in candidates}
    default_candidate_id = str(row.get("default_candidate_id") or "")
    selected_candidate_id = str(row.get("final_candidate_label") or "")
    selected_candidate = candidate_by_id.get(selected_candidate_id)
    default_candidate = candidate_by_id.get(default_candidate_id)
    if selected_candidate is None:
        return f"I compare the visible candidates and choose {selected_candidate_id}."
    history_stats = _history_stats(historical_data, target_column=target_column)
    recent_stats = history_stats["recent"]
    selected_stats = _candidate_stats(selected_candidate)

    compare_candidate = default_candidate
    if selected_candidate_id == default_candidate_id or compare_candidate is None:
        compare_candidate = _best_alternative_candidate(
            row=row,
            candidates=candidates,
            default_candidate_id=default_candidate_id,
        )
    if compare_candidate is None:
        return (
            f"{selected_candidate_id} keeps the forecast aligned with the recent window. "
            f"It preserves the recent level ({selected_stats['start']:.2f}) and local movement "
            f"({selected_stats['mean_abs_step']:.2f}) without changing the candidate set."
        )
    compare_candidate_id = str(compare_candidate.get("candidate_id") or "")
    compare_stats = _candidate_stats(compare_candidate)
    selected_gaps = _candidate_gap_report(recent_stats, selected_stats)
    compare_gaps = _candidate_gap_report(recent_stats, compare_stats)
    ranked_metrics = sorted(
        selected_gaps.keys(),
        key=lambda key: (compare_gaps[key] - selected_gaps[key], -selected_gaps[key]),
        reverse=True,
    )
    primary_metric = ranked_metrics[0]
    secondary_metric = ranked_metrics[1] if len(ranked_metrics) > 1 else ranked_metrics[0]
    sentence_one = _metric_sentence(
        metric_name=primary_metric,
        selected_id=selected_candidate_id,
        selected_stats=selected_stats,
        compare_id=compare_candidate_id,
        compare_stats=compare_stats,
        recent_stats=recent_stats,
    )
    sentence_two = _metric_sentence(
        metric_name=secondary_metric,
        selected_id=selected_candidate_id,
        selected_stats=selected_stats,
        compare_id=compare_candidate_id,
        compare_stats=compare_stats,
        recent_stats=recent_stats,
    )
    return f"{sentence_one} {sentence_two}"


def _build_final_select_answer(*, row: dict[str, Any], historical_data: str, target_column: str, candidates: list[dict[str, Any]]) -> str:
    candidate_id = str(row.get("final_candidate_label") or "")
    reasoning = _build_final_select_reasoning(
        row=row,
        historical_data=historical_data,
        target_column=target_column,
        candidates=candidates,
    )
    return (
        "<think>\n"
        f"{reasoning}\n"
        "</think>\n"
        "<answer>\n"
        f"candidate_id={candidate_id}\n"
        "</answer>"
    )


def _build_proposal_reflection(*, fixed_expand: bool, decision: str) -> str:
    if fixed_expand:
        return (
            "I do not commit to a final expert here. I mark the default path as risky so the next turn can compare "
            "the default candidates against alternative expert baselines."
        )
    return (
        "I only decide whether the default path looks risky enough to expand more candidates. "
        f"My decision is {decision}."
    )


def _build_diagnostic_reflection(
    *,
    rationale: str,
    batch_tool_names: list[str],
    completed_feature_tools: list[str],
) -> str:
    batch_text = ", ".join(batch_tool_names)
    completed_text = ", ".join(completed_feature_tools) if completed_feature_tools else "none"
    return (
        f"I gather {batch_text} next because {rationale}. "
        f"Completed diagnostic tools before this turn: {completed_text}."
    )


def _normalize_mode(mode: str) -> str:
    normalized = str(mode or "").strip().lower()
    if normalized not in SUPPORTED_MODES:
        raise ValueError(f"Unsupported v19 SFT mode: {mode!r}")
    return normalized


def _load_candidate_rows(input_dir: Path, split: str) -> list[dict[str, Any]]:
    return load_jsonl_records(input_dir / f"{split}.jsonl")


def _shared_fields(row: dict[str, Any], *, trajectory_turn_count: int, mode: str, turn2_policy: str) -> dict[str, Any]:
    return {
        "task": "time_series_forecasting",
        "data_source": row.get("data_source"),
        "target_column": row.get("target_column"),
        "lookback_window": int(row.get("lookback_window") or 96),
        "forecast_horizon": int(row.get("forecast_horizon") or 96),
        "source_sample_index": int(row.get("source_sample_index", -1)),
        "uid": row.get("uid"),
        "default_expert": row.get("default_expert"),
        "default_candidate_id": row.get("default_candidate_id"),
        "risk_label": row.get("risk_label"),
        "risk_value_rel": row.get("risk_value_rel"),
        "final_candidate_label": row.get("final_candidate_label"),
        "final_candidate_error": row.get("final_candidate_error"),
        "default_candidate_error": row.get("default_candidate_error"),
        "final_vs_default_error": row.get("final_vs_default_error"),
        "candidate_score_details": json.dumps(row.get("candidate_score_details") or {}, ensure_ascii=False),
        "candidate_prediction_text_map": json.dumps(row.get("candidate_prediction_text_map") or {}, ensure_ascii=False),
        "turn2_policy": turn2_policy,
        "sft_stage_mode": mode,
        "paper_turn3_required": False,
        "trajectory_turn_count": int(trajectory_turn_count),
    }


def _final_select_candidates(row: dict[str, Any], *, turn2_policy: str) -> list[dict[str, Any]]:
    default_candidates = list(row.get("default_candidates") or [])
    alt_candidates = list(row.get("alt_candidates") or [])
    if turn2_policy == "fixed_expand":
        return [*default_candidates, *alt_candidates]
    risk_label = str(row.get("risk_label") or "default_risky")
    if risk_label == "default_ok":
        return default_candidates
    return [*default_candidates, *alt_candidates]


def _shuffle_candidates(
    candidates: list[dict[str, Any]],
    *,
    row: dict[str, Any],
    split_name: str,
    shuffle_seed: int | None,
) -> list[dict[str, Any]]:
    if not candidates or shuffle_seed is None:
        return candidates
    stable_uid = str(row.get("uid") or row.get("source_sample_index") or "")
    candidate_key = "|".join(str(item.get("candidate_id") or "") for item in candidates)
    digest = hashlib.sha256(f"{shuffle_seed}:{split_name}:{stable_uid}:{candidate_key}".encode("utf-8")).hexdigest()
    local_random = random.Random(int(digest[:16], 16))
    shuffled = list(candidates)
    local_random.shuffle(shuffled)
    return shuffled


def _balance_final_select_frame(dataframe: pd.DataFrame, *, seed: int) -> pd.DataFrame:
    if dataframe.empty or "final_candidate_label" not in dataframe.columns:
        return dataframe
    label_counts = dataframe["final_candidate_label"].astype(str).value_counts()
    if label_counts.empty or label_counts.nunique() <= 1:
        return dataframe
    target_count = int(label_counts.max())
    rng = np.random.default_rng(seed)
    balanced_parts: list[pd.DataFrame] = []
    for label in sorted(label_counts.index.astype(str).tolist()):
        part = dataframe.loc[dataframe["final_candidate_label"].astype(str) == label]
        if len(part) >= target_count:
            balanced_parts.append(part.copy())
            continue
        sampled_positions = rng.choice(part.index.to_numpy(), size=target_count, replace=True)
        balanced_parts.append(part.loc[sampled_positions].copy())
    balanced = pd.concat(balanced_parts, axis=0, ignore_index=True)
    balanced = balanced.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return balanced


def build_v19_sft_records(
    row: dict[str, Any],
    *,
    mode: str,
    turn2_policy: str,
    split_name: str,
    shuffle_candidate_seed: int | None = None,
) -> list[dict[str, Any]]:
    normalized_mode = _normalize_mode(mode)
    historical_data = str(row.get("historical_data") or "")
    data_source = str(row.get("data_source") or "ETTh1")
    target_column = str(row.get("target_column") or "OT")
    lookback_window = int(row.get("lookback_window") or 96)
    forecast_horizon = int(row.get("forecast_horizon") or 96)
    history_values = parse_time_series_string(historical_data, target_column=target_column)[1]
    feature_results = build_feature_tool_results(history_values)
    feature_result_by_name = {result.tool_name: result for result in feature_results}
    selected_feature_tools = select_feature_tool_names(history_values)
    diagnostic_plan = build_diagnostic_plan(history_values)
    diagnostic_batches = plan_diagnostic_tool_batches(
        selected_feature_tools,
    )
    analysis_history: list[str] = []
    completed_feature_tools: list[str] = []
    trajectory_turn_count = 1 if normalized_mode == MODE_FINAL_SELECT_ONLY else len(diagnostic_batches) + 2
    shared = _shared_fields(
        row,
        trajectory_turn_count=trajectory_turn_count,
        mode=normalized_mode,
        turn2_policy=turn2_policy,
    )
    system_prompt = build_timeseries_system_prompt(data_source=data_source, target_column=target_column)
    records: list[dict[str, Any]] = []
    turn_stage_order = 0
    feature_call_serial = 1

    if normalized_mode == MODE_FULL_STEPWISE_V19:
        for batch_index, batch_tool_names in enumerate(diagnostic_batches):
            diagnostic_prompt = build_runtime_user_prompt(
                data_source=data_source,
                target_column=target_column,
                lookback_window=lookback_window,
                forecast_horizon=forecast_horizon,
                time_series_data=historical_data,
                history_analysis=analysis_history,
                prediction_results=None,
                available_feature_tools=batch_tool_names,
                completed_feature_tools=completed_feature_tools,
                turn_stage="diagnostic",
            )
            tool_calls = [
                _make_tool_call(
                    tool_name=tool_name,
                    arguments={},
                    call_id=f"call_{feature_call_serial + offset}_{tool_name}",
                )
                for offset, tool_name in enumerate(batch_tool_names)
            ]
            records.append(
                _make_stage_record(
                    shared_fields=shared,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": diagnostic_prompt},
                        {
                            "role": "assistant",
                            "content": "",
                            "reasoning_content": _build_diagnostic_reflection(
                                rationale=diagnostic_plan.rationale,
                                batch_tool_names=list(batch_tool_names),
                                completed_feature_tools=list(completed_feature_tools),
                            ),
                            "tool_calls": tool_calls,
                        },
                    ],
                    tools=[copy.deepcopy(FEATURE_TOOL_SCHEMA_BY_NAME[name]) for name in batch_tool_names],
                    turn_stage="diagnostic",
                    turn_stage_order=turn_stage_order,
                    trajectory_turn_count=trajectory_turn_count,
                    current_required_feature_tools=list(batch_tool_names),
                    completed_feature_tools_before_turn=list(completed_feature_tools),
                    history_analysis_count_before_turn=len(analysis_history),
                    paper_turn3_required=False,
                    diagnostic_batch_index=batch_index,
                )
            )
            for tool_name in batch_tool_names:
                analysis_history.append(feature_result_by_name[tool_name].tool_output)
                completed_feature_tools.append(tool_name)
            feature_call_serial += len(batch_tool_names)
            turn_stage_order += 1

        proposal_prompt = build_v19_risk_gate_prompt(
            data_source=data_source,
            target_column=target_column,
            lookback_window=lookback_window,
            forecast_horizon=forecast_horizon,
            time_series_data=historical_data,
            history_analysis=analysis_history,
            default_expert=str(row.get("default_expert") or ""),
            fixed_expand=(turn2_policy == "fixed_expand"),
        )
        proposal_decision = "default_risky" if turn2_policy == "fixed_expand" else str(row.get("risk_label") or "default_risky")
        proposal_tool_call = _make_tool_call(
            tool_name="route_time_series",
            arguments={"decision": proposal_decision},
            call_id="call_route_time_series",
        )
        records.append(
            _make_stage_record(
                shared_fields=shared,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": proposal_prompt},
                    {
                        "role": "assistant",
                        "content": "",
                        "reasoning_content": _build_proposal_reflection(
                            fixed_expand=(turn2_policy == "fixed_expand"),
                            decision=proposal_decision,
                        ),
                        "tool_calls": [proposal_tool_call],
                    },
                ],
                tools=[copy.deepcopy(RISK_GATE_TIMESERIES_TOOL_SCHEMA)],
                turn_stage="proposal",
                turn_stage_order=turn_stage_order,
                trajectory_turn_count=trajectory_turn_count,
                current_required_feature_tools=list(selected_feature_tools),
                completed_feature_tools_before_turn=list(completed_feature_tools),
                history_analysis_count_before_turn=len(analysis_history),
                paper_turn3_required=False,
            )
        )
        turn_stage_order += 1
    else:
        analysis_history = list(row.get("analysis_history") or [])

    final_candidates = _final_select_candidates(row, turn2_policy=turn2_policy)
    final_candidates = _shuffle_candidates(
        final_candidates,
        row=row,
        split_name=split_name,
        shuffle_seed=shuffle_candidate_seed,
    )
    visible_candidate_metrics = compute_candidate_visible_metrics(
        historical_data=historical_data,
        target_column=target_column,
        candidates=final_candidates,
        default_candidate_id=str(row.get("default_candidate_id") or ""),
    )
    prompt_candidate_ids = [str(candidate.get("candidate_id") or "") for candidate in final_candidates]
    gold_candidate_id = str(row.get("final_candidate_label") or "")
    try:
        gold_prompt_rank = prompt_candidate_ids.index(gold_candidate_id)
    except ValueError:
        gold_prompt_rank = -1
    final_prompt = build_v19_final_select_prompt(
        data_source=data_source,
        target_column=target_column,
        lookback_window=lookback_window,
        forecast_horizon=forecast_horizon,
        time_series_data=historical_data,
        history_analysis=analysis_history,
        default_expert=str(row.get("default_expert") or ""),
        default_candidate_id=str(row.get("default_candidate_id") or ""),
        expanded=(turn2_policy == "fixed_expand" or str(row.get("risk_label") or "") == "default_risky"),
        candidates=final_candidates,
    )
    final_shared = {
        **shared,
        "prompt_candidate_order": json.dumps(prompt_candidate_ids, ensure_ascii=False),
        "gold_candidate_prompt_rank": int(gold_prompt_rank),
        "visible_candidate_metrics": json.dumps(visible_candidate_metrics, ensure_ascii=False),
    }
    final_record = _make_stage_record(
        shared_fields=final_shared,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": final_prompt},
            {
                "role": "assistant",
                "content": _build_final_select_answer(
                    row=row,
                    historical_data=historical_data,
                    target_column=target_column,
                    candidates=final_candidates,
                ),
            },
        ],
        tools=None,
        turn_stage="final_select",
        turn_stage_order=turn_stage_order,
        trajectory_turn_count=trajectory_turn_count,
        current_required_feature_tools=list(selected_feature_tools),
        completed_feature_tools_before_turn=list(completed_feature_tools),
        history_analysis_count_before_turn=len(analysis_history),
        paper_turn3_required=True,
    )
    records.append(final_record)

    if normalized_mode == MODE_FINAL_SELECT_ONLY:
        return [final_record]
    return records


def convert_candidate_bank_to_parquet(
    *,
    input_path: Path,
    output_path: Path,
    mode: str,
    turn2_policy: str,
    balance_final_select_train: bool = False,
    balance_seed: int = 19,
    shuffle_final_candidates: bool = False,
    candidate_shuffle_seed: int = 19,
) -> pd.DataFrame:
    rows = load_jsonl_records(input_path)
    records: list[dict[str, Any]] = []
    split_name = input_path.stem
    for row in rows:
        records.extend(
            build_v19_sft_records(
                row,
                mode=mode,
                turn2_policy=turn2_policy,
                split_name=split_name,
                shuffle_candidate_seed=(candidate_shuffle_seed if shuffle_final_candidates else None),
            )
        )
    dataframe = pd.DataFrame(records)
    if balance_final_select_train and split_name == "train" and _normalize_mode(mode) == MODE_FINAL_SELECT_ONLY:
        dataframe = _balance_final_select_frame(dataframe, seed=balance_seed)
    _validate_paper_turn3_protocol(
        dataframe,
        split_name=split_name,
        output_path=output_path,
        allow_no_refinement=False,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_parquet(output_path, index=False)
    return dataframe


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    split_frames: dict[str, pd.DataFrame] = {}
    metadata: dict[str, Any] = {
        "dataset_kind": "etth1_runtime_sft_parquet",
        "pipeline_stage": "sft_v19",
        "task_type": "multivariate time-series forecasting",
        "historical_data_protocol": "timestamped_named_rows",
        "target_column": "OT",
        "observed_feature_columns": list(ETTH1_FEATURE_COLUMNS),
        "observed_covariates": list(ETTH1_COVARIATE_COLUMNS),
        "model_input_width": len(ETTH1_FEATURE_COLUMNS),
        "sft_stage_mode": args.mode,
        "turn2_policy": args.turn2_policy,
        "balance_final_select_train": bool(args.balance_final_select_train),
        "balance_final_select_train_seed": int(args.balance_final_select_train_seed),
        "shuffle_final_candidates": bool(args.shuffle_final_candidates),
        "candidate_shuffle_seed": int(args.candidate_shuffle_seed),
        "turn3_protocol": "paper_think_answer_xml",
        "source_candidate_bank_dir": str(args.input_dir.resolve()),
    }

    for split_name in args.splits:
        frame = convert_candidate_bank_to_parquet(
            input_path=args.input_dir / f"{split_name}.jsonl",
            output_path=output_dir / f"{split_name}.parquet",
            mode=args.mode,
            turn2_policy=args.turn2_policy,
            balance_final_select_train=bool(args.balance_final_select_train),
            balance_seed=int(args.balance_final_select_train_seed),
            shuffle_final_candidates=bool(args.shuffle_final_candidates),
            candidate_shuffle_seed=int(args.candidate_shuffle_seed),
        )
        split_frames[split_name] = frame
        summary = _summarize_paper_turn3_protocol(frame)
        metadata[f"{split_name}_rows"] = int(len(frame))
        metadata[f"{split_name}_turn3_protocol_checked_count"] = int(summary.get("turn3_protocol_checked_count", 0))
        metadata[f"{split_name}_turn3_protocol_valid_ratio"] = float(summary.get("turn3_protocol_valid_ratio", 0.0))
        metadata[f"{split_name}_stage_distribution"] = {
            str(k): int(v)
            for k, v in frame["turn_stage"].astype(str).value_counts().sort_index().items()
        }
        if "final_candidate_label" in frame.columns:
            metadata[f"{split_name}_final_candidate_label_distribution"] = {
                str(k): int(v)
                for k, v in frame["final_candidate_label"].astype(str).value_counts().sort_index().items()
            }
    write_metadata_file(output_dir, metadata)
    print(json.dumps(metadata, ensure_ascii=False))


if __name__ == "__main__":
    main()
