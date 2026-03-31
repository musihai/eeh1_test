from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from recipe.time_series_forecast.tool_call_protocol import extract_tool_calls


ALLOWED_ROUTING_MODELS = {"chronos2", "arima", "patchtst", "itransformer"}


def _to_builtin(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _to_builtin(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_builtin(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_to_builtin(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _take_evenly_spaced(dataframe: pd.DataFrame, max_samples: int) -> pd.DataFrame:
    if max_samples <= 0 or len(dataframe) <= max_samples:
        return dataframe.reset_index(drop=True)
    if max_samples == 1:
        return dataframe.iloc[[len(dataframe) // 2]].reset_index(drop=True)
    indexes = sorted({round(i * (len(dataframe) - 1) / (max_samples - 1)) for i in range(max_samples)})
    return dataframe.iloc[indexes].reset_index(drop=True)


def _parse_jsonish_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return {}
        try:
            value = json.loads(text)
        except Exception:
            return {}
    if not isinstance(value, dict):
        return {}
    return {str(key): item for key, item in value.items()}


def _model_error_by_name(row: pd.Series) -> dict[str, float]:
    score_details = _parse_jsonish_mapping(row.get("teacher_eval_score_details"))
    error_by_name: dict[str, float] = {}
    for model_name, payload in score_details.items():
        if not isinstance(payload, dict):
            continue
        try:
            orig_mse = float(payload.get("orig_mse"))
        except (TypeError, ValueError):
            continue
        if np.isfinite(orig_mse):
            error_by_name[str(model_name).strip().lower()] = float(orig_mse)
    return error_by_name


def _top2_models(row: pd.Series) -> list[str]:
    error_by_name = _model_error_by_name(row)
    if error_by_name:
        return [
            model_name
            for model_name, _error in sorted(error_by_name.items(), key=lambda item: (item[1], item[0]))[:2]
        ]

    score_map = _parse_jsonish_mapping(row.get("teacher_eval_scores"))
    ranked: list[tuple[str, float]] = []
    for model_name, score in score_map.items():
        try:
            score_value = float(score)
        except (TypeError, ValueError):
            continue
        if np.isfinite(score_value):
            ranked.append((str(model_name).strip().lower(), score_value))
    ranked.sort(key=lambda item: (-item[1], item[0]))
    return [model_name for model_name, _score in ranked[:2]]


def _route_regret(row: pd.Series, requested_model: str) -> float | None:
    requested_model = str(requested_model or "").strip().lower()
    if not requested_model:
        return None
    error_by_name = _model_error_by_name(row)
    if requested_model not in error_by_name or not error_by_name:
        return None
    best_error = min(error_by_name.values())
    return float(error_by_name[requested_model] - best_error)


def _delta_vs_default(row: pd.Series, requested_model: str, default_expert: str) -> float | None:
    requested_model = str(requested_model or "").strip().lower()
    default_expert = str(default_expert or "").strip().lower()
    if not requested_model or not default_expert:
        return None
    error_by_name = _model_error_by_name(row)
    if requested_model not in error_by_name or default_expert not in error_by_name:
        return None
    return float(error_by_name[requested_model] - error_by_name[default_expert])


def _confidence_tier(row: pd.Series) -> str:
    for key in ("route_label_confidence", "route_bootstrap_confidence_tier", "routing_confidence_tier"):
        value = str(row.get(key) or "").strip().lower()
        if value:
            return value
    return ""


def _default_expert(row: pd.Series) -> str:
    for key in ("route_default_expert", "default_expert"):
        value = str(row.get(key) or "").strip().lower()
        if value in ALLOWED_ROUTING_MODELS:
            return value
    return ""


def _route_label(row: pd.Series) -> str:
    value = str(row.get("route_label") or "").strip().lower()
    if value == "keep_default":
        return value
    if value.startswith("override_to_"):
        return value
    return ""


def _route_mode(row: pd.Series) -> str:
    return "override" if _default_expert(row) and _route_label(row) else "exact"


def _parse_generated_route(
    *,
    generated: str,
    default_expert: str,
) -> dict[str, Any]:
    assistant_text, tool_calls = extract_tool_calls(
        generated,
        allowed_tool_names=["predict_time_series", "route_time_series"],
        max_calls=1,
    )
    requested_model = ""
    predicted_label = ""
    route_decision = ""
    valid_tool_call = False
    tool_name = ""

    if tool_calls:
        call = tool_calls[0]
        tool_name = str(call.name or "")
        if tool_name == "route_time_series":
            route_decision = str(call.arguments.get("decision") or "").strip().lower()
            if route_decision == "keep_default" and default_expert:
                requested_model = default_expert
                predicted_label = "keep_default"
                valid_tool_call = True
            elif route_decision == "override":
                requested_model = str(call.arguments.get("model_name") or "").strip().lower()
                if requested_model in ALLOWED_ROUTING_MODELS and requested_model != default_expert:
                    predicted_label = f"override_to_{requested_model}"
                    valid_tool_call = True
        elif tool_name == "predict_time_series":
            requested_model = str(call.arguments.get("model_name") or "").strip().lower()
            if requested_model in ALLOWED_ROUTING_MODELS:
                valid_tool_call = True
                route_decision = "keep_default" if requested_model == default_expert else "override"
                predicted_label = "keep_default" if requested_model == default_expert else f"override_to_{requested_model}"

    return {
        "assistant_text": assistant_text,
        "tool_call_count": len(tool_calls),
        "tool_name": tool_name,
        "requested_model": requested_model,
        "route_decision": route_decision,
        "predicted_label": predicted_label,
        "valid_tool_call": bool(valid_tool_call),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe routing greedy policy for a step-wise SFT checkpoint.")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--stepwise-parquet", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=64)
    parser.add_argument("--max-new-tokens", type=int, default=640)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    dataframe = pd.read_parquet(args.stepwise_parquet)
    routing = dataframe.loc[dataframe["turn_stage"].astype(str).str.lower() == "routing"].copy().reset_index(drop=True)
    if routing.empty:
        raise ValueError(f"No routing rows found in {args.stepwise_parquet}")
    routing = _take_evenly_spaced(routing, int(args.max_samples))

    tokenizer = AutoTokenizer.from_pretrained(str(args.model_path), trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        str(args.model_path),
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        trust_remote_code=True,
    )
    model.eval()

    samples_path = args.output_dir / "routing_greedy_probe_samples.jsonl"
    summary_path = args.output_dir / "routing_greedy_probe_summary.json"
    samples_path.unlink(missing_ok=True)

    results: list[dict[str, Any]] = []
    for row_idx, row in routing.iterrows():
        messages = [
            {
                "role": str(message.get("role") or ""),
                "content": str(message.get("content") or ""),
            }
            for message in list(row["messages"][:-1])
        ]
        prompt_text = tokenizer.apply_chat_template(
            messages,
            tools=_to_builtin(row["tools"]),
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=int(args.max_new_tokens),
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        generated = tokenizer.decode(outputs[0][inputs.input_ids.shape[1] :], skip_special_tokens=False)
        default_expert = _default_expert(row)
        route_parse = _parse_generated_route(generated=generated, default_expert=default_expert)
        requested_model = str(route_parse["requested_model"] or "")
        confidence_tier = _confidence_tier(row)
        top2_models = _top2_models(row)
        route_regret = _route_regret(row, requested_model)
        delta_vs_default = _delta_vs_default(row, requested_model, default_expert)
        route_mode = _route_mode(row)
        gold_label = _route_label(row)
        gold_is_override = bool(gold_label and gold_label != "keep_default")
        predicted_is_override = bool(route_parse["predicted_label"] and route_parse["predicted_label"] != "keep_default")
        result = {
            "row_index": int(row_idx),
            "source_sample_index": int(row.get("source_sample_index", -1)),
            "route_mode": route_mode,
            "default_expert": default_expert,
            "route_label": gold_label,
            "reference_teacher_model": str(row.get("reference_teacher_model") or ""),
            "selected_prediction_model": str(row.get("selected_prediction_model") or ""),
            "requested_model": requested_model,
            "predicted_label": str(route_parse["predicted_label"] or ""),
            "predicted_is_override": predicted_is_override,
            "gold_is_override": gold_is_override,
            "confidence_tier": confidence_tier,
            "tool_name": str(route_parse["tool_name"] or ""),
            "valid_tool_call": bool(route_parse["valid_tool_call"]),
            "tool_call_count": int(route_parse["tool_call_count"]),
            "top2_models": top2_models,
            "in_top2": bool(requested_model and requested_model in top2_models),
            "route_regret": route_regret,
            "delta_vs_default": delta_vs_default,
            "assistant_text": str(route_parse["assistant_text"] or ""),
            "raw_output_head": generated[:1200],
            "exact_match": (
                bool(gold_label and str(route_parse["predicted_label"] or "") == gold_label)
                if route_mode == "override"
                else bool(requested_model and requested_model == str(row.get("reference_teacher_model") or "").strip().lower())
            ),
        }
        results.append(result)
        with samples_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
        if row_idx == 0 or (row_idx + 1) % 8 == 0 or row_idx + 1 == len(routing):
            print(
                f"[routing_probe] {row_idx + 1}/{len(routing)} tool={result['tool_name'] or '<none>'} "
                f"requested={requested_model or '<none>'} mode={route_mode}",
                flush=True,
            )

    requested_distribution = Counter(result["requested_model"] or "<none>" for result in results)
    predicted_label_distribution = Counter(result["predicted_label"] or "<none>" for result in results)
    exact_matches = [int(result["exact_match"]) for result in results]
    top2_matches = [int(result["in_top2"]) for result in results]
    valid_regrets = [float(result["route_regret"]) for result in results if result["route_regret"] is not None]
    delta_values = [float(result["delta_vs_default"]) for result in results if result["delta_vs_default"] is not None]
    high_conf_results = [result for result in results if result["confidence_tier"] in {"mid", "high"}]
    high_conf_exact_matches = [int(result["exact_match"]) for result in high_conf_results]
    high_conf_top2_matches = [int(result["in_top2"]) for result in high_conf_results]
    high_conf_regrets = [
        float(result["route_regret"]) for result in high_conf_results if result["route_regret"] is not None
    ]
    single_model_max_share = (
        max(requested_distribution.values()) / max(len(results), 1) if requested_distribution else 0.0
    )

    override_results = [result for result in results if result["route_mode"] == "override"]
    gold_override_results = [result for result in override_results if result["gold_is_override"]]
    tp = sum(int(result["gold_is_override"] and result["predicted_is_override"]) for result in override_results)
    fp = sum(int((not result["gold_is_override"]) and result["predicted_is_override"]) for result in override_results)
    fn = sum(int(result["gold_is_override"] and (not result["predicted_is_override"])) for result in override_results)
    tn = sum(int((not result["gold_is_override"]) and (not result["predicted_is_override"])) for result in override_results)
    override_precision = float(tp / max(tp + fp, 1))
    override_recall = float(tp / max(tp + fn, 1))
    override_f1 = (
        float(2 * override_precision * override_recall / max(override_precision + override_recall, 1e-8))
        if (override_precision + override_recall) > 0
        else 0.0
    )
    keep_precision = float(tn / max(tn + fn, 1))
    keep_recall = float(tn / max(tn + fp, 1))
    keep_f1 = (
        float(2 * keep_precision * keep_recall / max(keep_precision + keep_recall, 1e-8))
        if (keep_precision + keep_recall) > 0
        else 0.0
    )
    keep_vs_override_f1 = float((keep_f1 + override_f1) / 2.0)
    override_exact_matches = [int(result["exact_match"]) for result in gold_override_results]
    override_top2_matches = [int(result["in_top2"]) for result in gold_override_results]

    summary = {
        "count": len(results),
        "valid_tool_call_rate": sum(int(result["valid_tool_call"]) for result in results) / max(len(results), 1),
        "requested_model_distribution": dict(requested_distribution),
        "predicted_label_distribution": dict(predicted_label_distribution),
        "agree_with_reference_teacher_rate": sum(
            int(result["requested_model"] == result["reference_teacher_model"]) for result in results
        )
        / max(len(results), 1),
        "agree_with_heuristic_label_rate": sum(
            int(result["requested_model"] == result["selected_prediction_model"]) for result in results
        )
        / max(len(results), 1),
        "overall_exact_agreement": sum(exact_matches) / max(len(results), 1),
        "high_conf_count": len(high_conf_results),
        "high_conf_exact_agreement": sum(high_conf_exact_matches) / max(len(high_conf_results), 1),
        "top2_agreement": sum(top2_matches) / max(len(results), 1),
        "high_conf_top2_agreement": sum(high_conf_top2_matches) / max(len(high_conf_results), 1),
        "mean_route_regret": float(np.mean(valid_regrets)) if valid_regrets else None,
        "high_conf_mean_route_regret": float(np.mean(high_conf_regrets)) if high_conf_regrets else None,
        "single_model_max_share": single_model_max_share,
        "predicted_model_coverage": len([name for name in requested_distribution if name in ALLOWED_ROUTING_MODELS]),
        "reference_teacher_distribution": dict(Counter(result["reference_teacher_model"] for result in results)),
        "heuristic_label_distribution": dict(Counter(result["selected_prediction_model"] for result in results)),
        "route_override_count": len(override_results),
        "delta_vs_default_mean": float(np.mean(delta_values)) if delta_values else None,
        "keep_vs_override_f1": keep_vs_override_f1,
        "override_precision": override_precision,
        "override_recall": override_recall,
        "override_f1": override_f1,
        "override_subset_count": len(gold_override_results),
        "override_subset_exact_agreement": sum(override_exact_matches) / max(len(gold_override_results), 1),
        "override_subset_top2_agreement": sum(override_top2_matches) / max(len(gold_override_results), 1),
        "keep_default_share": (
            sum(int(result["predicted_label"] == "keep_default") for result in override_results)
            / max(len(override_results), 1)
        ),
        "override_share": (
            sum(int(result["predicted_is_override"]) for result in override_results) / max(len(override_results), 1)
        ),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
