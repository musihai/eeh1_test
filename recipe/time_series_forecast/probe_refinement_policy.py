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

from recipe.time_series_forecast.reward import extract_values_from_time_series_string
from recipe.time_series_forecast.refinement_support import materialize_refinement_decision
from recipe.time_series_forecast.reward_protocol import clamp_turn3_answer_horizon, parse_final_answer_protocol


def _take_evenly_spaced(dataframe: pd.DataFrame, max_samples: int) -> pd.DataFrame:
    if max_samples <= 0 or len(dataframe) <= max_samples:
        return dataframe.reset_index(drop=True)
    indexes = sorted({round(i * (len(dataframe) - 1) / (max_samples - 1)) for i in range(max_samples)})
    return dataframe.iloc[indexes].reset_index(drop=True)


def _take_stratified_refinement_rows(dataframe: pd.DataFrame, max_per_type: int) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for target_type in ("local_refine", "validated_keep"):
        subset = dataframe.loc[dataframe["turn3_target_type"].astype(str).str.lower() == target_type].copy()
        if subset.empty:
            continue
        frames.append(_take_evenly_spaced(subset, max_per_type))
    if not frames:
        return dataframe.iloc[:0].copy()
    return pd.concat(frames, ignore_index=True)


def _allclose(values_a: list[float], values_b: list[float], atol: float = 1e-6) -> bool:
    return bool(values_a and len(values_a) == len(values_b) and np.allclose(values_a, values_b, atol=atol))


def _mse(values_a: list[float], values_b: list[float]) -> float:
    if not values_a or len(values_a) != len(values_b):
        return float("inf")
    arr_a = np.asarray(values_a, dtype=float)
    arr_b = np.asarray(values_b, dtype=float)
    return float(np.mean((arr_a - arr_b) ** 2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe refinement copy-vs-edit policy for a step-wise SFT checkpoint.")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--stepwise-parquet", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-samples-per-type", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=3200)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    dataframe = pd.read_parquet(args.stepwise_parquet)
    refinement = dataframe.loc[dataframe["turn_stage"].astype(str).str.lower() == "refinement"].copy().reset_index(drop=True)
    if refinement.empty:
        raise ValueError(f"No refinement rows found in {args.stepwise_parquet}")
    refinement = _take_stratified_refinement_rows(refinement, int(args.max_samples_per_type))

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

    samples_path = args.output_dir / "refinement_policy_probe_samples.jsonl"
    summary_path = args.output_dir / "refinement_policy_probe_summary.json"
    samples_path.unlink(missing_ok=True)

    results: list[dict[str, Any]] = []
    for row_idx, row in refinement.iterrows():
        messages = [
            {
                "role": str(message.get("role") or ""),
                "content": str(message.get("content") or ""),
            }
            for message in list(row["messages"][:-1])
        ]
        prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
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

        rendered_answer = None
        decision_name = ""
        candidate_prediction_text_map_raw = row.get("refinement_candidate_prediction_text_map")
        if candidate_prediction_text_map_raw:
            try:
                candidate_prediction_text_map = json.loads(str(candidate_prediction_text_map_raw))
            except Exception:
                candidate_prediction_text_map = {}
            rendered_answer, decision_name, parse_mode, reject_reason = materialize_refinement_decision(
                response_text=generated,
                candidate_prediction_text_map=candidate_prediction_text_map,
            )
            if rendered_answer is not None:
                parsed_answer, parse_mode, reject_reason = parse_final_answer_protocol(
                    rendered_answer,
                    int(row.get("forecast_horizon") or 96),
                    allow_recovery=False,
                )
            else:
                parsed_answer = None
        else:
            clamped = clamp_turn3_answer_horizon(generated, expected_len=int(row.get("forecast_horizon") or 96))
            parsed_answer, parse_mode, reject_reason = parse_final_answer_protocol(
                clamped,
                int(row.get("forecast_horizon") or 96),
                allow_recovery=False,
            )
        generated_values = extract_values_from_time_series_string(parsed_answer or "")
        base_values = extract_values_from_time_series_string(str(row.get("base_teacher_prediction_text") or ""))
        refined_values = extract_values_from_time_series_string(str(row.get("refined_prediction_text") or ""))

        copy_base = _allclose(generated_values, base_values)
        copy_refined = _allclose(generated_values, refined_values)
        mse_to_base = _mse(generated_values, base_values)
        mse_to_refined = _mse(generated_values, refined_values)
        closer_to_refined = bool(np.isfinite(mse_to_refined) and mse_to_refined + 1e-8 < mse_to_base)

        result = {
            "row_index": int(row_idx),
            "source_sample_index": int(row.get("source_sample_index", -1)),
            "turn3_target_type": str(row.get("turn3_target_type") or ""),
            "selected_prediction_model": str(row.get("selected_prediction_model") or ""),
            "strict_ok": parsed_answer is not None,
            "parse_mode": parse_mode,
            "reject_reason": reject_reason,
            "decision_name": decision_name,
            "copy_base": bool(copy_base),
            "copy_refined": bool(copy_refined),
            "changed_vs_base": bool(not copy_base and np.isfinite(mse_to_base)),
            "closer_to_refined": bool(closer_to_refined),
            "mse_to_base": mse_to_base,
            "mse_to_refined": mse_to_refined,
            "raw_output_head": generated[:1200],
        }
        results.append(result)
        with samples_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
        if row_idx == 0 or (row_idx + 1) % 8 == 0 or row_idx + 1 == len(refinement):
            print(
                f"[refinement_probe] {row_idx + 1}/{len(refinement)} "
                f"type={result['turn3_target_type']} copy_base={result['copy_base']} "
                f"copy_refined={result['copy_refined']}",
                flush=True,
            )

    def _subset(target_type: str) -> list[dict[str, Any]]:
        return [item for item in results if item["turn3_target_type"] == target_type]

    def _rate(items: list[dict[str, Any]], key: str) -> float:
        if not items:
            return float("nan")
        return sum(int(bool(item[key])) for item in items) / len(items)

    local_refine = _subset("local_refine")
    validated_keep = _subset("validated_keep")
    summary = {
        "count": len(results),
        "strict_ok_rate": _rate(results, "strict_ok"),
        "turn3_target_type_distribution": dict(Counter(item["turn3_target_type"] for item in results)),
        "parse_mode_counts": dict(Counter(item["parse_mode"] for item in results)),
        "reject_reason_counts": dict(Counter(item["reject_reason"] for item in results if item["reject_reason"])),
        "local_refine_count": len(local_refine),
        "local_refine_copy_base_rate": _rate(local_refine, "copy_base"),
        "local_refine_copy_refined_rate": _rate(local_refine, "copy_refined"),
        "local_refine_changed_rate": _rate(local_refine, "changed_vs_base"),
        "local_refine_closer_to_refined_rate": _rate(local_refine, "closer_to_refined"),
        "validated_keep_count": len(validated_keep),
        "validated_keep_copy_base_rate": _rate(validated_keep, "copy_base"),
        "validated_keep_changed_rate": _rate(validated_keep, "changed_vs_base"),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
