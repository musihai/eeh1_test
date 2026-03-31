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

from recipe.time_series_forecast.candidate_selection_support import (
    materialize_candidate_selection,
    parse_candidate_selection_protocol,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe the v19 final-select policy.")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--stepwise-parquet", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=64)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


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


def _json_mapping(value: Any) -> dict[str, Any]:
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


def _candidate_error_map(row: pd.Series) -> dict[str, float]:
    details = _json_mapping(row.get("candidate_score_details"))
    output: dict[str, float] = {}
    for candidate_id, payload in details.items():
        if not isinstance(payload, dict):
            continue
        try:
            error = float(payload.get("orig_mse"))
        except (TypeError, ValueError):
            continue
        output[str(candidate_id)] = error
    return output


def _candidate_score_map(row: pd.Series) -> dict[str, float]:
    details = _json_mapping(row.get("candidate_score_details"))
    output: dict[str, float] = {}
    for candidate_id, payload in details.items():
        if not isinstance(payload, dict):
            continue
        try:
            score = float(payload.get("score"))
        except (TypeError, ValueError):
            continue
        output[str(candidate_id)] = score
    return output


def _top2_candidates(row: pd.Series) -> list[str]:
    score_map = _candidate_score_map(row)
    if score_map:
        return [candidate_id for candidate_id, _ in sorted(score_map.items(), key=lambda item: (-item[1], item[0]))[:2]]
    error_map = _candidate_error_map(row)
    return [candidate_id for candidate_id, _ in sorted(error_map.items(), key=lambda item: (item[1], item[0]))[:2]]


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    dataframe = pd.read_parquet(args.stepwise_parquet)
    final_rows = dataframe.loc[dataframe["turn_stage"].astype(str).str.lower() == "final_select"].copy().reset_index(drop=True)
    if final_rows.empty:
        raise ValueError(f"No final_select rows found in {args.stepwise_parquet}")
    final_rows = _take_evenly_spaced(final_rows, int(args.max_samples))

    tokenizer = AutoTokenizer.from_pretrained(str(args.model_path), trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    device = str(args.device or "cuda").strip().lower()
    torch_dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        str(args.model_path),
        torch_dtype=torch_dtype,
        device_map=device,
        trust_remote_code=True,
    )
    model.eval()

    samples_path = args.output_dir / "final_select_probe_samples.jsonl"
    summary_path = args.output_dir / "final_select_probe_summary.json"
    samples_path.unlink(missing_ok=True)

    results: list[dict[str, Any]] = []
    for row_idx, row in final_rows.iterrows():
        messages = [
            {"role": str(message.get("role") or ""), "content": str(message.get("content") or "")}
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
        generated = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=False)
        candidate_prediction_text_map = _json_mapping(row.get("candidate_prediction_text_map"))
        think_text, selected_candidate_id, parse_mode, reject_reason = parse_candidate_selection_protocol(
            generated,
            allowed_candidate_ids=list(candidate_prediction_text_map.keys()),
        )
        rendered_answer, materialized_candidate_id, materialize_mode, materialize_reject_reason = materialize_candidate_selection(
            response_text=generated,
            candidate_prediction_text_map=candidate_prediction_text_map,
        )
        selected_candidate_id = materialized_candidate_id or selected_candidate_id or ""
        error_map = _candidate_error_map(row)
        default_candidate_id = str(row.get("default_candidate_id") or "")
        selected_error = error_map.get(selected_candidate_id)
        default_error = error_map.get(default_candidate_id)
        final_vs_default = None
        if selected_error is not None and default_error is not None:
            final_vs_default = float(selected_error - default_error)
        result = {
            "row_index": int(row_idx),
            "source_sample_index": int(row.get("source_sample_index", -1)),
            "uid": str(row.get("uid") or ""),
            "risk_label": str(row.get("risk_label") or ""),
            "default_candidate_id": default_candidate_id,
            "gold_candidate_id": str(row.get("final_candidate_label") or ""),
            "selected_candidate_id": selected_candidate_id,
            "parse_mode": parse_mode,
            "reject_reason": reject_reason,
            "materialize_mode": materialize_mode,
            "materialize_reject_reason": materialize_reject_reason,
            "protocol_ok": bool(selected_candidate_id),
            "materialize_ok": rendered_answer is not None,
            "exact_match": bool(selected_candidate_id and selected_candidate_id == str(row.get("final_candidate_label") or "")),
            "top2_hit": bool(selected_candidate_id and selected_candidate_id in _top2_candidates(row)),
            "final_vs_default": final_vs_default,
            "generated": generated,
        }
        with samples_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
        results.append(result)

    selected_distribution = Counter(result["selected_candidate_id"] for result in results if result["selected_candidate_id"])
    final_vs_default_values = [result["final_vs_default"] for result in results if result["final_vs_default"] is not None]
    risky_values = [
        result["final_vs_default"]
        for result in results
        if result["final_vs_default"] is not None and result["risk_label"] == "default_risky"
    ]
    exact_values = [bool(result["exact_match"]) for result in results]
    top2_values = [bool(result["top2_hit"]) for result in results]
    summary = {
        "count": int(len(results)),
        "protocol_ok_rate": float(sum(1 for result in results if result["protocol_ok"]) / len(results)) if results else 0.0,
        "materialize_ok_rate": float(sum(1 for result in results if result["materialize_ok"]) / len(results)) if results else 0.0,
        "exact_match_rate": float(np.mean(exact_values)) if exact_values else 0.0,
        "top2_hit_rate": float(np.mean(top2_values)) if top2_values else 0.0,
        "selected_candidate_distribution": {str(k): int(v) for k, v in sorted(selected_distribution.items())},
        "single_candidate_max_share": float(max(selected_distribution.values()) / len(results)) if selected_distribution and results else 0.0,
        "final_vs_default_mean": float(np.mean(final_vs_default_values)) if final_vs_default_values else None,
        "risky_final_vs_default_mean": float(np.mean(risky_values)) if risky_values else None,
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
