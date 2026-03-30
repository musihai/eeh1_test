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
    indexes = sorted({round(i * (len(dataframe) - 1) / (max_samples - 1)) for i in range(max_samples)})
    return dataframe.iloc[indexes].reset_index(drop=True)


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
        assistant_text, tool_calls = extract_tool_calls(
            generated,
            allowed_tool_names=["predict_time_series"],
            max_calls=1,
        )
        requested_model = ""
        if tool_calls:
            requested_model = str(tool_calls[0].arguments.get("model_name") or "").strip().lower()
        result = {
            "row_index": int(row_idx),
            "source_sample_index": int(row.get("source_sample_index", -1)),
            "reference_teacher_model": str(row.get("reference_teacher_model") or ""),
            "selected_prediction_model": str(row.get("selected_prediction_model") or ""),
            "requested_model": requested_model,
            "valid_tool_call": requested_model in ALLOWED_ROUTING_MODELS,
            "tool_call_count": len(tool_calls),
            "assistant_text": assistant_text,
            "raw_output_head": generated[:1200],
        }
        results.append(result)
        with samples_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
        if row_idx == 0 or (row_idx + 1) % 8 == 0 or row_idx + 1 == len(routing):
            print(
                f"[routing_probe] {row_idx + 1}/{len(routing)} requested="
                f"{requested_model or '<none>'} ref={result['reference_teacher_model']} "
                f"heur={result['selected_prediction_model']}",
                flush=True,
            )

    requested_distribution = Counter(result["requested_model"] or "<none>" for result in results)
    summary = {
        "count": len(results),
        "valid_tool_call_rate": sum(int(result["valid_tool_call"]) for result in results) / max(len(results), 1),
        "requested_model_distribution": dict(requested_distribution),
        "agree_with_reference_teacher_rate": sum(
            int(result["requested_model"] == result["reference_teacher_model"]) for result in results
        )
        / max(len(results), 1),
        "agree_with_heuristic_label_rate": sum(
            int(result["requested_model"] == result["selected_prediction_model"]) for result in results
        )
        / max(len(results), 1),
        "reference_teacher_distribution": dict(Counter(result["reference_teacher_model"] for result in results)),
        "heuristic_label_distribution": dict(Counter(result["selected_prediction_model"] for result in results)),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
