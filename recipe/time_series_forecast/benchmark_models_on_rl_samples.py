from __future__ import annotations

import argparse
import asyncio
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

from recipe.time_series_forecast.build_etth1_high_quality_sft import LocalTeacherPredictor
from recipe.time_series_forecast.task_protocol import parse_task_prompt
from recipe.time_series_forecast.utils import parse_time_series_to_dataframe


@dataclass(frozen=True)
class SplitSpec:
    name: str
    path: Path
    max_samples: int


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def evenly_spaced_records(records: Sequence[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    if count <= 0 or len(records) <= count:
        return list(records)
    if count == 1:
        return [records[0]]
    ordered = sorted(records, key=lambda item: int(item.get("index", 0)))
    selected: list[dict[str, Any]] = []
    total = len(ordered)
    for bucket in range(count):
        pos = round(bucket * (total - 1) / (count - 1))
        selected.append(ordered[pos])
    # dedup if round() collided
    dedup: dict[int, dict[str, Any]] = {}
    for row in selected:
        dedup[int(row.get("index", 0))] = row
    out = sorted(dedup.values(), key=lambda item: int(item.get("index", 0)))
    if len(out) < count:
        for row in ordered:
            idx = int(row.get("index", 0))
            if idx not in dedup:
                out.append(row)
                dedup[idx] = row
                if len(out) >= count:
                    break
    return sorted(out, key=lambda item: int(item.get("index", 0)))[:count]


def parse_ground_truth_values(ground_truth: str) -> list[float]:
    values: list[float] = []
    for line in str(ground_truth or "").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        token = stripped.split()[-1]
        try:
            values.append(float(token))
        except ValueError:
            continue
    return values


def compute_mae_mse(y_true: list[float], y_pred: list[float]) -> tuple[float, float, int]:
    n = min(len(y_true), len(y_pred))
    if n <= 0:
        return float("nan"), float("nan"), 0
    abs_sum = 0.0
    sq_sum = 0.0
    for i in range(n):
        diff = float(y_pred[i]) - float(y_true[i])
        abs_sum += abs(diff)
        sq_sum += diff * diff
    return abs_sum / n, sq_sum / n, n


async def evaluate_one_sample(
    sample: dict[str, Any],
    models: Sequence[str],
    predictor: LocalTeacherPredictor,
) -> list[dict[str, Any]]:
    raw_prompt = sample["raw_prompt"][0]["content"]
    reward_model = sample.get("reward_model", {}) if isinstance(sample.get("reward_model"), dict) else {}
    ground_truth = str(reward_model.get("ground_truth", "") or "")
    gt_values = parse_ground_truth_values(ground_truth)

    task_spec = parse_task_prompt(raw_prompt, data_source=sample.get("data_source"))
    historical_data = task_spec.historical_data or raw_prompt
    target_column = task_spec.target_column or "OT"
    data_source = task_spec.data_source or str(sample.get("data_source") or "ETTh1")
    forecast_horizon = int(task_spec.forecast_horizon or 96)

    context_df = parse_time_series_to_dataframe(
        historical_data,
        series_id=data_source,
        target_column=target_column,
    )

    sample_results: list[dict[str, Any]] = []
    for model_name in models:
        try:
            pred_df = await predictor.predict(
                context_df=context_df,
                prediction_length=forecast_horizon,
                model_name=model_name,
            )
            pred_values = [float(v) for v in pred_df["target_0.5"].tolist()]
            mae, mse, aligned_n = compute_mae_mse(gt_values, pred_values)
            sample_results.append(
                {
                    "model": model_name,
                    "sample_index": int(sample.get("index", -1)),
                    "uid": sample.get("uid", ""),
                    "gt_len": len(gt_values),
                    "pred_len": len(pred_values),
                    "aligned_n": int(aligned_n),
                    "mae": float(mae),
                    "mse": float(mse),
                    "error": "",
                }
            )
        except Exception as exc:
            sample_results.append(
                {
                    "model": model_name,
                    "sample_index": int(sample.get("index", -1)),
                    "uid": sample.get("uid", ""),
                    "gt_len": len(gt_values),
                    "pred_len": 0,
                    "aligned_n": 0,
                    "mae": float("nan"),
                    "mse": float("nan"),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    return sample_results


async def evaluate_split(
    split_name: str,
    records: Sequence[dict[str, Any]],
    models: Sequence[str],
    predictor: LocalTeacherPredictor,
    concurrency: int,
) -> list[dict[str, Any]]:
    sem = asyncio.Semaphore(concurrency)
    rows: list[dict[str, Any]] = []

    async def _runner(sample: dict[str, Any]) -> None:
        async with sem:
            one = await evaluate_one_sample(sample=sample, models=models, predictor=predictor)
            for row in one:
                row["split"] = split_name
                rows.append(row)

    await asyncio.gather(*[_runner(sample) for sample in records])
    return rows


def summarize(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = f"{row['split']}::{row['model']}"
        grouped[key].append(row)

    summary: dict[str, Any] = {}
    for key, items in sorted(grouped.items()):
        split, model = key.split("::", 1)
        valid = [item for item in items if not item["error"] and math.isfinite(float(item["mse"])) and item["aligned_n"] > 0]
        failures = [item for item in items if item["error"]]
        if valid:
            mae_mean = sum(float(v["mae"]) for v in valid) / len(valid)
            mse_mean = sum(float(v["mse"]) for v in valid) / len(valid)
        else:
            mae_mean = float("nan")
            mse_mean = float("nan")
        summary[key] = {
            "split": split,
            "model": model,
            "evaluated": len(items),
            "valid": len(valid),
            "failures": len(failures),
            "mae_mean": float(mae_mean),
            "mse_mean": float(mse_mean),
        }
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark chronos2/arima/patchtst/itransformer on RL samples.")
    parser.add_argument("--train-jsonl", default="dataset/ett_rl_etth1_paper_aligned_ot_20260315_151424/train.jsonl")
    parser.add_argument("--val-jsonl", default="dataset/ett_rl_etth1_paper_aligned_ot_20260315_151424/val.jsonl")
    parser.add_argument("--test-jsonl", default="dataset/ett_rl_etth1_paper_aligned_ot_20260315_151424/test.jsonl")
    parser.add_argument("--train-samples", type=int, default=256, help="0 means use all")
    parser.add_argument("--val-samples", type=int, default=256, help="0 means use all")
    parser.add_argument("--test-samples", type=int, default=0, help="0 means skip when --include-test is false")
    parser.add_argument("--include-test", action="store_true", help="Also benchmark RL test split")
    parser.add_argument("--models", default="patchtst,itransformer,chronos2,arima")
    parser.add_argument("--predictor-device", default="cuda")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--output-dir", default="artifacts/reports")
    return parser.parse_args()


def build_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# RL样本四模型对比（MSE/MAE）",
        "",
        "| split | model | evaluated | valid | failures | MAE(mean) | MSE(mean) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in sorted(summary.items(), key=lambda kv: (kv[1]["split"], kv[1]["model"])):
        lines.append(
            "| {split} | {model} | {evaluated} | {valid} | {failures} | {mae:.6f} | {mse:.6f} |".format(
                split=row["split"],
                model=row["model"],
                evaluated=row["evaluated"],
                valid=row["valid"],
                failures=row["failures"],
                mae=float(row["mae_mean"]),
                mse=float(row["mse_mean"]),
            )
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    models = [item.strip().lower() for item in args.models.split(",") if item.strip()]

    split_specs: list[SplitSpec] = [
        SplitSpec("train", Path(args.train_jsonl), args.train_samples or 12060),
        SplitSpec("val", Path(args.val_jsonl), args.val_samples or 1722),
    ]
    if args.include_test:
        split_specs.append(SplitSpec("test", Path(args.test_jsonl), args.test_samples or 3065))

    predictor = LocalTeacherPredictor(args.predictor_device)
    all_rows: list[dict[str, Any]] = []
    split_meta: dict[str, Any] = {}
    
    # Increase concurrency for better GPU utilization
    effective_concurrency = max(8, int(args.concurrency))
    print(f"[BENCH] Using concurrency={effective_concurrency}, device={args.predictor_device}")

    for spec in split_specs:
        if not spec.path.exists():
            raise FileNotFoundError(f"Split file not found: {spec.path}")
        records = load_jsonl(spec.path)
        selected = evenly_spaced_records(records, spec.max_samples) if spec.max_samples > 0 else list(records)
        split_meta[spec.name] = {
            "path": str(spec.path),
            "total_records": len(records),
            "selected_records": len(selected),
        }
        rows = asyncio.run(
            evaluate_split(
                split_name=spec.name,
                records=selected,
                models=models,
                predictor=predictor,
                concurrency=effective_concurrency,
            )
        )
        all_rows.extend(rows)

    summary = summarize(all_rows)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = output_dir / f"rl_model_benchmark_{ts}.json"
    md_path = output_dir / f"rl_model_benchmark_{ts}.md"

    payload = {
        "models": models,
        "split_meta": split_meta,
        "summary": summary,
        "rows": all_rows,
    }
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    md_path.write_text(build_markdown(summary), encoding="utf-8")

    print(f"[BENCH] wrote json: {json_path}")
    print(f"[BENCH] wrote markdown: {md_path}")
    for _, row in sorted(summary.items(), key=lambda kv: (kv[1]["split"], kv[1]["model"])):
        print(
            "[BENCH] split={split} model={model} valid={valid}/{evaluated} "
            "MAE={mae:.6f} MSE={mse:.6f}".format(
                split=row["split"],
                model=row["model"],
                valid=row["valid"],
                evaluated=row["evaluated"],
                mae=float(row["mae_mean"]),
                mse=float(row["mse_mean"]),
            )
        )


if __name__ == "__main__":
    main()
