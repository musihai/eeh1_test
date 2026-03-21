from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_INPUT_DIR = Path("dataset/ett_sft_etth1_runtime_ot_teacher200_paper_same2")
DEFAULT_OUTPUT_DIR = Path("dataset/ett_sft_etth1_runtime_ot_teacher200_paper_same2_subset")


def select_evenly_spaced_rows(dataframe: pd.DataFrame, count: int) -> pd.DataFrame:
    if count <= 0 or len(dataframe) <= count:
        return dataframe.copy()

    sorted_df = dataframe.sort_values(by="sample_index", kind="stable").reset_index(drop=True)
    total = len(sorted_df)
    chosen_positions = []
    for idx in range(count):
        position = round(idx * (total - 1) / (count - 1))
        chosen_positions.append(position)

    deduped_positions = []
    seen = set()
    for position in chosen_positions:
        if position not in seen:
            deduped_positions.append(position)
            seen.add(position)

    cursor = 0
    while len(deduped_positions) < count and cursor < total:
        if cursor not in seen:
            deduped_positions.append(cursor)
            seen.add(cursor)
        cursor += 1

    deduped_positions.sort()
    return sorted_df.iloc[deduped_positions].reset_index(drop=True)


def write_metadata(output_dir: Path, **payload: Any) -> None:
    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def build_subset(
    *,
    input_dir: Path,
    output_dir: Path,
    train_samples: int,
    val_samples: int,
    test_samples: int,
) -> dict[str, int]:
    output_dir.mkdir(parents=True, exist_ok=True)

    counts: dict[str, int] = {}
    split_plan = (
        ("train", train_samples),
        ("val", val_samples),
        ("test", test_samples),
    )

    for split_name, sample_limit in split_plan:
        input_path = input_dir / f"{split_name}.parquet"
        if not input_path.exists():
            counts[f"{split_name}_samples"] = 0
            continue

        frame = pd.read_parquet(input_path)
        subset = select_evenly_spaced_rows(frame, sample_limit)
        subset.to_parquet(output_dir / f"{split_name}.parquet", index=False)
        counts[f"{split_name}_samples"] = len(subset)

    write_metadata(
        output_dir,
        selection_method="evenly_spaced_by_sample_index",
        source_dir=str(input_dir),
        train_samples=counts.get("train_samples", 0),
        val_samples=counts.get("val_samples", 0),
        test_samples=counts.get("test_samples", 0),
        requested_train_samples=train_samples,
        requested_val_samples=val_samples,
        requested_test_samples=test_samples,
    )
    return counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a deterministic ETTh1 SFT subset for paper-like runs.")
    parser.add_argument("--input-dir", default=str(DEFAULT_INPUT_DIR), help="Source SFT parquet directory.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Output subset directory.")
    parser.add_argument("--train-samples", type=int, default=200, help="Number of train samples to keep.")
    parser.add_argument("--val-samples", type=int, default=64, help="Number of validation samples to keep.")
    parser.add_argument("--test-samples", type=int, default=128, help="Number of test samples to keep.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    counts = build_subset(
        input_dir=Path(args.input_dir),
        output_dir=Path(args.output_dir),
        train_samples=args.train_samples,
        val_samples=args.val_samples,
        test_samples=args.test_samples,
    )
    print(
        "Built ETTh1 SFT subset:",
        f"train={counts.get('train_samples', 0)}",
        f"val={counts.get('val_samples', 0)}",
        f"test={counts.get('test_samples', 0)}",
    )


if __name__ == "__main__":
    main()
