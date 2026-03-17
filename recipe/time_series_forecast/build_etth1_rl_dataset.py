from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


DEFAULT_CSV_PATH = Path("dataset/ETT-small/ETTh1.csv")
DEFAULT_OUTPUT_DIR = Path("dataset/ett_rl_etth1_paper_aligned_ot_20260315_151424")

# These row counts reproduce the historical split sizes that the rest of the
# repo was built around:
# train windows = 12251 - 96 - 96 + 1 = 12060
# val windows   = 1913  - 96 - 96 + 1 = 1722
# test windows  = 3256  - 96 - 96 + 1 = 3065
DEFAULT_TRAIN_ROWS = 12251
DEFAULT_VAL_ROWS = 1913
DEFAULT_TEST_ROWS = 3256


@dataclass(frozen=True)
class SplitConfig:
    name: str
    start_row: int
    end_row: int

    @property
    def num_rows(self) -> int:
        return self.end_row - self.start_row


def build_prompt(
    historical_values: Iterable[float],
    *,
    lookback_window: int,
    forecast_horizon: int,
    target_column: str,
) -> str:
    value_lines = "\n".join(f"{float(value):.4f}" for value in historical_values)
    return (
        "[Task] Single-variable time-series forecasting.\n"
        f"Target Column: {target_column}\n"
        f"Lookback Window: {lookback_window}\n"
        f"Forecast Horizon: {forecast_horizon}\n"
        "Requirements:\n"
        "1) Extract feature evidence before selecting a forecasting model.\n"
        "2) Choose one model from the enabled experts and then predict.\n"
        "3) Follow the required output protocol with <think> and <answer>.\n"
        "Historical Data:\n"
        f"{value_lines}"
    )


def build_ground_truth(forecast_df: pd.DataFrame, *, target_column: str) -> str:
    return "\n".join(
        f"{row.date} {float(getattr(row, target_column)):.4f}"
        for row in forecast_df.itertuples(index=False)
    )


def build_split_configs(
    *,
    total_rows: int,
    train_rows: int,
    val_rows: int,
    test_rows: int,
) -> list[SplitConfig]:
    if train_rows + val_rows + test_rows != total_rows:
        raise ValueError(
            f"Split row counts must sum to total rows. "
            f"Got train={train_rows}, val={val_rows}, test={test_rows}, total={total_rows}."
        )

    return [
        SplitConfig(name="train", start_row=0, end_row=train_rows),
        SplitConfig(name="val", start_row=train_rows, end_row=train_rows + val_rows),
        SplitConfig(name="test", start_row=train_rows + val_rows, end_row=total_rows),
    ]


def iter_split_samples(
    df: pd.DataFrame,
    split: SplitConfig,
    *,
    lookback_window: int,
    forecast_horizon: int,
    target_column: str,
) -> Iterable[dict]:
    split_df = df.iloc[split.start_row : split.end_row].reset_index(drop=True)
    window_span = lookback_window + forecast_horizon
    num_samples = len(split_df) - window_span + 1
    if num_samples <= 0:
        raise ValueError(
            f"Split {split.name} is too short for lookback={lookback_window} and horizon={forecast_horizon}. "
            f"Rows: {len(split_df)}"
        )

    for sample_index in range(num_samples):
        hist_end = sample_index + lookback_window
        forecast_end = hist_end + forecast_horizon
        historical = split_df.iloc[sample_index:hist_end]
        forecast = split_df.iloc[hist_end:forecast_end]

        prompt_text = build_prompt(
            historical[target_column].tolist(),
            lookback_window=lookback_window,
            forecast_horizon=forecast_horizon,
            target_column=target_column,
        )
        ground_truth = build_ground_truth(forecast, target_column=target_column)

        yield {
            "index": sample_index,
            "uid": f"etth1-{split.name}-{sample_index:05d}",
            "agent_name": "time_series_forecast_agent",
            "data_source": "ETTh1",
            "target_column": target_column,
            "lookback_window": lookback_window,
            "forecast_horizon": forecast_horizon,
            "split": split.name,
            "raw_prompt": [{"role": "user", "content": prompt_text}],
            "reward_model": {
                "style": "rule",
                "ground_truth": ground_truth,
            },
        }


def write_jsonl(records: Iterable[dict], output_path: Path) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rebuild ETTh1 RL jsonl datasets from ETTh1.csv.")
    parser.add_argument("--csv-path", default=str(DEFAULT_CSV_PATH), help="Path to ETTh1.csv.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Output RL dataset directory.")
    parser.add_argument("--lookback-window", type=int, default=96, help="Historical window length.")
    parser.add_argument("--forecast-horizon", type=int, default=96, help="Forecast horizon length.")
    parser.add_argument("--target-column", default="OT", help="Forecast target column.")
    parser.add_argument("--train-rows", type=int, default=DEFAULT_TRAIN_ROWS, help="Number of raw rows in train split.")
    parser.add_argument("--val-rows", type=int, default=DEFAULT_VAL_ROWS, help="Number of raw rows in val split.")
    parser.add_argument("--test-rows", type=int, default=DEFAULT_TEST_ROWS, help="Number of raw rows in test split.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    csv_path = Path(args.csv_path)
    output_dir = Path(args.output_dir)

    if not csv_path.exists():
        raise FileNotFoundError(f"ETTh1 csv not found: {csv_path}")

    df = pd.read_csv(csv_path)
    required_columns = {"date", args.target_column}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {sorted(missing)}")

    splits = build_split_configs(
        total_rows=len(df),
        train_rows=args.train_rows,
        val_rows=args.val_rows,
        test_rows=args.test_rows,
    )

    metadata: dict[str, object] = {
        "source_csv": str(csv_path),
        "target_column": args.target_column,
        "lookback_window": args.lookback_window,
        "forecast_horizon": args.forecast_horizon,
        "total_rows": len(df),
        "splits": [asdict(split) | {"num_rows": split.num_rows} for split in splits],
    }

    for split in splits:
        output_path = output_dir / f"{split.name}.jsonl"
        count = write_jsonl(
            iter_split_samples(
                df,
                split,
                lookback_window=args.lookback_window,
                forecast_horizon=args.forecast_horizon,
                target_column=args.target_column,
            ),
            output_path,
        )
        metadata[f"{split.name}_samples"] = count
        print(
            f"[RL-DATA] split={split.name} rows={split.num_rows} samples={count} -> {output_path}"
        )

    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[RL-DATA] wrote metadata to {metadata_path}")


if __name__ == "__main__":
    main()
