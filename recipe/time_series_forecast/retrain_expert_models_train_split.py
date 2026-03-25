from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from recipe.time_series_forecast.build_etth1_rl_dataset import (
    DEFAULT_TEST_ROWS,
    DEFAULT_TRAIN_ROWS,
    DEFAULT_VAL_ROWS,
)
from recipe.time_series_forecast.models.itransformer.model import create_itransformer_model
from recipe.time_series_forecast.models.patchtst.model import create_patchtst_model


DEFAULT_CSV_PATH = Path("dataset/ETT-small/ETTh1.csv")
DEFAULT_MODELS = ("patchtst", "itransformer")
DEFAULT_OUTPUT_ROOT = Path("recipe/time_series_forecast/models")
DEFAULT_FEATURE_COLUMNS = ("HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT")
DEFAULT_CONFIG_UPDATES: dict[str, dict[str, Any]] = {
    "patchtst": {
        "model_name": "PatchTST",
        "d_model": 16,
        "n_heads": 4,
        "e_layers": 3,
        "d_ff": 128,
        "dropout": 0.3,
        "patch_len": 16,
        "stride": 8,
        "batch_size": 128,
        "learning_rate": 1e-4,
        "epochs": 10,
    },
    "itransformer": {
        "model_name": "iTransformer",
        "d_model": 256,
        "n_heads": 8,
        "e_layers": 2,
        "d_ff": 256,
        "dropout": 0.1,
        "batch_size": 64,
        "learning_rate": 1e-4,
        "epochs": 3,
    },
}


@dataclass(frozen=True)
class TrainConfig:
    model_name: str
    output_name: str
    seq_len: int
    pred_len: int
    enc_in: int
    target_column: str
    target_channel_index: int
    feature_columns: tuple[str, ...]
    batch_size: int
    learning_rate: float
    epochs: int
    device: str
    num_workers: int
    max_windows: int | None
    output_dir: Path
    model_config: dict[str, Any]


class MultivariateWindowDataset(Dataset):
    def __init__(
        self,
        *,
        frame: pd.DataFrame,
        feature_columns: list[str],
        target_column: str,
        lookback_window: int,
        forecast_horizon: int,
        max_windows: int | None = None,
    ) -> None:
        if target_column not in feature_columns:
            raise ValueError(f"target column {target_column!r} is not included in feature columns {feature_columns!r}")

        self.values = frame.loc[:, feature_columns].to_numpy(dtype=np.float32)
        self.target_idx = feature_columns.index(target_column)
        self.lookback_window = int(lookback_window)
        self.forecast_horizon = int(forecast_horizon)
        self.num_windows = len(frame) - self.lookback_window - self.forecast_horizon + 1
        if self.num_windows <= 0:
            raise ValueError(
                f"Frame is too short for lookback={self.lookback_window} and horizon={self.forecast_horizon}. "
                f"rows={len(frame)}"
            )
        if max_windows is not None:
            self.num_windows = min(self.num_windows, int(max_windows))

    def __len__(self) -> int:
        return self.num_windows

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        hist_end = index + self.lookback_window
        forecast_end = hist_end + self.forecast_horizon
        x = self.values[index:hist_end]
        y = self.values[hist_end:forecast_end, self.target_idx]
        return torch.from_numpy(x), torch.from_numpy(y)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Retrain PatchTST/iTransformer experts on ETTh1 TRAIN split with multivariate input and OT target loss."
    )
    parser.add_argument("--csv-path", default=str(DEFAULT_CSV_PATH))
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--target-column", default="OT")
    parser.add_argument("--feature-columns", default=",".join(DEFAULT_FEATURE_COLUMNS))
    parser.add_argument("--lookback-window", type=int, default=96)
    parser.add_argument("--forecast-horizon", type=int, default=96)
    parser.add_argument("--train-rows", type=int, default=DEFAULT_TRAIN_ROWS)
    parser.add_argument("--val-rows", type=int, default=DEFAULT_VAL_ROWS)
    parser.add_argument("--test-rows", type=int, default=DEFAULT_TEST_ROWS)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-windows", type=int, default=0, help="Optional cap for smoke runs; 0 disables the cap.")
    parser.add_argument("--override-epochs", type=int, default=0)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device: str) -> str:
    if device.startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    return device


def load_source_frame(csv_path: Path, *, feature_columns: list[str], total_rows: int) -> pd.DataFrame:
    frame = pd.read_csv(csv_path)
    required_columns = ["date", *feature_columns]
    missing = [column for column in required_columns if column not in frame.columns]
    if missing:
        raise ValueError(f"ETTh1 source csv is missing required columns: {missing}")
    frame = frame.loc[:, required_columns].copy()
    if total_rows > len(frame):
        raise ValueError(f"Requested total_rows={total_rows}, but csv only has {len(frame)} rows")
    return frame.iloc[:total_rows].reset_index(drop=True)


def build_model_config(
    model_name: str,
    *,
    feature_columns: list[str],
    target_column: str,
    lookback_window: int,
    forecast_horizon: int,
) -> dict[str, Any]:
    if model_name not in DEFAULT_CONFIG_UPDATES:
        raise ValueError(f"Unsupported model {model_name!r}")
    config = dict(DEFAULT_CONFIG_UPDATES[model_name])
    config.update(
        {
            "seq_len": int(lookback_window),
            "pred_len": int(forecast_horizon),
            "enc_in": int(len(feature_columns)),
            "features": "MS",
            "target": str(target_column),
            "feature_columns": list(feature_columns),
            "target_channel_index": int(feature_columns.index(target_column)),
            "description": (
                f"{config['model_name']} retrained on ETTh1 TRAIN split only with multivariate input "
                f"({len(feature_columns)} variables) and OT-target supervision."
            ),
        }
    )
    return config


def create_model(model_name: str, config: dict[str, Any]) -> torch.nn.Module:
    if model_name == "patchtst":
        return create_patchtst_model(config)
    if model_name == "itransformer":
        return create_itransformer_model(config)
    raise ValueError(f"Unsupported model {model_name!r}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def git_metadata(project_root: Path) -> dict[str, Any]:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            text=True,
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"],
                cwd=project_root,
                text=True,
            ).strip()
        )
    except Exception:
        commit = ""
        dirty = True
    return {"git_commit": commit, "git_dirty": dirty}


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def train_single_model(
    *,
    config: TrainConfig,
    dataset: MultivariateWindowDataset,
    csv_path: Path,
    train_rows: int,
    val_rows: int,
    test_rows: int,
    seed: int,
) -> dict[str, Any]:
    device = torch.device(config.device)
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        drop_last=False,
    )

    model = create_model(config.output_name, config.model_config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    model.train()
    last_epoch_metrics = {
        "epoch": 0,
        "train_loss": float("nan"),
        "train_mae": float("nan"),
        "train_mse": float("nan"),
    }

    for epoch in range(1, config.epochs + 1):
        epoch_loss = 0.0
        epoch_abs_error = 0.0
        epoch_sq_error = 0.0
        epoch_examples = 0

        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad(set_to_none=True)
            predictions = model(batch_x)[:, :, config.target_channel_index]
            loss = F.mse_loss(predictions, batch_y)
            loss.backward()
            optimizer.step()

            epoch_loss += float(loss.item()) * batch_x.size(0)
            abs_error = torch.abs(predictions.detach() - batch_y)
            sq_error = torch.square(predictions.detach() - batch_y)
            epoch_abs_error += float(abs_error.sum().item())
            epoch_sq_error += float(sq_error.sum().item())
            epoch_examples += int(batch_y.numel())

        num_batches = max(len(loader), 1)
        num_series = max(len(dataset), 1)
        last_epoch_metrics = {
            "epoch": epoch,
            "train_loss": float(epoch_loss / num_series),
            "train_mae": float(epoch_abs_error / epoch_examples),
            "train_mse": float(epoch_sq_error / epoch_examples),
            "model": config.output_name,
            "dataset": "ETTh1",
            "batch_size": config.batch_size,
            "learning_rate": config.learning_rate,
            "split_policy": "train_only_multivariate_ms",
            "train_batches": num_batches,
        }

    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "checkpoint.pth"
    config_path = output_dir / "config.json"
    provenance_path = output_dir / "provenance.json"

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "metrics": last_epoch_metrics,
            "target_channel_index": config.target_channel_index,
            "feature_columns": list(config.feature_columns),
        },
        checkpoint_path,
    )
    save_json(config_path, config.model_config)

    git_info = git_metadata(PROJECT_ROOT)
    provenance = {
        "schema_version": "1.0",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model_name": config.output_name,
        "data": {
            "source_csv": str(csv_path.resolve()),
            "target_column": config.target_column,
            "feature_columns": list(config.feature_columns),
            "features": "MS",
            "total_rows": int(train_rows + val_rows + test_rows),
            "split_boundaries": [
                {"name": "train", "start_row": 0, "end_row": int(train_rows)},
                {"name": "val", "start_row": int(train_rows), "end_row": int(train_rows + val_rows)},
                {
                    "name": "test",
                    "start_row": int(train_rows + val_rows),
                    "end_row": int(train_rows + val_rows + test_rows),
                },
            ],
            "training_rows_used": {
                "start_row": 0,
                "end_row": int(train_rows),
                "exclusive_end": True,
                "policy": "train_split_only",
            },
            "lookback_window": config.seq_len,
            "forecast_horizon": config.pred_len,
            "num_windows_used": int(len(dataset)),
        },
        "training": {
            "epochs": config.epochs,
            "batch_size": config.batch_size,
            "learning_rate": config.learning_rate,
            "num_workers": config.num_workers,
            "num_windows": int(len(dataset)),
            "device": config.device,
            "seed": seed,
            "metrics": last_epoch_metrics,
        },
        "artifacts": {
            "checkpoint_path": str(checkpoint_path.resolve()),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "checkpoint_size_bytes": checkpoint_path.stat().st_size,
            "config_path": str(config_path.resolve()),
            "config_sha256": sha256_file(config_path),
            "provenance_path": str(provenance_path.resolve()),
        },
        "code_version": {
            "project_root": str(PROJECT_ROOT),
            **git_info,
            "train_script_path": str(Path(__file__).resolve()),
            "model_definition_path": str(
                (PROJECT_ROOT / f"recipe/time_series_forecast/models/{config.output_name}/model.py").resolve()
            ),
        },
        "command": {
            "cwd": str(PROJECT_ROOT),
            "argv": sys.argv,
            "command_line": " ".join(sys.argv),
        },
    }
    save_json(provenance_path, provenance)
    return provenance


def main() -> None:
    args = parse_args()
    set_seed(int(args.seed))
    device = resolve_device(str(args.device))
    feature_columns = [column.strip() for column in str(args.feature_columns).split(",") if column.strip()]
    models = [name.strip().lower() for name in str(args.models).split(",") if name.strip()]
    max_windows = int(args.max_windows) if int(args.max_windows) > 0 else None
    total_rows = int(args.train_rows + args.val_rows + args.test_rows)

    source_frame = load_source_frame(Path(args.csv_path), feature_columns=feature_columns, total_rows=total_rows)
    train_frame = source_frame.iloc[: int(args.train_rows)].reset_index(drop=True)
    output_root = Path(args.output_root)

    for model_name in models:
        model_config = build_model_config(
            model_name,
            feature_columns=feature_columns,
            target_column=str(args.target_column),
            lookback_window=int(args.lookback_window),
            forecast_horizon=int(args.forecast_horizon),
        )
        if int(args.override_epochs) > 0:
            model_config["epochs"] = int(args.override_epochs)

        train_config = TrainConfig(
            model_name=model_config["model_name"],
            output_name=model_name,
            seq_len=int(model_config["seq_len"]),
            pred_len=int(model_config["pred_len"]),
            enc_in=int(model_config["enc_in"]),
            target_column=str(args.target_column),
            target_channel_index=int(model_config["target_channel_index"]),
            feature_columns=tuple(feature_columns),
            batch_size=int(model_config["batch_size"]),
            learning_rate=float(model_config["learning_rate"]),
            epochs=int(model_config["epochs"]),
            device=device,
            num_workers=int(args.num_workers),
            max_windows=max_windows,
            output_dir=output_root / model_name,
            model_config=model_config,
        )

        dataset = MultivariateWindowDataset(
            frame=train_frame,
            feature_columns=feature_columns,
            target_column=str(args.target_column),
            lookback_window=train_config.seq_len,
            forecast_horizon=train_config.pred_len,
            max_windows=train_config.max_windows,
        )
        provenance = train_single_model(
            config=train_config,
            dataset=dataset,
            csv_path=Path(args.csv_path),
            train_rows=int(args.train_rows),
            val_rows=int(args.val_rows),
            test_rows=int(args.test_rows),
            seed=int(args.seed),
        )
        print(
            json.dumps(
                {
                    "model": model_name,
                    "checkpoint_path": provenance["artifacts"]["checkpoint_path"],
                    "epochs": train_config.epochs,
                    "num_windows": len(dataset),
                    "enc_in": train_config.enc_in,
                    "target_column": train_config.target_column,
                },
                ensure_ascii=True,
            )
        )


if __name__ == "__main__":
    main()
