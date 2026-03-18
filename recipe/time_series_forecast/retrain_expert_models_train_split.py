from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from recipe.time_series_forecast.config_utils import get_default_lengths
from recipe.time_series_forecast.models.itransformer.model import create_itransformer_model
from recipe.time_series_forecast.models.patchtst.model import create_patchtst_model


DEFAULT_CSV_PATH = Path("dataset/ETT-small/ETTh1.csv")
DEFAULT_MODELS_ROOT = Path("recipe/time_series_forecast/models")
DEFAULT_TRAIN_ROWS = 12251
DEFAULT_VAL_ROWS = 1913
DEFAULT_TEST_ROWS = 3256


@dataclass(frozen=True)
class SplitBoundary:
    name: str
    start_row: int
    end_row: int


class SlidingWindowDataset(Dataset):
    def __init__(self, values: np.ndarray, lookback_window: int, forecast_horizon: int):
        if values.ndim != 1:
            raise ValueError(f"values must be 1D array, got shape={values.shape}")
        self.values = values.astype(np.float32)
        self.lookback_window = int(lookback_window)
        self.forecast_horizon = int(forecast_horizon)
        self.window_span = self.lookback_window + self.forecast_horizon
        self.num_samples = len(self.values) - self.window_span + 1
        if self.num_samples <= 0:
            raise ValueError(
                f"Not enough rows to build sliding windows. rows={len(self.values)} "
                f"lookback={self.lookback_window} horizon={self.forecast_horizon}"
            )

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        hist_start = idx
        hist_end = idx + self.lookback_window
        fut_end = hist_end + self.forecast_horizon
        x = self.values[hist_start:hist_end]
        y = self.values[hist_end:fut_end]
        return (
            torch.from_numpy(x).unsqueeze(-1),
            torch.from_numpy(y).unsqueeze(-1),
        )


def parse_args() -> argparse.Namespace:
    default_lookback, default_horizon = get_default_lengths()
    parser = argparse.ArgumentParser(
        description="Retrain PatchTST/iTransformer on ETTh1 train split only and write provenance metadata."
    )
    parser.add_argument("--csv-path", default=str(DEFAULT_CSV_PATH), help="Path to ETTh1.csv")
    parser.add_argument("--models-root", default=str(DEFAULT_MODELS_ROOT), help="Directory containing patchtst/itransformer folders")
    parser.add_argument("--models", default="patchtst,itransformer", help="Comma-separated model list")
    parser.add_argument("--target-column", default="OT", help="Target column in ETTh1.csv")
    parser.add_argument("--lookback-window", type=int, default=default_lookback, help="Lookback window length")
    parser.add_argument("--forecast-horizon", type=int, default=default_horizon, help="Forecast horizon length")
    parser.add_argument("--train-rows", type=int, default=DEFAULT_TRAIN_ROWS, help="Rows reserved for train split")
    parser.add_argument("--val-rows", type=int, default=DEFAULT_VAL_ROWS, help="Rows reserved for val split metadata")
    parser.add_argument("--test-rows", type=int, default=DEFAULT_TEST_ROWS, help="Rows reserved for test split metadata")
    parser.add_argument("--seed", type=int, default=42, help="Global random seed")
    parser.add_argument("--num-workers", type=int, default=0, help="Dataloader workers")
    parser.add_argument("--max-train-samples", type=int, default=0, help="Optional cap on sliding windows (0 means all)")
    parser.add_argument("--device", default="cuda", help="Training device, e.g. cuda or cpu")

    parser.add_argument("--patchtst-epochs", type=int, default=10)
    parser.add_argument("--patchtst-batch-size", type=int, default=128)
    parser.add_argument("--patchtst-learning-rate", type=float, default=1e-4)

    parser.add_argument("--itransformer-epochs", type=int, default=3)
    parser.add_argument("--itransformer-batch-size", type=int, default=64)
    parser.add_argument("--itransformer-learning-rate", type=float, default=1e-4)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(device: str) -> torch.device:
    req = device.strip().lower()
    if req.startswith("cuda") and not torch.cuda.is_available():
        print("[WARN] CUDA is unavailable, falling back to CPU")
        return torch.device("cpu")
    return torch.device(req)


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def sha256_text(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def run_git(cmd: list[str]) -> str:
    try:
        out = subprocess.check_output(cmd, cwd=PROJECT_ROOT, stderr=subprocess.DEVNULL)
        return out.decode("utf-8").strip()
    except Exception:
        return ""


def load_model_config(model_dir: Path, model_name: str, lookback_window: int, forecast_horizon: int) -> dict[str, Any]:
    config_path = model_dir / "config.json"
    config: dict[str, Any] = {}
    if config_path.exists():
        with config_path.open("r", encoding="utf-8") as handle:
            loaded = json.load(handle)
            if isinstance(loaded, dict):
                config.update(loaded)

    config["model_name"] = "PatchTST" if model_name == "patchtst" else "iTransformer"
    config["seq_len"] = int(lookback_window)
    config["pred_len"] = int(forecast_horizon)
    config.setdefault("enc_in", 1)
    config.setdefault("features", "S")
    config.setdefault("target", "OT")
    config["description"] = (
        f"{config['model_name']} retrained on ETTh1 TRAIN split only "
        f"(rows [0,{DEFAULT_TRAIN_ROWS})) with strict provenance."
    )
    return config


def build_boundaries(total_rows: int, train_rows: int, val_rows: int, test_rows: int) -> list[SplitBoundary]:
    if train_rows + val_rows + test_rows != total_rows:
        raise ValueError(
            "Split rows must sum to CSV total rows, got "
            f"train={train_rows}, val={val_rows}, test={test_rows}, total={total_rows}"
        )
    return [
        SplitBoundary("train", 0, train_rows),
        SplitBoundary("val", train_rows, train_rows + val_rows),
        SplitBoundary("test", train_rows + val_rows, total_rows),
    ]


def prepare_dataset(
    csv_path: Path,
    target_column: str,
    lookback_window: int,
    forecast_horizon: int,
    train_rows: int,
    max_train_samples: int,
) -> tuple[SlidingWindowDataset, int]:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in {csv_path}")

    values = df[target_column].to_numpy(dtype=np.float32)
    train_values = values[:train_rows]
    dataset = SlidingWindowDataset(train_values, lookback_window=lookback_window, forecast_horizon=forecast_horizon)

    if max_train_samples > 0 and max_train_samples < len(dataset):
        dataset.values = dataset.values[: max_train_samples + lookback_window + forecast_horizon - 1]
        dataset.num_samples = max_train_samples

    return dataset, len(df)


def train_one_model(
    *,
    model_name: str,
    model_config: dict[str, Any],
    dataset: SlidingWindowDataset,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    num_workers: int,
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    if model_name == "patchtst":
        model = create_patchtst_model(model_config)
    elif model_name == "itransformer":
        model = create_itransformer_model(model_config)
    else:
        raise ValueError(f"Unsupported model_name: {model_name}")

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=False)

    epoch_losses: list[float] = []
    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        steps = 0
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            running += float(loss.detach().item())
            steps += 1

        avg = running / max(steps, 1)
        epoch_losses.append(avg)
        print(f"[{model_name}] epoch={epoch}/{epochs} train_mse={avg:.6f}")

    model.eval()
    total_abs = 0.0
    total_sq = 0.0
    total_count = 0
    with torch.no_grad():
        eval_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False)
        for x, y in eval_loader:
            x = x.to(device)
            y = y.to(device)
            pred = model(x)
            diff = pred - y
            total_abs += float(diff.abs().sum().item())
            total_sq += float((diff ** 2).sum().item())
            total_count += int(diff.numel())

    train_mae = total_abs / max(total_count, 1)
    train_mse = total_sq / max(total_count, 1)
    metrics = {
        "epoch": int(epochs),
        "train_loss": float(epoch_losses[-1] if epoch_losses else 0.0),
        "train_mae": float(train_mae),
        "train_mse": float(train_mse),
        "model": model_name,
        "dataset": "ETTh1",
        "batch_size": int(batch_size),
        "learning_rate": float(learning_rate),
        "split_policy": "train_only",
    }

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "config": model_config,
        "metrics": metrics,
    }

    training_meta = {
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "learning_rate": float(learning_rate),
        "num_workers": int(num_workers),
        "num_windows": int(len(dataset)),
    }
    return checkpoint, metrics, training_meta


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)

    models = [item.strip().lower() for item in args.models.split(",") if item.strip()]
    for model_name in models:
        if model_name not in {"patchtst", "itransformer"}:
            raise ValueError(f"Unsupported model in --models: {model_name}")

    csv_path = Path(args.csv_path)
    models_root = Path(args.models_root)
    dataset, total_rows = prepare_dataset(
        csv_path=csv_path,
        target_column=args.target_column,
        lookback_window=args.lookback_window,
        forecast_horizon=args.forecast_horizon,
        train_rows=args.train_rows,
        max_train_samples=args.max_train_samples,
    )
    boundaries = build_boundaries(total_rows, args.train_rows, args.val_rows, args.test_rows)

    cmdline = " ".join([sys.executable, *sys.argv])
    git_commit = run_git(["git", "rev-parse", "HEAD"])
    git_status = run_git(["git", "status", "--porcelain"])
    now = datetime.now(timezone.utc).isoformat()
    script_path = Path(__file__).resolve()
    script_hash = sha256_text(script_path.read_text(encoding="utf-8"))

    model_source_paths = {
        "patchtst": PROJECT_ROOT / "recipe/time_series_forecast/models/patchtst/model.py",
        "itransformer": PROJECT_ROOT / "recipe/time_series_forecast/models/itransformer/model.py",
    }

    for model_name in models:
        model_dir = models_root / model_name
        model_dir.mkdir(parents=True, exist_ok=True)

        config = load_model_config(
            model_dir=model_dir,
            model_name=model_name,
            lookback_window=args.lookback_window,
            forecast_horizon=args.forecast_horizon,
        )

        if model_name == "patchtst":
            epochs = args.patchtst_epochs
            batch_size = args.patchtst_batch_size
            lr = args.patchtst_learning_rate
        else:
            epochs = args.itransformer_epochs
            batch_size = args.itransformer_batch_size
            lr = args.itransformer_learning_rate

        checkpoint, metrics, training_meta = train_one_model(
            model_name=model_name,
            model_config=config,
            dataset=dataset,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=lr,
            num_workers=args.num_workers,
            device=device,
        )

        checkpoint_path = model_dir / "checkpoint.pth"
        config_path = model_dir / "config.json"
        provenance_path = model_dir / "provenance.json"

        torch.save(checkpoint, checkpoint_path)
        config_path.write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

        checkpoint_sha = sha256_file(checkpoint_path)
        model_source = model_source_paths[model_name]
        model_source_hash = sha256_text(model_source.read_text(encoding="utf-8"))

        provenance = {
            "schema_version": "1.0",
            "timestamp_utc": now,
            "model_name": model_name,
            "data": {
                "source_csv": str(csv_path.resolve()),
                "target_column": args.target_column,
                "total_rows": int(total_rows),
                "split_boundaries": [asdict(item) for item in boundaries],
                "training_rows_used": {
                    "start_row": 0,
                    "end_row": int(args.train_rows),
                    "exclusive_end": True,
                    "policy": "train_split_only",
                },
                "lookback_window": int(args.lookback_window),
                "forecast_horizon": int(args.forecast_horizon),
                "num_windows_used": int(len(dataset)),
            },
            "training": {
                **training_meta,
                "device": str(device),
                "seed": int(args.seed),
                "metrics": metrics,
            },
            "artifacts": {
                "checkpoint_path": str(checkpoint_path.resolve()),
                "checkpoint_sha256": checkpoint_sha,
                "checkpoint_size_bytes": int(checkpoint_path.stat().st_size),
                "config_path": str(config_path.resolve()),
                "config_sha256": sha256_file(config_path),
                "provenance_path": str(provenance_path.resolve()),
            },
            "code_version": {
                "project_root": str(PROJECT_ROOT.resolve()),
                "git_commit": git_commit,
                "git_dirty": bool(git_status.strip()),
                "train_script_path": str(script_path.resolve()),
                "train_script_sha256": script_hash,
                "model_definition_path": str(model_source.resolve()),
                "model_definition_sha256": model_source_hash,
            },
            "command": {
                "cwd": str(Path.cwd().resolve()),
                "argv": sys.argv,
                "command_line": cmdline,
            },
        }

        provenance_path.write_text(json.dumps(provenance, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(
            f"[DONE] {model_name} checkpoint={checkpoint_path} "
            f"sha256={checkpoint_sha[:16]}... provenance={provenance_path}"
        )


if __name__ == "__main__":
    main()
