from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from recipe.time_series_forecast.build_etth1_sft_dataset import _compute_routing_feature_snapshot
from recipe.time_series_forecast.dataset_file_utils import load_jsonl_records
from recipe.time_series_forecast.task_protocol import parse_task_prompt
from recipe.time_series_forecast.time_series_io import parse_time_series_string


FEATURE_NAMES = [
    "acf1",
    "acf_seasonal",
    "cusum_max",
    "changepoint_count",
    "peak_count",
    "peak_spacing_cv",
    "monotone_duration",
    "residual_exceed_ratio",
    "quality_quantization_score",
    "quality_saturation_ratio",
]
DEFAULT_OUTPUT_DIR = Path("artifacts/reports/v19_risk_gate_audit")
DEFAULT_DATASET_DIR = Path("dataset/ett_sft_etth1_runtime_teacher200_refteacher_v15")
DEFAULT_TAUS = (0.15, 0.20)
SUPPORTED_MODELS = ("patchtst", "itransformer", "arima", "chronos2")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit v19 default expert choice and Risk Gate learnability.")
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--taus", type=float, nargs="+", default=list(DEFAULT_TAUS))
    return parser.parse_args()


def _load_merged_curated_split(dataset_dir: Path, split: str) -> list[dict[str, Any]]:
    source_rows = load_jsonl_records(dataset_dir / f"{split}_curated.jsonl")
    teacher_rows = load_jsonl_records(dataset_dir / f"{split}_teacher_eval_curated.jsonl")
    teacher_by_index = {int(row["sample_index"]): row for row in teacher_rows}
    merged: list[dict[str, Any]] = []
    for source_row in source_rows:
        sample_index = int(source_row["index"])
        teacher_row = teacher_by_index.get(sample_index)
        if teacher_row is None:
            continue
        merged_row = dict(source_row)
        merged_row.update(teacher_row)
        merged.append(merged_row)
    return merged


def _full_split_teacher_rows(dataset_dir: Path, split: str) -> list[dict[str, Any]]:
    return load_jsonl_records(dataset_dir / f"{split}_teacher_eval.jsonl")


def _teacher_error_by_model(row: dict[str, Any]) -> dict[str, float]:
    details = row.get("model_score_details") or {}
    output: dict[str, float] = {}
    if not isinstance(details, dict):
        return output
    for model_name, payload in details.items():
        if not isinstance(payload, dict):
            continue
        try:
            value = float(payload.get("orig_mse"))
        except (TypeError, ValueError):
            continue
        if np.isfinite(value):
            output[str(model_name).strip().lower()] = value
    return output


def _default_expert_summary(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    per_model_errors: dict[str, list[float]] = defaultdict(list)
    per_model_regrets: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        error_by_model = _teacher_error_by_model(row)
        if not error_by_model:
            continue
        best_error = min(error_by_model.values())
        for model_name, error in error_by_model.items():
            per_model_errors[model_name].append(float(error))
            per_model_regrets[model_name].append(float(error - best_error))

    summary: dict[str, dict[str, float]] = {}
    for model_name in sorted(per_model_errors):
        regrets = sorted(per_model_regrets[model_name])
        p90_idx = max(0, int(np.ceil(0.9 * len(regrets))) - 1)
        p95_idx = max(0, int(np.ceil(0.95 * len(regrets))) - 1)
        summary[model_name] = {
            "mean_mse": float(np.mean(per_model_errors[model_name])),
            "mean_regret": float(np.mean(per_model_regrets[model_name])),
            "p90_regret": float(regrets[p90_idx]),
            "p95_regret": float(regrets[p95_idx]),
        }
    return summary


def _choose_default_expert(val_summary: dict[str, dict[str, float]]) -> str:
    ranking = sorted(
        val_summary.items(),
        key=lambda item: (
            float(item[1]["mean_mse"]),
            float(item[1]["mean_regret"]),
            float(item[1]["p90_regret"]),
            str(item[0]),
        ),
    )
    if not ranking:
        raise ValueError("No valid teacher-eval rows were found for default expert selection.")
    return str(ranking[0][0])


def _extract_visible_features(row: dict[str, Any]) -> list[float]:
    raw_prompt = str(row["raw_prompt"][0]["content"])
    task = parse_task_prompt(raw_prompt, data_source=row.get("data_source"))
    historical_data = task.historical_data or raw_prompt
    target_column = task.target_column or "OT"
    _timestamps, history_values = parse_time_series_string(historical_data, target_column=target_column)
    feature_snapshot = _compute_routing_feature_snapshot(history_values)
    return [float(feature_snapshot.get(name, 0.0) or 0.0) for name in FEATURE_NAMES]


def _risk_value(row: dict[str, Any], *, default_expert: str) -> float:
    error_by_model = _teacher_error_by_model(row)
    if default_expert not in error_by_model or not error_by_model:
        return 0.0
    default_error = float(error_by_model[default_expert])
    best_error = float(min(error_by_model.values()))
    if default_error <= 0:
        return 0.0
    return float((default_error - best_error) / default_error)


def _risk_label(row: dict[str, Any], *, default_expert: str, tau: float) -> int:
    return int(_risk_value(row, default_expert=default_expert) >= float(tau))


def _evaluate_binary_classifier(
    model: Any,
    *,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_eval: np.ndarray,
    y_eval: np.ndarray,
) -> dict[str, Any]:
    model.fit(X_train, y_train)
    if hasattr(model, "predict_proba"):
        score = np.asarray(model.predict_proba(X_eval)[:, 1], dtype=float)
    elif hasattr(model, "decision_function"):
        raw_score = np.asarray(model.decision_function(X_eval), dtype=float)
        score = 1.0 / (1.0 + np.exp(-raw_score))
    else:
        score = np.asarray(model.predict(X_eval), dtype=float)
    pred = (score >= 0.5).astype(int)
    auc = float(roc_auc_score(y_eval, score)) if len(set(y_eval.tolist())) > 1 else float("nan")
    return {
        "auc": auc,
        "f1": float(f1_score(y_eval, pred, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_eval, pred)),
        "predicted_distribution": {str(k): int(v) for k, v in sorted(Counter(pred.tolist()).items())},
    }


def _risk_audit_for_tau(
    *,
    train_rows: list[dict[str, Any]],
    val_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    default_expert: str,
    tau: float,
) -> dict[str, Any]:
    X_train = np.asarray([_extract_visible_features(row) for row in train_rows], dtype=float)
    X_val = np.asarray([_extract_visible_features(row) for row in val_rows], dtype=float)
    X_test = np.asarray([_extract_visible_features(row) for row in test_rows], dtype=float)

    y_train = np.asarray([_risk_label(row, default_expert=default_expert, tau=tau) for row in train_rows], dtype=int)
    y_val = np.asarray([_risk_label(row, default_expert=default_expert, tau=tau) for row in val_rows], dtype=int)
    y_test = np.asarray([_risk_label(row, default_expert=default_expert, tau=tau) for row in test_rows], dtype=int)

    classifiers = {
        "logistic_regression": make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight="balanced", random_state=0),
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=400,
            random_state=0,
            class_weight="balanced_subsample",
        ),
        # v19 plan asks for LightGBM/XGBoost; the environment does not ship either,
        # so we use HistGradientBoosting as the closest built-in tree-boosting fallback
        # and record that explicitly in the audit report.
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            max_depth=4,
            learning_rate=0.05,
            max_iter=300,
            random_state=0,
        ),
    }

    model_results: dict[str, Any] = {}
    for name, classifier in classifiers.items():
        model_results[name] = {
            "val": _evaluate_binary_classifier(
                classifier,
                X_train=X_train,
                y_train=y_train,
                X_eval=X_val,
                y_eval=y_val,
            ),
            "test": _evaluate_binary_classifier(
                classifier,
                X_train=X_train,
                y_train=y_train,
                X_eval=X_test,
                y_eval=y_test,
            ),
        }

    passes = any(
        float(metrics["val"]["auc"]) >= 0.70 and float(metrics["val"]["f1"]) >= 0.55
        for metrics in model_results.values()
    )
    return {
        "tau": float(tau),
        "default_expert": default_expert,
        "label_distribution": {
            "train": {str(k): int(v) for k, v in sorted(Counter(y_train.tolist()).items())},
            "val": {str(k): int(v) for k, v in sorted(Counter(y_val.tolist()).items())},
            "test": {str(k): int(v) for k, v in sorted(Counter(y_test.tolist()).items())},
        },
        "pass": bool(passes),
        "models": model_results,
    }


def _write_markdown_report(
    *,
    output_path: Path,
    default_expert: str,
    default_summary_by_split: dict[str, Any],
    risk_audits: list[dict[str, Any]],
) -> None:
    lines: list[str] = []
    lines.append("# v19 Risk Gate Learnability Audit")
    lines.append("")
    lines.append("## Default Expert Audit")
    lines.append("")
    lines.append(f"Selected default expert: `{default_expert}`")
    lines.append("")
    for split_name, split_summary in default_summary_by_split.items():
        lines.append(f"### {split_name}")
        lines.append("")
        lines.append("| model | mean_mse | mean_regret | p90_regret | p95_regret |")
        lines.append("| --- | ---: | ---: | ---: | ---: |")
        for model_name in SUPPORTED_MODELS:
            metrics = split_summary.get(model_name)
            if not metrics:
                continue
            lines.append(
                f"| {model_name} | {metrics['mean_mse']:.6f} | {metrics['mean_regret']:.6f} | "
                f"{metrics['p90_regret']:.6f} | {metrics['p95_regret']:.6f} |"
            )
        lines.append("")

    lines.append("## Risk Gate Audit")
    lines.append("")
    lines.append(
        "Visible inputs use the current Turn-2 feature view only: the routing feature snapshot "
        "derived from historical data. Hidden oracle quantities such as `best_model`, "
        "`default_error`, and `route_margin_rel` are not given to the classifier."
    )
    lines.append("")
    lines.append(
        "The environment does not include `xgboost` or `lightgbm`, so `HistGradientBoostingClassifier` "
        "is used as the tree-boosting fallback for the third audit model."
    )
    lines.append("")

    for audit in risk_audits:
        tau = float(audit["tau"])
        lines.append(f"### tau={tau:.2f}")
        lines.append("")
        lines.append(
            f"Label distribution: train={audit['label_distribution']['train']}, "
            f"val={audit['label_distribution']['val']}, test={audit['label_distribution']['test']}"
        )
        lines.append("")
        lines.append("| model | split | auc | f1 | balanced_acc | predicted_distribution |")
        lines.append("| --- | --- | ---: | ---: | ---: | --- |")
        for model_name, result in audit["models"].items():
            for split_name in ("val", "test"):
                metrics = result[split_name]
                lines.append(
                    f"| {model_name} | {split_name} | {metrics['auc']:.4f} | {metrics['f1']:.4f} | "
                    f"{metrics['balanced_accuracy']:.4f} | {metrics['predicted_distribution']} |"
                )
        lines.append("")
        lines.append(f"Gate pass at tau={tau:.2f}: `{audit['pass']}`")
        lines.append("")

    overall_pass = any(bool(audit["pass"]) for audit in risk_audits)
    lines.append("## Gate 0 Decision")
    lines.append("")
    if overall_pass:
        lines.append("Gate 0 passed. Learned Risk Gate remains viable for v19.")
    else:
        lines.append("Gate 0 failed. Learned Risk Gate is not learnable enough from the current Turn-2 visible state.")
        lines.append("")
        lines.append("Per v19, Turn 2 should switch to fixed expand (`default_risky`) and the main decision should move to Turn 3 final selection.")
    lines.append("")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    default_summary_by_split = {
        split: _default_expert_summary(_full_split_teacher_rows(args.dataset_dir, split))
        for split in ("train", "val", "test")
    }
    default_expert = _choose_default_expert(default_summary_by_split["val"])

    train_rows = _load_merged_curated_split(args.dataset_dir, "train")
    val_rows = _load_merged_curated_split(args.dataset_dir, "val")
    test_rows = _load_merged_curated_split(args.dataset_dir, "test")
    risk_audits = [
        _risk_audit_for_tau(
            train_rows=train_rows,
            val_rows=val_rows,
            test_rows=test_rows,
            default_expert=default_expert,
            tau=float(tau),
        )
        for tau in args.taus
    ]

    summary = {
        "dataset_dir": str(args.dataset_dir.resolve()),
        "default_expert": default_expert,
        "default_summary_by_split": default_summary_by_split,
        "risk_audits": risk_audits,
        "gate0_pass": any(bool(audit["pass"]) for audit in risk_audits),
        "turn2_policy_recommendation": (
            "learned_risk_gate" if any(bool(audit["pass"]) for audit in risk_audits) else "fixed_expand"
        ),
    }
    (args.output_dir / "risk_gate_audit_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    _write_markdown_report(
        output_path=args.output_dir / "v19_risk_gate_audit.md",
        default_expert=default_expert,
        default_summary_by_split=default_summary_by_split,
        risk_audits=risk_audits,
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
