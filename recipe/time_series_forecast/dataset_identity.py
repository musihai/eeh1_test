from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


DATASET_KIND_RL_JSONL = "etth1_rl_jsonl"
DATASET_KIND_TEACHER_CURATED_SFT = "etth1_teacher_curated_sft"
DATASET_KIND_RUNTIME_SFT_PARQUET = "etth1_runtime_sft_parquet"
DATASET_KIND_RUNTIME_SFT_SUBSET = "etth1_runtime_sft_subset"


def load_metadata(metadata_path: str | Path) -> dict[str, Any]:
    path = Path(metadata_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset metadata not found: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Dataset metadata is not valid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Dataset metadata must be a JSON object: {path}")
    return payload


def metadata_path_for_data_file(data_file: str | Path) -> Path:
    return Path(data_file).resolve().parent / "metadata.json"


def validate_metadata_payload(
    payload: dict[str, Any],
    *,
    metadata_path: str | Path,
    expected_kind: str | Iterable[str],
    allowed_pipeline_stages: Iterable[str] | None = None,
) -> dict[str, Any]:
    if isinstance(expected_kind, str):
        expected_kinds = {expected_kind}
    else:
        expected_kinds = {str(item).strip() for item in expected_kind if str(item).strip()}
    actual_kind = str(payload.get("dataset_kind") or "").strip()
    if actual_kind not in expected_kinds:
        kind_text = next(iter(expected_kinds)) if len(expected_kinds) == 1 else sorted(expected_kinds)
        raise ValueError(
            f"Dataset metadata kind mismatch: expected `{kind_text}`, got `{actual_kind or '__missing__'}` "
            f"from {metadata_path}"
        )

    if allowed_pipeline_stages is not None:
        allowed = {str(item).strip() for item in allowed_pipeline_stages if str(item).strip()}
        actual_stage = str(payload.get("pipeline_stage") or "").strip()
        if actual_stage not in allowed:
            raise ValueError(
                f"Dataset metadata stage mismatch: expected one of {sorted(allowed)}, "
                f"got `{actual_stage or '__missing__'}` from {metadata_path}"
            )
    return payload


def validate_metadata_file(
    metadata_path: str | Path,
    *,
    expected_kind: str | Iterable[str],
    allowed_pipeline_stages: Iterable[str] | None = None,
) -> tuple[dict[str, Any], Path]:
    path = Path(metadata_path).resolve()
    payload = load_metadata(path)
    validate_metadata_payload(
        payload,
        metadata_path=path,
        expected_kind=expected_kind,
        allowed_pipeline_stages=allowed_pipeline_stages,
    )
    return payload, path


def validate_sibling_metadata(
    data_file: str | Path,
    *,
    expected_kind: str | Iterable[str],
    allowed_pipeline_stages: Iterable[str] | None = None,
) -> tuple[dict[str, Any], Path]:
    metadata_path = metadata_path_for_data_file(data_file)
    return validate_metadata_file(
        metadata_path,
        expected_kind=expected_kind,
        allowed_pipeline_stages=allowed_pipeline_stages,
    )
