from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence


CURRICULUM_PIPELINE_STAGE = "curriculum_rl"
CURRICULUM_PHASE_FILE_MAP = {
    "stage1": "train_stage1.jsonl",
    "stage12": "train_stage12.jsonl",
    "stage123": "train_stage123.jsonl",
    "full": "train.jsonl",
}


def normalize_curriculum_phase(phase: str | None) -> str:
    normalized = str(phase or "").strip().lower()
    if not normalized:
        return ""
    if normalized not in CURRICULUM_PHASE_FILE_MAP:
        raise ValueError(
            f"Unsupported curriculum phase `{phase}`. "
            f"Expected one of {sorted(CURRICULUM_PHASE_FILE_MAP)}."
        )
    return normalized


def parse_curriculum_phase_list(phases: str | Sequence[str] | None) -> list[str]:
    if phases is None:
        return []
    if isinstance(phases, str):
        raw_items = [item.strip() for item in phases.split(",")]
    else:
        raw_items = [str(item).strip() for item in phases]

    parsed: list[str] = []
    for item in raw_items:
        if not item:
            continue
        normalized = normalize_curriculum_phase(item)
        if normalized not in parsed:
            parsed.append(normalized)
    return parsed


def curriculum_train_file_for_phase(dataset_dir: str | Path, phase: str) -> Path:
    normalized = normalize_curriculum_phase(phase)
    if not normalized:
        raise ValueError("Curriculum phase is required to resolve a staged train file.")
    dataset_path = Path(dataset_dir).resolve()
    return dataset_path / CURRICULUM_PHASE_FILE_MAP[normalized]


def resolve_curriculum_train_file(
    *,
    train_file: str | Path,
    metadata_payload: dict,
    run_mode: str,
    curriculum_phase: str | None = None,
    allow_noncurriculum_train: bool = False,
) -> Path:
    train_path = Path(train_file).resolve()
    pipeline_stage = str(metadata_payload.get("pipeline_stage") or "").strip()
    if pipeline_stage != CURRICULUM_PIPELINE_STAGE:
        return train_path

    normalized_phase = normalize_curriculum_phase(curriculum_phase)
    if normalized_phase:
        resolved = curriculum_train_file_for_phase(train_path.parent, normalized_phase)
        if not resolved.exists():
            raise FileNotFoundError(
                f"Curriculum phase file not found for phase `{normalized_phase}`: {resolved}"
            )
        return resolved

    if run_mode == "train" and train_path.name == CURRICULUM_PHASE_FILE_MAP["full"] and not allow_noncurriculum_train:
        raise ValueError(
            "Refusing to launch train mode from curriculum dataset root `train.jsonl`. "
            "Use a staged file (`train_stage1.jsonl`, `train_stage12.jsonl`, `train_stage123.jsonl`) "
            "or set RL_CURRICULUM_PHASE."
        )

    return train_path
