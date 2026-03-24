from __future__ import annotations

from pathlib import Path


LOADABLE_TRANSFORMERS_WEIGHT_FILES = (
    "model.safetensors",
    "model.safetensors.index.json",
    "pytorch_model.bin",
    "pytorch_model.bin.index.json",
    "tf_model.h5",
    "model.ckpt.index",
    "flax_model.msgpack",
)


def has_loadable_transformers_weights(model_dir: str | Path) -> bool:
    path = Path(model_dir).expanduser()
    if not path.is_dir():
        return False
    return any((path / filename).is_file() for filename in LOADABLE_TRANSFORMERS_WEIGHT_FILES)


def resolve_transformers_model_dir(model_dir: str | Path) -> Path:
    path = Path(model_dir).expanduser()
    candidates: list[Path] = []
    seen: set[Path] = set()

    def add_candidate(candidate: Path) -> None:
        resolved = candidate.expanduser().resolve(strict=False)
        if resolved in seen:
            return
        seen.add(resolved)
        candidates.append(resolved)

    add_candidate(path)
    add_candidate(path / "hf_merged")
    add_candidate(path.parent / "hf_merged")

    for candidate in candidates:
        if has_loadable_transformers_weights(candidate):
            return candidate

    checked = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(
        f"No loadable Transformers weights found for {path}. Checked: {checked}"
    )
