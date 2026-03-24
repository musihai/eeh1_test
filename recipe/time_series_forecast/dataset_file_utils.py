from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


def load_jsonl_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def write_jsonl_records(path: Path, records: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def write_metadata_file(output_dir: Path, payload: dict[str, Any]) -> Path:
    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return metadata_path


__all__ = [
    "load_jsonl_records",
    "write_jsonl_records",
    "write_metadata_file",
]
