import json
import tempfile
import unittest
from pathlib import Path

from recipe.time_series_forecast.dataset_identity import (
    DATASET_KIND_RL_JSONL,
    validate_metadata_file,
    validate_sibling_metadata,
)


class TestDatasetIdentity(unittest.TestCase):
    def test_validate_metadata_file_accepts_matching_kind(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            metadata_path = Path(tmpdir) / "metadata.json"
            metadata_path.write_text(
                json.dumps({"dataset_kind": DATASET_KIND_RL_JSONL, "pipeline_stage": "curriculum_rl"}) + "\n",
                encoding="utf-8",
            )
            payload, resolved_path = validate_metadata_file(metadata_path, expected_kind=DATASET_KIND_RL_JSONL)
            self.assertEqual(payload["pipeline_stage"], "curriculum_rl")
            self.assertEqual(resolved_path, metadata_path.resolve())

    def test_validate_sibling_metadata_rejects_wrong_kind(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = Path(tmpdir) / "train.jsonl"
            data_path.write_text("{}\n", encoding="utf-8")
            (Path(tmpdir) / "metadata.json").write_text(
                json.dumps({"dataset_kind": "wrong_kind", "pipeline_stage": "oops"}) + "\n",
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                validate_sibling_metadata(data_path, expected_kind=DATASET_KIND_RL_JSONL)


if __name__ == "__main__":
    unittest.main()
