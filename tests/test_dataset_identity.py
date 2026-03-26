import json
import tempfile
import unittest
from pathlib import Path

from recipe.time_series_forecast.dataset_identity import (
    DATASET_KIND_RL_JSONL,
    HISTORICAL_DATA_PROTOCOL_TIMESTAMPED_NAMED_ROWS,
    require_multivariate_etth1_metadata,
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

    def test_require_multivariate_etth1_metadata_accepts_paper_aligned_payload(self) -> None:
        payload = {
            "dataset_kind": DATASET_KIND_RL_JSONL,
            "pipeline_stage": "curriculum_rl",
            "historical_data_protocol": HISTORICAL_DATA_PROTOCOL_TIMESTAMPED_NAMED_ROWS,
            "task_type": "multivariate time-series forecasting",
            "target_column": "OT",
            "observed_feature_columns": ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"],
            "observed_covariates": ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL"],
            "model_input_width": 7,
        }
        resolved = require_multivariate_etth1_metadata(payload, metadata_path="metadata.json")
        self.assertEqual(resolved["target_column"], "OT")

    def test_require_multivariate_etth1_metadata_rejects_value_only_payload(self) -> None:
        payload = {
            "dataset_kind": DATASET_KIND_RL_JSONL,
            "pipeline_stage": "curriculum_rl",
            "historical_data_protocol": "value_only_rows",
            "task_type": "single-variable time-series forecasting",
            "target_column": "OT",
            "observed_feature_columns": ["OT"],
            "observed_covariates": [],
            "model_input_width": 1,
        }
        with self.assertRaises(ValueError):
            require_multivariate_etth1_metadata(payload, metadata_path="metadata.json")


if __name__ == "__main__":
    unittest.main()
