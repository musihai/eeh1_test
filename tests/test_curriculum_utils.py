from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from recipe.time_series_forecast.curriculum_utils import (
    CURRICULUM_PIPELINE_STAGE,
    curriculum_train_file_for_phase,
    parse_curriculum_phase_list,
    resolve_curriculum_train_file,
)


class TestCurriculumUtils(unittest.TestCase):
    def test_parse_curriculum_phase_list_deduplicates_and_normalizes(self) -> None:
        self.assertEqual(
            parse_curriculum_phase_list(" stage1,stage12 , stage123,stage12 "),
            ["stage1", "stage12", "stage123"],
        )

    def test_curriculum_train_file_for_phase_maps_expected_filename(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_dir = Path(tmpdir)
            self.assertEqual(
                curriculum_train_file_for_phase(dataset_dir, "stage12"),
                dataset_dir.resolve() / "train_stage12.jsonl",
            )

    def test_resolve_curriculum_train_file_rejects_full_train_in_train_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_dir = Path(tmpdir)
            train_file = dataset_dir / "train.jsonl"
            train_file.write_text("", encoding="utf-8")
            with self.assertRaises(ValueError):
                resolve_curriculum_train_file(
                    train_file=train_file,
                    metadata_payload={"pipeline_stage": CURRICULUM_PIPELINE_STAGE},
                    run_mode="train",
                )

    def test_resolve_curriculum_train_file_accepts_staged_phase(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_dir = Path(tmpdir)
            (dataset_dir / "train.jsonl").write_text("", encoding="utf-8")
            staged = dataset_dir / "train_stage1.jsonl"
            staged.write_text("", encoding="utf-8")
            resolved = resolve_curriculum_train_file(
                train_file=dataset_dir / "train.jsonl",
                metadata_payload={"pipeline_stage": CURRICULUM_PIPELINE_STAGE},
                run_mode="train",
                curriculum_phase="stage1",
            )
            self.assertEqual(resolved, staged.resolve())

    def test_resolve_curriculum_train_file_passes_through_non_curriculum_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_dir = Path(tmpdir)
            train_file = dataset_dir / "train.jsonl"
            train_file.write_text("", encoding="utf-8")
            resolved = resolve_curriculum_train_file(
                train_file=train_file,
                metadata_payload={"pipeline_stage": "base_rl"},
                run_mode="train",
            )
            self.assertEqual(resolved, train_file.resolve())


if __name__ == "__main__":
    unittest.main()
