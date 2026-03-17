import tempfile
import unittest
from pathlib import Path

import pandas as pd

from recipe.time_series_forecast.build_etth1_sft_subset import build_subset


class TestETTh1SFTSubsetBuilder(unittest.TestCase):
    def test_build_subset_writes_expected_counts(self):
        source_dir = Path("dataset/ett_sft_etth1_runtime_ot")

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "subset"
            counts = build_subset(
                input_dir=source_dir,
                output_dir=output_dir,
                train_samples=7,
                val_samples=5,
                test_samples=3,
            )

            self.assertEqual(counts["train_samples"], 7)
            self.assertEqual(counts["val_samples"], 5)
            self.assertEqual(counts["test_samples"], 3)
            self.assertTrue((output_dir / "metadata.json").exists())

            train_df = pd.read_parquet(output_dir / "train.parquet")
            val_df = pd.read_parquet(output_dir / "val.parquet")
            test_df = pd.read_parquet(output_dir / "test.parquet")

            self.assertEqual(len(train_df), 7)
            self.assertEqual(len(val_df), 5)
            self.assertEqual(len(test_df), 3)
            self.assertEqual(list(train_df["sample_index"]), sorted(train_df["sample_index"].tolist()))


if __name__ == "__main__":
    unittest.main()
