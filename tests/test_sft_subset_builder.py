import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from recipe.time_series_forecast.build_etth1_sft_subset import build_subset
from recipe.time_series_forecast.dataset_identity import (
    DATASET_KIND_RUNTIME_SFT_PARQUET,
    DATASET_KIND_RUNTIME_SFT_SUBSET,
)


class TestETTh1SFTSubsetBuilder(unittest.TestCase):
    def test_build_subset_writes_expected_counts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            source_dir = Path(tmpdir) / "source"
            source_dir.mkdir(parents=True, exist_ok=True)
            for split_name, count in (("train", 11), ("val", 9), ("test", 5)):
                pd.DataFrame(
                    {
                        "sample_index": list(range(count)),
                        "messages": [[{"role": "assistant", "content": "<answer>\n1.0000\n</answer>"}]] * count,
                    }
                ).to_parquet(source_dir / f"{split_name}.parquet", index=False)
            (source_dir / "metadata.json").write_text(
                json.dumps(
                    {
                        "dataset_kind": DATASET_KIND_RUNTIME_SFT_PARQUET,
                        "pipeline_stage": "runtime_multiturn_sft",
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )

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

            metadata = json.loads((output_dir / "metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["dataset_kind"], DATASET_KIND_RUNTIME_SFT_SUBSET)
            self.assertEqual(metadata["source_pipeline_stage"], "runtime_multiturn_sft")


if __name__ == "__main__":
    unittest.main()
