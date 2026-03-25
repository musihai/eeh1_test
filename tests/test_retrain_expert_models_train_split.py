import unittest

import pandas as pd

from recipe.time_series_forecast.retrain_expert_models_train_split import (
    DEFAULT_FEATURE_COLUMNS,
    MultivariateWindowDataset,
    build_model_config,
)


class TestRetrainExpertModelsTrainSplit(unittest.TestCase):
    def test_build_model_config_switches_to_multivariate_ms(self) -> None:
        config = build_model_config(
            "patchtst",
            feature_columns=list(DEFAULT_FEATURE_COLUMNS),
            target_column="OT",
            lookback_window=96,
            forecast_horizon=96,
        )
        self.assertEqual(config["enc_in"], 7)
        self.assertEqual(config["features"], "MS")
        self.assertEqual(config["target"], "OT")
        self.assertEqual(config["target_channel_index"], 6)

    def test_multivariate_window_dataset_uses_target_channel_for_labels(self) -> None:
        frame = pd.DataFrame(
            {
                "date": pd.date_range("2016-01-01 00:00:00", periods=6, freq="h"),
                "HUFL": [10, 11, 12, 13, 14, 15],
                "HULL": [20, 21, 22, 23, 24, 25],
                "MUFL": [30, 31, 32, 33, 34, 35],
                "MULL": [40, 41, 42, 43, 44, 45],
                "LUFL": [50, 51, 52, 53, 54, 55],
                "LULL": [60, 61, 62, 63, 64, 65],
                "OT": [1, 2, 3, 4, 5, 6],
            }
        )
        dataset = MultivariateWindowDataset(
            frame=frame,
            feature_columns=list(DEFAULT_FEATURE_COLUMNS),
            target_column="OT",
            lookback_window=3,
            forecast_horizon=2,
        )

        features, labels = dataset[0]
        self.assertEqual(tuple(features.shape), (3, 7))
        self.assertEqual(labels.tolist(), [4.0, 5.0])


if __name__ == "__main__":
    unittest.main()
