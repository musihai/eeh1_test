import asyncio
import warnings
import unittest

import pandas as pd

from recipe.time_series_forecast.utils import predict_with_arima_async


class TestArimaRuntime(unittest.TestCase):
    def test_arima_fallback_handles_nearly_constant_series_without_leaking_known_warnings(self) -> None:
        timestamps = pd.date_range("2016-01-01 00:00:00", periods=96, freq="H")
        df = pd.DataFrame(
            {
                "id": ["ETTh1"] * 96,
                "timestamp": timestamps,
                "target": [25.0] * 96,
            }
        )

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            pred_df = asyncio.run(predict_with_arima_async(df, prediction_length=8))

        self.assertEqual(len(pred_df), 8)
        joined_messages = "\n".join(str(item.message) for item in caught)
        self.assertNotIn("Non-stationary starting autoregressive parameters found", joined_messages)
        self.assertNotIn("Non-invertible starting MA parameters found", joined_messages)
        self.assertNotIn("Maximum Likelihood optimization failed to converge", joined_messages)


if __name__ == "__main__":
    unittest.main()
