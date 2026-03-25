import asyncio
import unittest
from unittest import mock

import pandas as pd

from recipe.time_series_forecast import utils as ts_utils
from recipe.time_series_forecast.build_etth1_sft_dataset import _predict_with_runtime_tools


class TestTimeSeriesUtils(unittest.TestCase):
    def tearDown(self) -> None:
        async def _close_client():
            client = ts_utils._httpx_client
            if client is not None and not bool(getattr(client, "is_closed", False)):
                await client.aclose()
            ts_utils._httpx_client = None
            ts_utils._httpx_client_loop = None

        asyncio.run(_close_client())

    def test_get_httpx_client_recreates_client_across_asyncio_runs(self):
        async def _acquire_client_id() -> int:
            client = await ts_utils._get_httpx_client()
            return id(client)

        first_client_id = asyncio.run(_acquire_client_id())
        second_client_id = asyncio.run(_acquire_client_id())

        self.assertNotEqual(first_client_id, second_client_id)

    def test_predict_with_runtime_tools_formats_predictions_without_input_timestamps(self):
        prediction_frame = pd.DataFrame(
            {
                "timestamp": pd.date_range("2000-01-01 00:00:00", periods=3, freq="h"),
                "target_0.5": [1.0, 2.0, 3.0],
            }
        )

        async def fake_predict_time_series_async(context_df, prediction_length, model_name):
            self.assertEqual(prediction_length, 3)
            self.assertEqual(model_name, "chronos2")
            self.assertEqual(list(context_df["target"]), [1.0, 2.0, 3.0])
            return prediction_frame

        with mock.patch(
            "recipe.time_series_forecast.build_etth1_sft_dataset.predict_time_series_async",
            new=fake_predict_time_series_async,
        ):
            prediction_text = asyncio.run(
                _predict_with_runtime_tools(
                    historical_data="1.0000\n2.0000\n3.0000",
                    data_source="ETTh1",
                    target_column="OT",
                    forecast_horizon=3,
                    model_name="chronos2",
                )
            )

        self.assertEqual(
            prediction_text.splitlines(),
            [
                "2000-01-01 00:00:00 1.0000",
                "2000-01-01 01:00:00 2.0000",
                "2000-01-01 02:00:00 3.0000",
            ],
        )

    def test_build_prediction_request_preserves_multivariate_matrix_and_target_column(self):
        context_df = pd.DataFrame(
            {
                "id": ["ETTh1"] * 3,
                "timestamp": pd.date_range("2016-01-01 00:00:00", periods=3, freq="h"),
                "target": [1.0, 2.0, 3.0],
                "HUFL": [10.0, 11.0, 12.0],
                "OT": [1.0, 2.0, 3.0],
            }
        )
        context_df.attrs["feature_columns"] = ["HUFL", "OT"]
        context_df.attrs["target_column"] = "OT"

        request = ts_utils._build_prediction_request(context_df, prediction_length=2, model_name="patchtst")

        self.assertEqual(request["feature_columns"], ["HUFL", "OT"])
        self.assertEqual(request["target_column"], "OT")
        self.assertEqual(request["values"], [[10.0, 1.0], [11.0, 2.0], [12.0, 3.0]])


if __name__ == "__main__":
    unittest.main()
