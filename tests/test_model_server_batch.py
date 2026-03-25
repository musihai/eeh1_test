import unittest

import torch
from fastapi import HTTPException

from recipe.time_series_forecast import model_server


class _FakeForecastModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return x


class TestModelServerBatch(unittest.TestCase):
    def test_predict_with_pytorch_model_batch_returns_one_response_per_request(self):
        original_model = model_server._models.get("patchtst")
        model_server._models["patchtst"] = _FakeForecastModel()
        try:
            requests = [
                model_server.PredictRequest(
                    timestamps=[
                        "2016-01-01 00:00:00",
                        "2016-01-01 01:00:00",
                        "2016-01-01 02:00:00",
                    ],
                    values=[1.0, 2.0, 3.0],
                    prediction_length=2,
                    model_name="patchtst",
                ),
                model_server.PredictRequest(
                    timestamps=[
                        "2016-01-02 00:00:00",
                        "2016-01-02 01:00:00",
                        "2016-01-02 02:00:00",
                    ],
                    values=[4.0, 5.0, 6.0],
                    prediction_length=2,
                    model_name="patchtst",
                ),
            ]
            responses = model_server.predict_with_pytorch_model_batch(requests, "patchtst")
        finally:
            model_server._models["patchtst"] = original_model

        self.assertEqual(len(responses), 2)
        self.assertEqual(responses[0].values, [1.0, 2.0])
        self.assertEqual(responses[1].values, [4.0, 5.0])
        self.assertEqual(len(responses[0].timestamps), 2)
        self.assertEqual(responses[0].model_used, "patchtst")

    def test_predict_with_pytorch_model_batch_selects_target_channel_from_multivariate_input(self):
        original_model = model_server._models.get("patchtst")
        original_config = model_server._configs.get("patchtst")
        model_server._models["patchtst"] = _FakeForecastModel()
        model_server._configs["patchtst"] = {"enc_in": 2}
        try:
            requests = [
                model_server.PredictRequest(
                    timestamps=[
                        "2016-01-01 00:00:00",
                        "2016-01-01 01:00:00",
                        "2016-01-01 02:00:00",
                    ],
                    values=[[10.0, 1.0], [20.0, 2.0], [30.0, 3.0]],
                    feature_columns=["HUFL", "OT"],
                    target_column="OT",
                    prediction_length=2,
                    model_name="patchtst",
                ),
            ]
            responses = model_server.predict_with_pytorch_model_batch(requests, "patchtst")
        finally:
            model_server._models["patchtst"] = original_model
            if original_config is None:
                model_server._configs.pop("patchtst", None)
            else:
                model_server._configs["patchtst"] = original_config

        self.assertEqual(len(responses), 1)
        self.assertEqual(responses[0].values, [1.0, 2.0])

    def test_predict_with_pytorch_model_batch_rejects_multivariate_input_for_univariate_checkpoint(self):
        original_model = model_server._models.get("patchtst")
        original_config = model_server._configs.get("patchtst")
        model_server._models["patchtst"] = _FakeForecastModel()
        model_server._configs["patchtst"] = {"enc_in": 1}
        try:
            request = model_server.PredictRequest(
                timestamps=[
                    "2016-01-01 00:00:00",
                    "2016-01-01 01:00:00",
                    "2016-01-01 02:00:00",
                ],
                values=[[10.0, 1.0], [20.0, 2.0], [30.0, 3.0]],
                feature_columns=["HUFL", "OT"],
                target_column="OT",
                prediction_length=2,
                model_name="patchtst",
            )
            with self.assertRaises(HTTPException) as ctx:
                model_server.predict_with_pytorch_model_batch([request], "patchtst")
        finally:
            model_server._models["patchtst"] = original_model
            if original_config is None:
                model_server._configs.pop("patchtst", None)
            else:
                model_server._configs["patchtst"] = original_config

        self.assertEqual(ctx.exception.status_code, 503)


if __name__ == "__main__":
    unittest.main()
