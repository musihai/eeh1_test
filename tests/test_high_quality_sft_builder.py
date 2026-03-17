import asyncio
import unittest
from unittest import mock

from recipe.time_series_forecast.build_etth1_high_quality_sft import (
    ensure_service_ready,
    evenly_spaced_records,
    evaluate_teacher_for_sample,
    select_curated_evaluations,
)


class TestHighQualitySFTBuilder(unittest.TestCase):
    def test_evenly_spaced_records_returns_deterministic_subset(self):
        records = [{"index": idx} for idx in range(10)]
        selected = evenly_spaced_records(records, 4)
        self.assertEqual([item["index"] for item in selected], [0, 3, 6, 9])

    def test_select_curated_evaluations_prefers_best_score_per_bucket(self):
        evaluations = [
            {"sample_index": 0, "selection_score": 0.4, "best_score": 0.4, "score_margin": 0.1},
            {"sample_index": 1, "selection_score": 0.7, "best_score": 0.7, "score_margin": 0.2},
            {"sample_index": 2, "selection_score": 0.5, "best_score": 0.5, "score_margin": 0.2},
            {"sample_index": 3, "selection_score": 0.9, "best_score": 0.9, "score_margin": 0.3},
        ]
        selected = select_curated_evaluations(evaluations, 2)
        self.assertEqual([item["sample_index"] for item in selected], [1, 3])

    def test_service_health_check_accepts_loaded_models(self):
        class FakeResponse:
            def __init__(self, payload):
                self._payload = payload

            def raise_for_status(self):
                return None

            def json(self):
                return self._payload

        class FakeClient:
            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return None

            async def get(self, url):
                if url.endswith("/health"):
                    return FakeResponse(
                        {
                            "status": "healthy",
                            "models_loaded": {
                                "patchtst": True,
                                "chronos2": True,
                                "itransformer": True,
                            },
                        }
                    )
                if url.endswith("/models"):
                    return FakeResponse({"available_models": ["patchtst", "chronos2", "itransformer"]})
                raise AssertionError(f"Unexpected URL: {url}")

        with mock.patch("httpx.AsyncClient", return_value=FakeClient()):
            info = asyncio.run(
                ensure_service_ready(
                    ["patchtst", "chronos2", "itransformer"],
                    "http://127.0.0.1:8994",
                )
            )

        self.assertIn("patchtst", info["models_loaded"])
        self.assertIn("chronos2", info["available_models"])

    def test_teacher_evaluation_error_includes_model_failures(self):
        sample = {
            "index": 0,
            "data_source": "ETTh1",
            "raw_prompt": [
                {
                    "role": "user",
                    "content": (
                        "[Task] Single-variable time-series forecasting.\n"
                        "Target Column: OT\n"
                        "Lookback Window: 4\n"
                        "Forecast Horizon: 2\n"
                        "Historical Data:\n"
                        "1.0000\n2.0000\n3.0000\n4.0000"
                    ),
                }
            ],
            "reward_model": {
                "ground_truth": "2016-01-01 00:00:00 1.0000\n2016-01-01 01:00:00 2.0000",
            },
        }

        async def _raise_prediction_error(*args, **kwargs):
            raise RuntimeError("service unavailable")

        with mock.patch(
            "recipe.time_series_forecast.build_etth1_high_quality_sft.predict_time_series_async",
            side_effect=_raise_prediction_error,
        ):
            with self.assertRaises(RuntimeError) as ctx:
                asyncio.run(
                    evaluate_teacher_for_sample(
                        sample=sample,
                        models=["patchtst"],
                        predictor_mode="service",
                        predictor=None,
                        model_service_url="http://127.0.0.1:8994",
                    )
                )

        self.assertIn("patchtst", str(ctx.exception))
        self.assertIn("service unavailable", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
