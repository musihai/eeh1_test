import asyncio
import json
import tempfile
import unittest
from unittest import mock
from pathlib import Path

from recipe.time_series_forecast.build_etth1_high_quality_sft import (
    build_local_model_device_map,
    ensure_service_ready,
    evenly_spaced_records,
    evaluate_teacher_for_sample,
    process_split,
    select_curated_evaluations,
    split_predictor_device_groups,
)


class TestHighQualitySFTBuilder(unittest.TestCase):
    def test_build_local_model_device_map_uses_all_visible_gpus_round_robin(self):
        with mock.patch("recipe.time_series_forecast.build_etth1_high_quality_sft.torch.cuda.is_available", return_value=True):
            with mock.patch("recipe.time_series_forecast.build_etth1_high_quality_sft.torch.cuda.device_count", return_value=4):
                device_map = build_local_model_device_map(
                    ["patchtst", "chronos2", "itransformer", "arima"],
                    default_device="cuda",
                )

        self.assertEqual(
            device_map,
            {
                "patchtst": "cuda:0",
                "chronos2": "cuda:1",
                "itransformer": "cuda:2",
            },
        )

    def test_split_predictor_device_groups_balances_visible_gpus(self):
        with mock.patch("recipe.time_series_forecast.build_etth1_high_quality_sft.torch.cuda.is_available", return_value=True):
            with mock.patch("recipe.time_series_forecast.build_etth1_high_quality_sft.torch.cuda.device_count", return_value=4):
                groups = split_predictor_device_groups(
                    [],
                    default_device="cuda",
                    num_workers=2,
                )

        self.assertEqual(groups, [["cuda:0", "cuda:2"], ["cuda:1", "cuda:3"]])

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

    def test_select_curated_evaluations_balances_best_models(self):
        evaluations = [
            {"sample_index": 0, "best_model": "arima", "selection_score": 0.95, "best_score": 0.95, "score_margin": 0.3},
            {"sample_index": 1, "best_model": "arima", "selection_score": 0.90, "best_score": 0.90, "score_margin": 0.2},
            {"sample_index": 2, "best_model": "patchtst", "selection_score": 0.60, "best_score": 0.60, "score_margin": 0.1},
            {"sample_index": 3, "best_model": "patchtst", "selection_score": 0.55, "best_score": 0.55, "score_margin": 0.1},
            {"sample_index": 4, "best_model": "itransformer", "selection_score": 0.58, "best_score": 0.58, "score_margin": 0.1},
            {"sample_index": 5, "best_model": "itransformer", "selection_score": 0.57, "best_score": 0.57, "score_margin": 0.1},
            {"sample_index": 6, "best_model": "chronos2", "selection_score": 0.54, "best_score": 0.54, "score_margin": 0.1},
            {"sample_index": 7, "best_model": "chronos2", "selection_score": 0.53, "best_score": 0.53, "score_margin": 0.1},
        ]
        selected = select_curated_evaluations(evaluations, 4)
        self.assertEqual({item["best_model"] for item in selected}, {"arima", "patchtst", "itransformer", "chronos2"})

    def test_process_split_writes_full_eval_and_curated_eval(self):
        records = [{"index": idx, "uid": f"sample-{idx}"} for idx in range(4)]
        evaluations = [
            {
                "sample_index": idx,
                "best_model": ["arima", "patchtst", "itransformer", "chronos2"][idx],
                "best_score": 0.8 - idx * 0.05,
                "second_best_model": "chronos2",
                "second_best_score": 0.4,
                "score_margin": 0.1,
                "selection_score": 0.8 - idx * 0.05,
                "model_scores": {},
                "teacher_prediction_text": "1.0\n2.0",
                "teacher_prediction_source": "reference_teacher",
                "reference_teacher_error": 1.0 + idx,
            }
            for idx in range(4)
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            with mock.patch(
                "recipe.time_series_forecast.build_etth1_high_quality_sft.evaluate_candidates",
                return_value=evaluations,
            ):
                curated_records, full_evaluations = process_split(
                    split_name="train",
                    records=records,
                    models=["patchtst", "chronos2", "itransformer", "arima"],
                    eval_count=0,
                    candidate_count=2,
                    target_count=2,
                    max_concurrency=1,
                    predictor_mode="service",
                    predictor=None,
                    model_service_url="http://127.0.0.1:8994",
                    output_dir=output_dir,
                )

            self.assertEqual(len(full_evaluations), 4)
            self.assertEqual(len(curated_records), 2)

            full_eval_rows = [json.loads(line) for line in (output_dir / "train_teacher_eval.jsonl").read_text().splitlines()]
            curated_eval_rows = [json.loads(line) for line in (output_dir / "train_teacher_eval_curated.jsonl").read_text().splitlines()]
            self.assertEqual(len(full_eval_rows), 4)
            self.assertEqual(len(curated_eval_rows), 2)

    def test_process_split_reuses_existing_eval_rows(self):
        records = [{"index": idx, "uid": f"sample-{idx}"} for idx in range(3)]
        existing_row = {
            "sample_index": 0,
            "best_model": "patchtst",
            "best_score": 0.9,
            "second_best_model": "itransformer",
            "second_best_score": 0.8,
            "score_margin": 0.1,
            "selection_score": 0.925,
            "model_scores": {"patchtst": 0.9},
            "model_score_details": {},
            "model_errors": {},
            "teacher_prediction_text": "1.0\n2.0",
            "teacher_prediction_source": "reference_teacher",
            "reference_teacher_error": 1.0,
        }
        pending_rows = [
            {
                "sample_index": idx,
                "best_model": "arima",
                "best_score": 0.7,
                "second_best_model": "patchtst",
                "second_best_score": 0.6,
                "score_margin": 0.1,
                "selection_score": 0.725,
                "model_scores": {"arima": 0.7},
                "model_score_details": {},
                "model_errors": {},
                "teacher_prediction_text": "1.0\n2.0",
                "teacher_prediction_source": "reference_teacher",
                "reference_teacher_error": 2.0 + idx,
            }
            for idx in [1, 2]
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            (output_dir / "train_teacher_eval.jsonl").write_text(
                json.dumps(existing_row, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            with mock.patch(
                "recipe.time_series_forecast.build_etth1_high_quality_sft.evaluate_candidates",
                return_value=pending_rows,
            ) as evaluate_mock:
                curated_records, full_evaluations = process_split(
                    split_name="train",
                    records=records,
                    models=["patchtst", "chronos2", "itransformer", "arima"],
                    eval_count=0,
                    candidate_count=3,
                    target_count=2,
                    max_concurrency=1,
                    predictor_mode="service",
                    predictor=None,
                    model_service_url="http://127.0.0.1:8994",
                    output_dir=output_dir,
                    resume_eval=True,
                )

            self.assertEqual(evaluate_mock.call_count, 1)
            pending_samples = evaluate_mock.call_args.args[0]
            self.assertEqual([sample["index"] for sample in pending_samples], [1, 2])
            self.assertEqual(len(full_evaluations), 3)
            self.assertEqual(len(curated_records), 2)

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
