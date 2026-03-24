import asyncio
import json
import tempfile
import unittest
from argparse import Namespace
from unittest import mock
from pathlib import Path

import pandas as pd

from recipe.time_series_forecast.build_etth1_high_quality_sft import (
    _select_reference_teacher_model,
    build_local_model_device_map,
    ensure_service_ready,
    evenly_spaced_records,
    evaluate_teacher_for_sample,
    load_existing_evaluations,
    main,
    process_split,
    select_curated_evaluations,
    split_predictor_device_groups,
)
from recipe.time_series_forecast.dataset_identity import (
    DATASET_KIND_RL_JSONL,
    DATASET_KIND_RUNTIME_SFT_PARQUET,
    DATASET_KIND_TEACHER_CURATED_SFT,
)


class TestHighQualitySFTBuilder(unittest.TestCase):
    def test_select_reference_teacher_model_prefers_lowest_orig_mse(self):
        selected = _select_reference_teacher_model(
            {
                "arima": 0.72,
                "itransformer": 0.61,
                "patchtst": 0.59,
            },
            {
                "arima": {"orig_mse": 6.82},
                "itransformer": {"orig_mse": 5.42},
                "patchtst": {"orig_mse": 6.40},
            },
        )

        self.assertEqual(selected, "itransformer")

    def test_select_reference_teacher_model_falls_back_to_reward_best_when_errors_missing(self):
        selected = _select_reference_teacher_model(
            {
                "arima": 0.72,
                "itransformer": 0.61,
            },
            {
                "arima": {"orig_mse": float("nan")},
                "itransformer": {},
            },
        )

        self.assertEqual(selected, "arima")

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

    def test_select_curated_evaluations_preserves_local_refine_quota(self):
        evaluations = [
            {
                "sample_index": idx,
                "best_model": ["arima", "patchtst", "itransformer", "chronos2"][idx % 4],
                "selection_score": 0.9 - idx * 0.01,
                "best_score": 0.9 - idx * 0.01,
                "score_margin": 0.05,
                "turn3_target_type": "local_refine" if idx < 3 else "validated_keep",
            }
            for idx in range(10)
        ]
        selected = select_curated_evaluations(
            evaluations,
            6,
            min_local_refine_ratio=0.30,
        )
        local_refine_count = sum(1 for item in selected if item["turn3_target_type"] == "local_refine")
        self.assertGreaterEqual(local_refine_count, 2)

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
                with mock.patch(
                    "recipe.time_series_forecast.build_etth1_high_quality_sft.annotate_turn3_targets",
                    return_value=(
                        [
                            {
                                **record,
                                "turn3_target_type": "validated_keep",
                                "turn3_trigger_reason": "evidence_consistent",
                                "refine_ops_signature": "none",
                                "refine_gain_mse": 0.0,
                                "refine_gain_mae": 0.0,
                                "selected_feature_tool_signature": "extract_basic_statistics",
                                "selected_feature_tool_count": 1,
                                "selected_prediction_model": "chronos2",
                                "base_prediction_source": "reference_teacher_cached",
                            }
                            for record in records[::3]
                        ],
                        [],
                    ),
                ):
                    curated_records, full_evaluations, annotation_summary = process_split(
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
            self.assertEqual(annotation_summary["turn3_annotation_error_count"], 0)

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
                with mock.patch(
                    "recipe.time_series_forecast.build_etth1_high_quality_sft.annotate_turn3_targets",
                    return_value=(
                        [
                            {
                                **record,
                                "turn3_target_type": "validated_keep",
                                "turn3_trigger_reason": "evidence_consistent",
                                "refine_ops_signature": "none",
                                "refine_gain_mse": 0.0,
                                "refine_gain_mae": 0.0,
                                "selected_feature_tool_signature": "extract_basic_statistics",
                                "selected_feature_tool_count": 1,
                                "selected_prediction_model": "chronos2",
                                "base_prediction_source": "reference_teacher_cached",
                            }
                            for record in records
                        ],
                        [],
                    ),
                ):
                    curated_records, full_evaluations, annotation_summary = process_split(
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
            self.assertEqual(annotation_summary["turn3_annotation_error_count"], 0)

    def test_load_existing_evaluations_normalizes_reference_teacher_fields(self):
        existing_row = {
            "sample_index": 0,
            "best_model": "arima",
            "best_score": 0.72,
            "second_best_model": "itransformer",
            "second_best_score": 0.61,
            "score_margin": 0.11,
            "selection_score": 0.7475,
            "model_scores": {"arima": 0.72, "itransformer": 0.61},
            "model_score_details": {
                "arima": {"orig_mse": 6.82},
                "itransformer": {"orig_mse": 5.42},
            },
            "teacher_prediction_text": "1.0\n2.0",
            "teacher_prediction_source": "reference_teacher",
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "train_teacher_eval.jsonl"
            path.write_text(json.dumps(existing_row, ensure_ascii=False) + "\n", encoding="utf-8")
            loaded = load_existing_evaluations(path)

        self.assertEqual(loaded[0]["reference_teacher_model"], "itransformer")
        self.assertAlmostEqual(loaded[0]["reference_teacher_error"], 5.42, places=6)
        self.assertEqual(loaded[0]["reference_teacher_prediction_text"], "1.0\n2.0")

    def test_process_split_rejects_candidate_pool_larger_than_limited_eval_pool(self):
        records = [{"index": idx, "uid": f"sample-{idx}"} for idx in range(5)]

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            with self.assertRaisesRegex(ValueError, "candidate_count .* may not exceed eval_count"):
                process_split(
                    split_name="train",
                    records=records,
                    models=["patchtst", "chronos2", "itransformer", "arima"],
                    eval_count=2,
                    candidate_count=3,
                    target_count=2,
                    max_concurrency=1,
                    predictor_mode="service",
                    predictor=None,
                    model_service_url="http://127.0.0.1:8994",
                    output_dir=output_dir,
                )

    def test_process_split_raises_when_turn3_annotation_errors_exceed_budget(self):
        records = [{"index": idx, "uid": f"sample-{idx}"} for idx in range(3)]
        evaluations = [
            {
                "sample_index": idx,
                "best_model": "chronos2",
                "best_score": 0.8,
                "second_best_model": "patchtst",
                "second_best_score": 0.7,
                "score_margin": 0.1,
                "selection_score": 0.825,
                "model_scores": {"chronos2": 0.8},
                "model_score_details": {},
                "model_errors": {},
                "teacher_prediction_text": "1.0\n2.0",
                "teacher_prediction_source": "reference_teacher",
                "reference_teacher_error": 1.0,
            }
            for idx in range(3)
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            with mock.patch(
                "recipe.time_series_forecast.build_etth1_high_quality_sft.evaluate_candidates",
                return_value=evaluations,
            ):
                with mock.patch(
                    "recipe.time_series_forecast.build_etth1_high_quality_sft.build_sft_record",
                    side_effect=RuntimeError("annotation failed"),
                ):
                    with self.assertRaisesRegex(RuntimeError, "exceeded turn3 annotation error budget"):
                        process_split(
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
                            max_turn3_annotation_error_count=0,
                            max_turn3_annotation_error_ratio=0.0,
                        )

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

    def test_main_rebalances_train_parquet_and_writes_turn3_metadata(self):
        def _make_df(local_refine_count: int, keep_count: int) -> pd.DataFrame:
            rows = []
            sample_index = 0
            for _ in range(local_refine_count):
                rows.append(
                    {
                        "sample_index": sample_index,
                        "turn3_target_type": "local_refine",
                        "turn3_trigger_reason": "residual_signal",
                        "refine_ops_signature": "local_level_adjust",
                        "selected_feature_tool_signature": "extract_basic_statistics->extract_event_summary",
                        "reference_teacher_model": "chronos2",
                        "selected_prediction_model": "chronos2",
                        "base_prediction_source": "reference_teacher_cached",
                    }
                )
                sample_index += 1
            for _ in range(keep_count):
                rows.append(
                    {
                        "sample_index": sample_index,
                        "turn3_target_type": "validated_keep",
                        "turn3_trigger_reason": "evidence_consistent",
                        "refine_ops_signature": "none",
                        "selected_feature_tool_signature": "extract_basic_statistics->extract_event_summary",
                        "reference_teacher_model": "chronos2",
                        "selected_prediction_model": "chronos2",
                        "base_prediction_source": "reference_teacher_cached",
                    }
                )
                sample_index += 1
            return pd.DataFrame(rows)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            train_jsonl = tmp_path / "train.jsonl"
            val_jsonl = tmp_path / "val.jsonl"
            train_jsonl.write_text("{}\n", encoding="utf-8")
            val_jsonl.write_text("{}\n", encoding="utf-8")
            (tmp_path / "metadata.json").write_text(
                json.dumps(
                    {
                        "dataset_kind": DATASET_KIND_RL_JSONL,
                        "pipeline_stage": "curriculum_rl",
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            output_dir = tmp_path / "out"

            args = Namespace(
                train_jsonl=str(train_jsonl),
                val_jsonl=str(val_jsonl),
                test_jsonl=str(tmp_path / "missing_test.jsonl"),
                output_dir=str(output_dir),
                models="patchtst,chronos2,itransformer,arima",
                train_target_samples=10,
                val_target_samples=4,
                test_target_samples=0,
                train_eval_samples=10,
                val_eval_samples=4,
                test_eval_samples=0,
                train_candidate_samples=10,
                val_candidate_samples=4,
                test_candidate_samples=0,
                train_min_local_refine_ratio=0.30,
                max_concurrency=1,
                num_workers=2,
                local_batch_size=8,
                resume_teacher_eval=True,
                model_service_url="http://127.0.0.1:8994",
                predictor_mode="local",
                predictor_device="cpu",
                predictor_devices="",
                max_turn3_annotation_error_count=0,
                max_turn3_annotation_error_ratio=0.0,
            )

            train_curated = [{"reference_teacher_model": "chronos2"} for _ in range(10)]
            train_eval = [{"sample_index": idx} for idx in range(10)]
            val_curated = [{"reference_teacher_model": "chronos2"} for _ in range(4)]
            val_eval = [{"sample_index": idx} for idx in range(4)]

            def _fake_convert_jsonl_to_sft_parquet(*, input_path, output_path, max_samples=-1):
                input_name = Path(input_path).name
                if input_name == "train_curated.jsonl":
                    dataframe = _make_df(local_refine_count=1, keep_count=9)
                elif input_name == "val_curated.jsonl":
                    dataframe = _make_df(local_refine_count=1, keep_count=3)
                else:
                    raise AssertionError(f"Unexpected curated input: {input_name}")
                Path(output_path).parent.mkdir(parents=True, exist_ok=True)
                dataframe.to_parquet(output_path, index=False)
                return dataframe

            with mock.patch("recipe.time_series_forecast.build_etth1_high_quality_sft.parse_args", return_value=args):
                with mock.patch(
                    "recipe.time_series_forecast.build_etth1_high_quality_sft.load_jsonl_records",
                    side_effect=[[{"index": 0}], [{"index": 1}]],
                ):
                    with mock.patch(
                        "recipe.time_series_forecast.build_etth1_high_quality_sft.process_split",
                        side_effect=[
                            (
                                train_curated,
                                train_eval,
                                {"turn3_annotation_error_count": 0, "turn3_annotation_error_ratio": 0.0},
                            ),
                            (
                                val_curated,
                                val_eval,
                                {"turn3_annotation_error_count": 0, "turn3_annotation_error_ratio": 0.0},
                            ),
                        ],
                    ):
                        with mock.patch(
                            "recipe.time_series_forecast.build_etth1_high_quality_sft.convert_jsonl_to_sft_parquet",
                            side_effect=_fake_convert_jsonl_to_sft_parquet,
                        ):
                            main()

            train_df = pd.read_parquet(output_dir / "train.parquet")
            self.assertEqual(train_df["turn3_target_type"].value_counts(dropna=False).to_dict(), {"validated_keep": 2, "local_refine": 1})

            metadata = json.loads((output_dir / "metadata.json").read_text())
            self.assertEqual(metadata["dataset_kind"], DATASET_KIND_RUNTIME_SFT_PARQUET)
            self.assertEqual(metadata["pipeline_stage"], "teacher200_runtime_sft")
            self.assertEqual(metadata["curated_jsonl_dataset_kind"], DATASET_KIND_TEACHER_CURATED_SFT)
            self.assertEqual(metadata["train_samples_before_balance"], 10)
            self.assertEqual(metadata["train_samples"], 3)
            self.assertEqual(
                metadata["train_turn3_target_type_distribution_before_balance"],
                {"local_refine": 1, "validated_keep": 9},
            )
            self.assertEqual(
                metadata["train_turn3_target_type_distribution"],
                {"local_refine": 1, "validated_keep": 2},
            )
            self.assertEqual(
                metadata["val_turn3_target_type_distribution"],
                {"local_refine": 1, "validated_keep": 3},
            )


if __name__ == "__main__":
    unittest.main()
