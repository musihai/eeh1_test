import unittest

from recipe.time_series_forecast.build_etth1_routing_proposal_bootstrap import (
    ROUTE_BUCKET_AMBIGUOUS,
    ROUTE_BUCKET_MUST_KEEP,
    ROUTE_BUCKET_MUST_OVERRIDE,
    _compute_override_threshold_rel_by_model,
    _label_record_v18,
    _record_identity,
    _select_split_records_v18,
)
from recipe.time_series_forecast.prompts import build_runtime_user_prompt


def _score_details(
    best: str,
    *,
    default_error: float,
    best_error: float,
    second: str,
    second_error: float,
) -> dict[str, dict[str, float]]:
    payload = {
        "patchtst": {"orig_mse": 4.0},
        "itransformer": {"orig_mse": 4.0},
        "arima": {"orig_mse": 4.0},
        "chronos2": {"orig_mse": 4.0},
    }
    payload["itransformer"]["orig_mse"] = float(default_error)
    payload[best]["orig_mse"] = float(best_error)
    payload[second]["orig_mse"] = float(second_error)
    return payload


class RouteProposalV18Tests(unittest.TestCase):
    def test_label_record_marks_clear_non_default_winner_as_override(self) -> None:
        record = {
            "best_model": "arima",
            "default_expert": "itransformer",
            "improvement_vs_default_rel": 0.42,
            "route_margin_rel": 0.12,
            "default_in_top2": False,
        }
        labeled = _label_record_v18(
            record,
            default_expert="itransformer",
            tau_keep=0.05,
            tau_margin=0.08,
            override_threshold_rel_by_model={"patchtst": 0.35, "arima": 0.35, "chronos2": 0.35},
        )
        self.assertEqual(labeled["route_bucket"], ROUTE_BUCKET_MUST_OVERRIDE)
        self.assertEqual(labeled["route_label"], "override_to_arima")
        self.assertEqual(labeled["route_target_model"], "arima")

    def test_label_record_sends_mid_gain_sample_to_ambiguous(self) -> None:
        record = {
            "best_model": "patchtst",
            "default_expert": "itransformer",
            "improvement_vs_default_rel": 0.18,
            "route_margin_rel": 0.10,
            "default_in_top2": True,
        }
        labeled = _label_record_v18(
            record,
            default_expert="itransformer",
            tau_keep=0.05,
            tau_margin=0.08,
            override_threshold_rel_by_model={"patchtst": 0.35, "arima": 0.35, "chronos2": 0.35},
        )
        self.assertEqual(labeled["route_bucket"], ROUTE_BUCKET_AMBIGUOUS)
        self.assertEqual(labeled["route_label"], "")

    def test_compute_thresholds_use_quantile_with_floor(self) -> None:
        records = [
            {"best_model": "patchtst", "improvement_vs_default_rel": 0.70},
            {"best_model": "patchtst", "improvement_vs_default_rel": 0.60},
            {"best_model": "patchtst", "improvement_vs_default_rel": 0.50},
            {"best_model": "arima", "improvement_vs_default_rel": 0.10},
            {"best_model": "arima", "improvement_vs_default_rel": 0.20},
            {"best_model": "chronos2", "improvement_vs_default_rel": 0.45},
            {"best_model": "itransformer", "improvement_vs_default_rel": 0.00},
        ]
        thresholds = _compute_override_threshold_rel_by_model(
            records,
            default_expert="itransformer",
            override_quantile=0.80,
            override_floor=0.35,
        )
        self.assertGreater(thresholds["patchtst"], 0.35)
        self.assertEqual(thresholds["arima"], 0.35)
        self.assertGreaterEqual(thresholds["chronos2"], 0.45)

    def test_select_split_records_v18_balances_train_and_excludes_ambiguous(self) -> None:
        source_records = []
        evaluations = []
        sample_index = 0

        for _ in range(6):
            source_records.append(
                {
                    "index": sample_index,
                    "uid": f"keep-{sample_index}",
                    "raw_prompt": [{"role": "user", "content": "dummy"}],
                    "reward_model": {"ground_truth": "2017-01-01 00:00:00 1.0\n2017-01-01 01:00:00 2.0"},
                }
            )
            evaluations.append(
                {
                    "sample_index": sample_index,
                    "best_model": "itransformer",
                    "second_best_model": "arima",
                    "best_score": 0.8,
                    "second_best_score": 0.7,
                    "score_margin": 0.1,
                    "model_scores": {"itransformer": 0.8, "arima": 0.7},
                    "model_score_details": _score_details(
                        "itransformer",
                        default_error=2.0,
                        best_error=2.0,
                        second="arima",
                        second_error=2.4,
                    ),
                }
            )
            sample_index += 1

        for model_name in ("patchtst", "arima", "chronos2"):
            for gain in (0.9, 0.8):
                second_model = "patchtst" if model_name != "patchtst" else "arima"
                source_records.append(
                    {
                        "index": sample_index,
                        "uid": f"override-{model_name}-{sample_index}",
                        "raw_prompt": [{"role": "user", "content": "dummy"}],
                        "reward_model": {"ground_truth": "2017-01-01 00:00:00 1.0\n2017-01-01 01:00:00 2.0"},
                    }
                )
                evaluations.append(
                    {
                        "sample_index": sample_index,
                        "best_model": model_name,
                        "second_best_model": second_model,
                        "best_score": 0.9,
                        "second_best_score": 0.7,
                        "score_margin": 0.2,
                        "model_scores": {model_name: 0.9, second_model: 0.7, "itransformer": 0.6},
                        "model_score_details": _score_details(
                            model_name,
                            default_error=2.0,
                            best_error=2.0 - gain,
                            second=second_model,
                            second_error=2.0 - gain + 0.2,
                        ),
                    }
                )
                sample_index += 1

        for model_name in ("patchtst", "arima", "chronos2"):
            second_model = "patchtst" if model_name != "patchtst" else "arima"
            source_records.append(
                {
                    "index": sample_index,
                    "uid": f"ambiguous-{model_name}-{sample_index}",
                    "raw_prompt": [{"role": "user", "content": "dummy"}],
                    "reward_model": {"ground_truth": "2017-01-01 00:00:00 1.0\n2017-01-01 01:00:00 2.0"},
                }
            )
            evaluations.append(
                {
                    "sample_index": sample_index,
                    "best_model": model_name,
                    "second_best_model": second_model,
                    "best_score": 0.82,
                    "second_best_score": 0.8,
                    "score_margin": 0.02,
                    "model_scores": {model_name: 0.82, second_model: 0.8, "itransformer": 0.79},
                    "model_score_details": _score_details(
                        model_name,
                        default_error=2.0,
                        best_error=1.7,
                        second=second_model,
                        second_error=1.72,
                    ),
                }
            )
            sample_index += 1

        selected_by_split, metadata = _select_split_records_v18(
            source_records=source_records,
            evaluations=evaluations,
            default_expert="itransformer",
            tau_keep=0.05,
            tau_margin=0.08,
            override_threshold_rel_by_model={"patchtst": 0.35, "arima": 0.35, "chronos2": 0.35},
            train_target_count=12,
            val_natural_target_count=6,
            val_balanced_target_count=6,
            test_target_count=6,
            split_name="train",
        )

        selected_train = selected_by_split["train"]
        self.assertEqual(len(selected_train), 12)
        self.assertEqual(metadata["contradictory_keep_count"], 0)
        self.assertEqual(
            metadata["selected_train"]["route_bucket_distribution"],
            {ROUTE_BUCKET_MUST_KEEP: 6, ROUTE_BUCKET_MUST_OVERRIDE: 6},
        )
        self.assertEqual(
            metadata["selected_train"]["route_label_distribution"],
            {
                "keep_default": 6,
                "override_to_arima": 2,
                "override_to_chronos2": 2,
                "override_to_patchtst": 2,
            },
        )

    def test_routing_prompt_uses_more_symmetric_proposal_wording(self) -> None:
        prompt = build_runtime_user_prompt(
            data_source="ETTh1",
            target_column="OT",
            lookback_window=96,
            forecast_horizon=96,
            time_series_data="1.0000\n2.0000\n3.0000",
            history_analysis=["Basic Statistics:\n  Median: 2.0000"],
            prediction_results=None,
            available_feature_tools=["extract_basic_statistics"],
            completed_feature_tools=["extract_basic_statistics"],
            routing_feature_payload={
                "extract_basic_statistics": {
                    "acf1": 0.91,
                    "acf_seasonal": 0.12,
                    "cusum_max": 42.0,
                }
            },
            turn_stage="routing",
            route_default_expert="itransformer",
        )
        self.assertIn("### Default Expert: itransformer", prompt)
        self.assertIn("`keep_default` keeps the default path", prompt)
        self.assertIn("`override` proposes one alternative expert path", prompt)
        self.assertNotIn("only when the evidence clearly supports", prompt)
        self.assertNotIn("does not justify a meaningful override", prompt)

    def test_val_balanced_and_val_natural_do_not_overlap(self) -> None:
        source_records = []
        evaluations = []
        sample_index = 0

        for _ in range(40):
            source_records.append(
                {
                    "index": sample_index,
                    "uid": f"keep-{sample_index}",
                    "raw_prompt": [{"role": "user", "content": "dummy"}],
                    "reward_model": {"ground_truth": "2017-01-01 00:00:00 1.0\n2017-01-01 01:00:00 2.0"},
                }
            )
            evaluations.append(
                {
                    "sample_index": sample_index,
                    "best_model": "itransformer",
                    "second_best_model": "arima",
                    "best_score": 0.8,
                    "second_best_score": 0.7,
                    "score_margin": 0.1,
                    "model_scores": {"itransformer": 0.8, "arima": 0.7},
                    "model_score_details": _score_details(
                        "itransformer",
                        default_error=2.0,
                        best_error=2.0,
                        second="arima",
                        second_error=2.4,
                    ),
                }
            )
            sample_index += 1

        for model_name in ("patchtst", "arima", "chronos2"):
            second_model = "patchtst" if model_name != "patchtst" else "arima"
            for gain in (0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6):
                source_records.append(
                    {
                        "index": sample_index,
                        "uid": f"override-{model_name}-{sample_index}",
                        "raw_prompt": [{"role": "user", "content": "dummy"}],
                        "reward_model": {"ground_truth": "2017-01-01 00:00:00 1.0\n2017-01-01 01:00:00 2.0"},
                    }
                )
                evaluations.append(
                    {
                        "sample_index": sample_index,
                        "best_model": model_name,
                        "second_best_model": second_model,
                        "best_score": 0.9,
                        "second_best_score": 0.7,
                        "score_margin": 0.2,
                        "model_scores": {model_name: 0.9, second_model: 0.7, "itransformer": 0.6},
                        "model_score_details": _score_details(
                            model_name,
                            default_error=2.0,
                            best_error=2.0 - gain,
                            second=second_model,
                            second_error=2.0 - gain + 0.2,
                        ),
                    }
                )
                sample_index += 1

        selected_by_split, _ = _select_split_records_v18(
            source_records=source_records,
            evaluations=evaluations,
            default_expert="itransformer",
            tau_keep=0.05,
            tau_margin=0.08,
            override_threshold_rel_by_model={"patchtst": 0.35, "arima": 0.35, "chronos2": 0.35},
            train_target_count=16,
            val_natural_target_count=18,
            val_balanced_target_count=18,
            test_target_count=16,
            split_name="val",
        )

        val_balanced_ids = {_record_identity(item) for item in selected_by_split["val_balanced"]}
        val_natural_ids = {_record_identity(item) for item in selected_by_split["val_natural"]}
        self.assertFalse(val_balanced_ids & val_natural_ids)


if __name__ == "__main__":
    unittest.main()
