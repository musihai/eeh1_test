import unittest

import pandas as pd

from recipe.time_series_forecast.build_etth1_routing_bootstrap import _select_split_records
from recipe.time_series_forecast.probe_refinement_protocol import _parse_generated_answer
from recipe.time_series_forecast.probe_routing_policy import _route_regret, _top2_models


def _score_details(best: str, second: str, *, best_error: float, second_error: float) -> dict[str, dict[str, float]]:
    payload = {
        "patchtst": {"orig_mse": 9.0},
        "itransformer": {"orig_mse": 9.5},
        "arima": {"orig_mse": 10.0},
        "chronos2": {"orig_mse": 10.5},
    }
    payload[best]["orig_mse"] = float(best_error)
    payload[second]["orig_mse"] = float(second_error)
    return payload


class RouteRescueV16Tests(unittest.TestCase):
    def test_select_split_records_balances_models_and_assigns_mid_high_tiers(self) -> None:
        source_records = []
        evaluations = []
        sample_index = 0
        for model_name in ("patchtst", "itransformer", "arima", "chronos2"):
            for rank in range(3):
                source_records.append(
                    {
                        "index": sample_index,
                        "uid": f"uid-{sample_index}",
                        "raw_prompt": [{"role": "user", "content": "dummy"}],
                        "reward_model": {"ground_truth": "2017-01-01 00:00:00 1.0\n2017-01-01 01:00:00 2.0"},
                    }
                )
                second = "chronos2" if model_name != "chronos2" else "arima"
                best_error = 1.0 + rank * 0.2
                second_error = 2.0 + rank * 0.2
                evaluations.append(
                    {
                        "sample_index": sample_index,
                        "best_model": model_name,
                        "second_best_model": second,
                        "best_score": 0.8 - rank * 0.01,
                        "second_best_score": 0.7 - rank * 0.01,
                        "score_margin": 0.1,
                        "model_scores": {model_name: 0.8, second: 0.7},
                        "model_score_details": _score_details(
                            model_name,
                            second,
                            best_error=best_error,
                            second_error=second_error,
                        ),
                        "teacher_prediction_text": "2017-01-01 00:00:00 1.0\n2017-01-01 01:00:00 2.0",
                        "teacher_prediction_source": "reference_teacher",
                    }
                )
                sample_index += 1

        selected, metadata = _select_split_records(
            source_records=source_records,
            evaluations=evaluations,
            target_count=8,
            top_fraction=0.67,
        )

        self.assertEqual(len(selected), 8)
        self.assertEqual(
            metadata["selected_distribution"],
            {"patchtst": 2, "itransformer": 2, "arima": 2, "chronos2": 2},
        )
        self.assertEqual(
            metadata["selected_confidence_distribution"],
            {"high": 4, "mid": 4},
        )
        self.assertTrue(all(item["route_bootstrap_confidence_tier"] in {"high", "mid"} for item in selected))

    def test_parse_generated_answer_materializes_refinement_decision_before_protocol_check(self) -> None:
        response_text = (
            "<think>\nChoose the supported local adjustment.\n</think>\n"
            "<answer>\n"
            "decision=local_level_adjust\n"
            "</answer><|im_end|>"
        )
        candidate_prediction_text_map = {
            "keep_baseline": "2017-01-01 00:00:00 1.0\n2017-01-01 01:00:00 2.0",
            "local_level_adjust": "2017-01-01 00:00:00 1.5\n2017-01-01 01:00:00 2.5",
        }

        parsed = _parse_generated_answer(
            response_text,
            2,
            candidate_prediction_text_map=candidate_prediction_text_map,
        )

        self.assertTrue(parsed["strict_ok"])
        self.assertEqual(parsed["decision_name"], "local_level_adjust")
        self.assertTrue(parsed["materialized"])
        self.assertTrue(parsed["exact_96_lines"])

    def test_route_probe_helpers_compute_top2_and_regret_from_score_details(self) -> None:
        row = pd.Series(
            {
                "teacher_eval_score_details": {
                    "patchtst": {"orig_mse": 2.0},
                    "itransformer": {"orig_mse": 1.0},
                    "arima": {"orig_mse": 1.3},
                    "chronos2": {"orig_mse": 3.0},
                }
            }
        )

        self.assertEqual(_top2_models(row), ["itransformer", "arima"])
        self.assertAlmostEqual(_route_regret(row, "arima"), 0.3)
        self.assertAlmostEqual(_route_regret(row, "itransformer"), 0.0)


if __name__ == "__main__":
    unittest.main()
