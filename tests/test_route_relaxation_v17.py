import json
import unittest
from pathlib import Path

import pandas as pd

from recipe.time_series_forecast.build_etth1_routing_override_bootstrap import _select_split_records
from recipe.time_series_forecast.build_etth1_sft_dataset import (
    TURN3_TARGET_MODE_ENGINEERING_REFINE,
    build_sft_records,
)
from recipe.time_series_forecast.probe_routing_policy import _delta_vs_default, _parse_generated_route


def _score_details(best: str, *, default_error: float, best_error: float, second: str, second_error: float) -> dict[str, dict[str, float]]:
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


class RouteRelaxationV17Tests(unittest.TestCase):
    def test_select_split_records_builds_keep_default_plus_override_distribution(self) -> None:
        source_records = []
        evaluations = []
        sample_index = 0

        for rank in range(8):
            source_records.append(
                {
                    "index": sample_index,
                    "uid": f"default-{sample_index}",
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
                        default_error=1.0 + rank * 0.1,
                        best_error=1.0 + rank * 0.1,
                        second="arima",
                        second_error=1.8 + rank * 0.1,
                    ),
                    "teacher_prediction_text": "2017-01-01 00:00:00 1.0\n2017-01-01 01:00:00 2.0",
                }
            )
            sample_index += 1

        for model_name, deltas in {
            "patchtst": [1.2, 0.9, 0.4],
            "arima": [1.1, 0.8, 0.3],
            "chronos2": [1.0, 0.7, 0.2],
        }.items():
            for delta in deltas:
                source_records.append(
                    {
                        "index": sample_index,
                        "uid": f"{model_name}-{sample_index}",
                        "raw_prompt": [{"role": "user", "content": "dummy"}],
                        "reward_model": {"ground_truth": "2017-01-01 00:00:00 1.0\n2017-01-01 01:00:00 2.0"},
                    }
                )
                evaluations.append(
                    {
                        "sample_index": sample_index,
                        "best_model": model_name,
                        "second_best_model": "itransformer",
                        "best_score": 0.8,
                        "second_best_score": 0.7,
                        "score_margin": 0.1,
                        "model_scores": {model_name: 0.8, "itransformer": 0.7},
                        "model_score_details": _score_details(
                            model_name,
                            default_error=2.0,
                            best_error=2.0 - delta,
                            second="itransformer",
                            second_error=2.0,
                        ),
                        "teacher_prediction_text": "2017-01-01 00:00:00 1.0\n2017-01-01 01:00:00 2.0",
                    }
                )
                sample_index += 1

        selected, metadata = _select_split_records(
            source_records=source_records,
            evaluations=evaluations,
            target_count=10,
            default_expert="itransformer",
            keep_default_ratio=0.7,
            override_top_fraction=0.34,
        )

        self.assertEqual(len(selected), 10)
        self.assertEqual(
            metadata["selected_route_label_distribution"],
            {
                "keep_default": 7,
                "override_to_arima": 1,
                "override_to_chronos2": 1,
                "override_to_patchtst": 1,
            },
        )
        self.assertAlmostEqual(metadata["selected_keep_default_ratio"], 0.7, places=6)
        self.assertTrue(all(record["route_label"] for record in selected))
        self.assertTrue(all(record["default_expert"] == "itransformer" for record in selected))

    def test_build_sft_records_uses_route_time_series_for_v17_override_sample(self) -> None:
        source_jsonl = Path("dataset/ett_rl_etth1_paper_same2/train.jsonl")
        with source_jsonl.open("r", encoding="utf-8") as handle:
            sample = json.loads(next(handle))

        sample["reference_teacher_model"] = "patchtst"
        sample["teacher_eval_second_best_model"] = "arima"
        sample["teacher_prediction_text"] = str(sample["reward_model"]["ground_truth"])
        sample["teacher_prediction_source"] = "reference_teacher"
        sample["default_expert"] = "itransformer"
        sample["route_label"] = "override_to_patchtst"
        sample["route_decision"] = "override"
        sample["route_override_model"] = "patchtst"
        sample["route_label_confidence"] = "high"

        records = build_sft_records(
            sample,
            turn3_target_mode=TURN3_TARGET_MODE_ENGINEERING_REFINE,
        )
        routing_record = [record for record in records if record["turn_stage"] == "routing"][0]
        routing_messages = routing_record["messages"]

        self.assertEqual(routing_messages[-1]["tool_calls"][0]["function"]["name"], "route_time_series")
        self.assertEqual(
            json.loads(routing_messages[-1]["tool_calls"][0]["function"]["arguments"]),
            {"decision": "override", "model_name": "patchtst"},
        )
        self.assertEqual([tool["function"]["name"] for tool in routing_record["tools"]], ["route_time_series"])
        self.assertEqual(routing_record["selected_prediction_model"], "patchtst")
        self.assertEqual(routing_record["route_default_expert"], "itransformer")
        self.assertIn("### Default Expert: itransformer", routing_messages[1]["content"])

    def test_route_probe_helpers_parse_default_override_actions(self) -> None:
        row = pd.Series(
            {
                "teacher_eval_score_details": {
                    "patchtst": {"orig_mse": 1.2},
                    "itransformer": {"orig_mse": 2.0},
                    "arima": {"orig_mse": 1.5},
                    "chronos2": {"orig_mse": 2.2},
                },
                "default_expert": "itransformer",
                "route_label": "override_to_patchtst",
            }
        )

        keep_result = _parse_generated_route(
            generated='<tool_call>{"name":"route_time_series","arguments":{"decision":"keep_default"}}</tool_call>',
            default_expert="itransformer",
        )
        override_result = _parse_generated_route(
            generated='<tool_call>{"name":"route_time_series","arguments":{"decision":"override","model_name":"patchtst"}}</tool_call>',
            default_expert="itransformer",
        )

        self.assertEqual(keep_result["requested_model"], "itransformer")
        self.assertEqual(keep_result["predicted_label"], "keep_default")
        self.assertEqual(override_result["requested_model"], "patchtst")
        self.assertEqual(override_result["predicted_label"], "override_to_patchtst")
        self.assertAlmostEqual(_delta_vs_default(row, "patchtst", "itransformer"), -0.8)
        self.assertAlmostEqual(_delta_vs_default(row, "itransformer", "itransformer"), 0.0)


if __name__ == "__main__":
    unittest.main()
