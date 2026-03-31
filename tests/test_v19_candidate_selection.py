import unittest

from recipe.time_series_forecast.build_etth1_sft_dataset import _paper_turn3_protocol_reason
from recipe.time_series_forecast.build_etth1_v19_sft_dataset import (
    MODE_FINAL_SELECT_ONLY,
    _balance_final_select_frame,
    build_v19_sft_records,
)
from recipe.time_series_forecast.candidate_selection_support import (
    compute_candidate_visible_metrics,
    materialize_candidate_selection,
    parse_candidate_selection_protocol,
)
from recipe.time_series_forecast.prompts import build_v19_final_select_prompt, build_v19_risk_gate_prompt


def _history_text() -> str:
    return "\n".join(
        f"2017-01-01 0{i}:00:00 {float(i):.4f}"
        for i in range(4)
    )


def _candidate(candidate_id: str, *, model_name: str, path_type: str, kind: str) -> dict:
    prediction_text = "\n".join(
        f"2017-01-02 0{i}:00:00 {float(i + 1):.4f}"
        for i in range(4)
    )
    return {
        "candidate_id": candidate_id,
        "model_name": model_name,
        "path_type": path_type,
        "candidate_kind": kind,
        "prediction_text": prediction_text,
        "compact_prediction_text": prediction_text,
        "score_details": {
            "score": 0.5,
            "orig_mse": 1.0,
        },
    }


class V19CandidateSelectionTests(unittest.TestCase):
    def test_parse_candidate_selection_protocol_accepts_candidate_id_answer(self) -> None:
        think, candidate_id, parse_mode, reject_reason = parse_candidate_selection_protocol(
            "<think>pick the smoother one</think><answer>candidate_id=itransformer__keep</answer>",
            allowed_candidate_ids=["itransformer__keep", "patchtst__keep"],
        )
        self.assertEqual(think, "pick the smoother one")
        self.assertEqual(candidate_id, "itransformer__keep")
        self.assertEqual(parse_mode, "candidate_selection_protocol")
        self.assertIsNone(reject_reason)

    def test_materialize_candidate_selection_renders_prediction_text(self) -> None:
        answer, candidate_id, parse_mode, reject_reason = materialize_candidate_selection(
            response_text="<think>pick it</think><answer>candidate_id=patchtst__keep</answer>",
            candidate_prediction_text_map={
                "patchtst__keep": "2017-01-02 00:00:00 1.0000\n2017-01-02 01:00:00 2.0000",
            },
        )
        self.assertEqual(candidate_id, "patchtst__keep")
        self.assertEqual(parse_mode, "candidate_selection_protocol")
        self.assertIsNone(reject_reason)
        self.assertIn("<answer>", answer)
        self.assertIn("2017-01-02 00:00:00 1.0000", answer)

    def test_turn3_protocol_validator_accepts_candidate_id_line(self) -> None:
        reason = _paper_turn3_protocol_reason(
            "<think>compare the visible candidates</think><answer>candidate_id=chronos2__keep</answer>",
            expected_len=96,
        )
        self.assertEqual(reason, "ok")

    def test_build_v19_prompts_include_new_decision_space(self) -> None:
        risk_prompt = build_v19_risk_gate_prompt(
            data_source="ETTh1",
            target_column="OT",
            lookback_window=96,
            forecast_horizon=96,
            time_series_data=_history_text(),
            history_analysis=["Basic Statistics:\n  Median: 2.0000"],
            default_expert="itransformer",
            fixed_expand=True,
        )
        self.assertIn("default_risky", risk_prompt)
        self.assertIn("does NOT choose the final expert", risk_prompt)

        final_prompt = build_v19_final_select_prompt(
            data_source="ETTh1",
            target_column="OT",
            lookback_window=96,
            forecast_horizon=96,
            time_series_data=_history_text(),
            history_analysis=["Diagnostic summary"],
            default_expert="itransformer",
            default_candidate_id="itransformer__keep",
            expanded=True,
            candidates=[
                _candidate("itransformer__keep", model_name="itransformer", path_type="default", kind="baseline"),
                _candidate("patchtst__keep", model_name="patchtst", path_type="alternative", kind="baseline"),
            ],
        )
        self.assertIn("candidate_id=itransformer__keep", final_prompt)
        self.assertIn("[candidate_id=patchtst__keep]", final_prompt)
        self.assertIn("recent_match=", final_prompt)
        self.assertIn("vs_default_gain=", final_prompt)
        self.assertIn("direction_check=", final_prompt)
        self.assertIn("Recent Target Rows", final_prompt)
        self.assertIn("Head Values:", final_prompt)
        self.assertIn("Tail Values:", final_prompt)
        self.assertNotIn("Forecast Values:", final_prompt)

    def test_build_v19_sft_records_final_select_only_emits_candidate_answer(self) -> None:
        row = {
            "uid": "sample-1",
            "source_sample_index": 1,
            "data_source": "ETTh1",
            "target_column": "OT",
            "lookback_window": 4,
            "forecast_horizon": 4,
            "historical_data": _history_text(),
            "default_expert": "itransformer",
            "default_candidate_id": "itransformer__keep",
            "analysis_history": ["Basic Statistics:\n  Median: 2.0000"],
            "risk_label": "default_risky",
            "risk_value_rel": 0.25,
            "final_candidate_label": "patchtst__keep",
            "final_candidate_error": 0.9,
            "default_candidate_error": 1.2,
            "final_vs_default_error": -0.3,
            "candidate_score_details": {"itransformer__keep": {"score": 0.4, "orig_mse": 1.2}},
            "candidate_prediction_text_map": {"patchtst__keep": "2017-01-02 00:00:00 1.0000"},
            "default_candidates": [
                _candidate("itransformer__keep", model_name="itransformer", path_type="default", kind="baseline"),
            ],
            "alt_candidates": [
                _candidate("patchtst__keep", model_name="patchtst", path_type="alternative", kind="baseline"),
            ],
        }
        records = build_v19_sft_records(
            row,
            mode=MODE_FINAL_SELECT_ONLY,
            turn2_policy="fixed_expand",
            split_name="train",
        )
        self.assertEqual(len(records), 1)
        record = records[0]
        self.assertEqual(record["turn_stage"], "final_select")
        self.assertTrue(record["paper_turn3_required"])
        assistant_content = record["messages"][-1]["content"]
        self.assertIn("candidate_id=patchtst__keep", assistant_content)
        self.assertIn("patchtst__keep", assistant_content)
        self.assertIn("itransformer__keep", assistant_content)
        self.assertIn("recent", assistant_content)
        self.assertIn("visible_candidate_metrics", record)

    def test_compute_candidate_visible_metrics_reports_default_gains(self) -> None:
        metrics = compute_candidate_visible_metrics(
            historical_data=_history_text(),
            target_column="OT",
            default_candidate_id="itransformer__keep",
            candidates=[
                _candidate("itransformer__keep", model_name="itransformer", path_type="default", kind="baseline"),
                _candidate("patchtst__keep", model_name="patchtst", path_type="alternative", kind="baseline"),
            ],
        )
        self.assertIn("itransformer__keep", metrics)
        self.assertIn("patchtst__keep", metrics)
        self.assertIn("level_gain_vs_default", metrics["patchtst__keep"])
        self.assertIn("direction_match", metrics["patchtst__keep"])

    def test_build_v19_sft_records_can_shuffle_candidate_order(self) -> None:
        row = {
            "uid": "sample-2",
            "source_sample_index": 2,
            "data_source": "ETTh1",
            "target_column": "OT",
            "lookback_window": 4,
            "forecast_horizon": 4,
            "historical_data": _history_text(),
            "default_expert": "itransformer",
            "default_candidate_id": "itransformer__keep",
            "analysis_history": ["Diagnostic summary"],
            "risk_label": "default_risky",
            "risk_value_rel": 0.25,
            "final_candidate_label": "chronos2__keep",
            "final_candidate_error": 0.9,
            "default_candidate_error": 1.2,
            "final_vs_default_error": -0.3,
            "candidate_score_details": {},
            "candidate_prediction_text_map": {},
            "default_candidates": [
                _candidate("itransformer__keep", model_name="itransformer", path_type="default", kind="baseline"),
            ],
            "alt_candidates": [
                _candidate("patchtst__keep", model_name="patchtst", path_type="alternative", kind="baseline"),
                _candidate("arima__keep", model_name="arima", path_type="alternative", kind="baseline"),
                _candidate("chronos2__keep", model_name="chronos2", path_type="alternative", kind="baseline"),
            ],
        }
        shuffled = build_v19_sft_records(
            row,
            mode=MODE_FINAL_SELECT_ONLY,
            turn2_policy="fixed_expand",
            split_name="train",
            shuffle_candidate_seed=19,
        )[0]
        prompt_order = shuffled["prompt_candidate_order"]
        self.assertIn("chronos2__keep", prompt_order)
        self.assertGreaterEqual(shuffled["gold_candidate_prompt_rank"], 0)

    def test_balance_final_select_frame_oversamples_minority_labels(self) -> None:
        import pandas as pd

        frame = pd.DataFrame(
            [
                {"final_candidate_label": "a"},
                {"final_candidate_label": "a"},
                {"final_candidate_label": "b"},
            ]
        )
        balanced = _balance_final_select_frame(frame, seed=19)
        counts = balanced["final_candidate_label"].value_counts().to_dict()
        self.assertEqual(counts["a"], counts["b"])


if __name__ == "__main__":
    unittest.main()
