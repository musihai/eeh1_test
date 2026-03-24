import unittest

from recipe.time_series_forecast.build_etth1_heuristic_curated_sft import (
    allocate_balanced_model_quotas,
    allocate_proportional_quotas,
    select_curated_records,
)


class TestHeuristicCuratedSFTBuilder(unittest.TestCase):
    def test_allocate_proportional_quotas_preserves_group_presence(self):
        quotas = allocate_proportional_quotas({"easy": 10, "medium": 20, "hard": 30}, 6)
        self.assertEqual(sum(quotas.values()), 6)
        self.assertGreaterEqual(quotas["easy"], 1)
        self.assertGreaterEqual(quotas["medium"], 1)
        self.assertGreaterEqual(quotas["hard"], 1)

    def test_allocate_balanced_model_quotas_prefers_equal_model_coverage(self):
        records_by_model = {
            "patchtst": [{} for _ in range(100)],
            "itransformer": [{} for _ in range(80)],
            "arima": [{} for _ in range(60)],
            "chronos2": [{} for _ in range(40)],
        }
        quotas = allocate_balanced_model_quotas(records_by_model, 20)
        self.assertEqual(quotas["patchtst"], 5)
        self.assertEqual(quotas["itransformer"], 5)
        self.assertEqual(quotas["arima"], 5)
        self.assertEqual(quotas["chronos2"], 5)

    def test_select_curated_records_balances_models_and_stages(self):
        records = []
        sample_index = 0
        for model_name in ("patchtst", "itransformer", "arima", "chronos2"):
            for stage_name in ("easy", "medium", "hard"):
                for replica in range(4):
                    records.append(
                        {
                            "index": sample_index,
                            "heuristic_selected_prediction_model": model_name,
                            "difficulty_stage": stage_name,
                            "heuristic_offline_best_agreement": replica % 2 == 0,
                            "offline_margin": 0.4 - replica * 0.01,
                            "heuristic_score_margin": 0.8 - replica * 0.01,
                        }
                    )
                    sample_index += 1

        selected = select_curated_records(records, target_count=12)
        self.assertEqual(len(selected), 12)

        by_model = {}
        by_stage = {}
        for record in selected:
            by_model[record["heuristic_selected_prediction_model"]] = (
                by_model.get(record["heuristic_selected_prediction_model"], 0) + 1
            )
            by_stage[record["difficulty_stage"]] = by_stage.get(record["difficulty_stage"], 0) + 1

        self.assertEqual(by_model, {"patchtst": 3, "itransformer": 3, "arima": 3, "chronos2": 3})
        self.assertEqual(by_stage, {"easy": 4, "medium": 4, "hard": 4})


if __name__ == "__main__":
    unittest.main()
