import unittest

from omegaconf import OmegaConf

from arft.main_agent_ppo import _ensure_custom_reward_function_config


class TestMainAgentPPOConfig(unittest.TestCase):
    def test_ensure_custom_reward_function_config_sets_timeseries_default(self) -> None:
        config = OmegaConf.create(
            {
                "reward": {
                    "custom_reward_function": {
                        "path": None,
                        "name": None,
                    }
                }
            }
        )

        _ensure_custom_reward_function_config(config)

        self.assertTrue(str(config.reward.custom_reward_function.path).endswith("recipe/time_series_forecast/reward.py"))
        self.assertEqual(config.reward.custom_reward_function.name, "compute_score")


if __name__ == "__main__":
    unittest.main()
