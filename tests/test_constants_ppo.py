import os
import unittest
from unittest.mock import patch

from verl.trainer.constants_ppo import get_ppo_ray_runtime_env


class TestPPORuntimeEnv(unittest.TestCase):
    def test_open_telemetry_disable_flag_is_always_kept_in_runtime_env(self) -> None:
        with patch.dict(os.environ, {"RAY_enable_open_telemetry": "0"}, clear=False):
            runtime_env = get_ppo_ray_runtime_env()

        self.assertEqual(runtime_env["env_vars"]["RAY_enable_open_telemetry"], "0")


if __name__ == "__main__":
    unittest.main()
