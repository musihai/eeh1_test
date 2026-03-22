import unittest

from omegaconf import OmegaConf

from arft.main_agent_ppo import _build_ray_init_kwargs


class TestBuildRayInitKwargs(unittest.TestCase):
    def test_defaults_disable_dashboard_and_inject_transfer_queue_env(self) -> None:
        config = OmegaConf.create(
            {
                "ray_kwargs": {
                    "ray_init": {
                        "num_cpus": 8,
                        "runtime_env": {
                            "env_vars": {
                                "EXISTING_FLAG": "1",
                            }
                        },
                    }
                },
                "transfer_queue": {"enable": True},
            }
        )

        ray_init_kwargs = _build_ray_init_kwargs(config)

        self.assertEqual(int(ray_init_kwargs.num_cpus), 8)
        self.assertFalse(bool(ray_init_kwargs.include_dashboard))
        self.assertEqual(ray_init_kwargs.runtime_env.env_vars["EXISTING_FLAG"], "1")
        self.assertEqual(ray_init_kwargs.runtime_env.env_vars["TRANSFER_QUEUE_ENABLE"], "1")

    def test_explicit_dashboard_override_is_respected(self) -> None:
        config = OmegaConf.create(
            {
                "ray_kwargs": {
                    "ray_init": {
                        "include_dashboard": True,
                    }
                },
                "transfer_queue": {"enable": False},
            }
        )

        ray_init_kwargs = _build_ray_init_kwargs(config)

        self.assertTrue(bool(ray_init_kwargs.include_dashboard))
        self.assertNotIn("TRANSFER_QUEUE_ENABLE", dict(ray_init_kwargs.runtime_env.get("env_vars", {})))


if __name__ == "__main__":
    unittest.main()
