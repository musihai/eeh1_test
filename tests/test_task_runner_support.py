import unittest

from omegaconf import OmegaConf

from arft.task_runner_support import (
    build_actor_rollout_spec,
    build_critic_worker_spec,
    build_resource_pool_spec,
    build_reward_model_worker_spec,
    resolve_legacy_worker_impl,
    should_register_ref_policy,
)


def _make_config():
    return OmegaConf.create(
        {
            "trainer": {
                "use_legacy_worker_impl": "auto",
                "n_gpus_per_node": 4,
                "nnodes": 2,
            },
            "algorithm": {
                "use_kl_in_reward": False,
            },
            "actor_rollout_ref": {
                "actor": {
                    "strategy": "fsdp",
                    "use_kl_loss": False,
                },
                "rollout": {
                    "mode": "async",
                },
            },
            "critic": {
                "strategy": "fsdp",
            },
            "reward_model": {
                "enable": True,
                "enable_resource_pool": True,
                "strategy": "fsdp",
                "n_gpus_per_node": 2,
                "nnodes": 1,
            },
        }
    )


class TestTaskRunnerSupport(unittest.TestCase):
    def test_actor_rollout_spec_legacy_async_fsdp(self) -> None:
        config = _make_config()

        spec = build_actor_rollout_spec(config)

        self.assertEqual(spec.worker.module_path, "verl.workers.fsdp_workers")
        self.assertEqual(spec.worker.attribute_name, "AsyncActorRolloutRefWorker")
        self.assertEqual(spec.ray_worker_group.module_path, "verl.single_controller.ray")
        self.assertEqual(spec.ray_worker_group.attribute_name, "RayWorkerGroup")
        self.assertEqual(spec.role_name, "ActorRollout")

    def test_actor_rollout_spec_new_engine_with_kl_uses_fused_role(self) -> None:
        config = _make_config()
        config.trainer.use_legacy_worker_impl = "disable"
        config.algorithm.use_kl_in_reward = True

        spec = build_actor_rollout_spec(config)

        self.assertEqual(spec.worker.module_path, "verl.workers.engine_workers")
        self.assertEqual(spec.worker.attribute_name, "ActorRolloutRefWorker")
        self.assertEqual(spec.role_name, "ActorRolloutRef")

    def test_actor_rollout_sync_mode_raises(self) -> None:
        config = _make_config()
        config.actor_rollout_ref.rollout.mode = "sync"

        with self.assertRaises(ValueError):
            build_actor_rollout_spec(config)

    def test_invalid_legacy_worker_mode_raises(self) -> None:
        config = _make_config()
        config.trainer.use_legacy_worker_impl = "bad-mode"

        with self.assertRaises(ValueError):
            resolve_legacy_worker_impl(config)

    def test_critic_spec_disable_uses_engine_worker(self) -> None:
        config = _make_config()
        config.trainer.use_legacy_worker_impl = "disable"

        spec = build_critic_worker_spec(config)

        self.assertEqual(spec.worker.module_path, "verl.workers.engine_workers")
        self.assertEqual(spec.worker.attribute_name, "CriticWorker")

    def test_reward_model_spec_selects_reward_pool(self) -> None:
        config = _make_config()

        spec = build_reward_model_worker_spec(config)

        self.assertIsNotNone(spec)
        self.assertEqual(spec.worker.module_path, "verl.workers.fsdp_workers")
        self.assertEqual(spec.worker.attribute_name, "RewardModelWorker")
        self.assertEqual(spec.pool_name, "reward_pool")

    def test_reward_model_disabled_returns_none(self) -> None:
        config = _make_config()
        config.reward_model.enable = False

        self.assertIsNone(build_reward_model_worker_spec(config))

    def test_build_resource_pool_spec_includes_reward_pool(self) -> None:
        config = _make_config()

        spec = build_resource_pool_spec(config)

        self.assertEqual(spec["global_pool"], [4, 4])
        self.assertEqual(spec["reward_pool"], [2])

    def test_build_resource_pool_spec_validates_reward_pool_shape(self) -> None:
        config = _make_config()
        config.reward_model.n_gpus_per_node = 0

        with self.assertRaises(ValueError):
            build_resource_pool_spec(config)

    def test_should_register_ref_policy_respects_legacy_mode(self) -> None:
        config = _make_config()
        config.actor_rollout_ref.actor.use_kl_loss = True

        self.assertTrue(should_register_ref_policy(config))

        config.trainer.use_legacy_worker_impl = "disable"
        self.assertFalse(should_register_ref_policy(config))


if __name__ == "__main__":
    unittest.main()
