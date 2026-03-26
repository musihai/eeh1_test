import unittest

from omegaconf import OmegaConf

from verl.workers.engine_workers import _materialize_ref_actor_fields


class TestEngineWorkersRefConfig(unittest.TestCase):
    def test_materialize_ref_actor_fields_preserves_existing_actor_fields_when_logprob_keys_missing(self) -> None:
        actor_cfg = OmegaConf.create({"ppo_mini_batch_size": 3})
        ref_cfg = OmegaConf.create(
            {
                "ppo_micro_batch_size": None,
                "ppo_micro_batch_size_per_gpu": 2,
                "ppo_max_token_len_per_gpu": 12288,
                "use_dynamic_bsz": False,
            }
        )

        _materialize_ref_actor_fields(ref_cfg, actor_cfg)

        self.assertEqual(ref_cfg.ppo_mini_batch_size, 3)
        self.assertEqual(ref_cfg.ppo_micro_batch_size_per_gpu, 2)
        self.assertEqual(ref_cfg.ppo_max_token_len_per_gpu, 12288)
        self.assertFalse(ref_cfg.use_dynamic_bsz)

    def test_materialize_ref_actor_fields_maps_logprob_keys_to_actor_fields(self) -> None:
        actor_cfg = OmegaConf.create({"ppo_mini_batch_size": 3})
        ref_cfg = OmegaConf.create(
            {
                "log_prob_micro_batch_size": None,
                "log_prob_micro_batch_size_per_gpu": 1,
                "log_prob_use_dynamic_bsz": False,
                "log_prob_max_token_len_per_gpu": 12288,
                "ppo_micro_batch_size": None,
                "ppo_micro_batch_size_per_gpu": None,
                "ppo_max_token_len_per_gpu": None,
                "use_dynamic_bsz": None,
            }
        )

        _materialize_ref_actor_fields(ref_cfg, actor_cfg)

        self.assertEqual(ref_cfg.ppo_mini_batch_size, 3)
        self.assertEqual(ref_cfg.ppo_micro_batch_size_per_gpu, 1)
        self.assertEqual(ref_cfg.ppo_max_token_len_per_gpu, 12288)
        self.assertFalse(ref_cfg.use_dynamic_bsz)
        self.assertNotIn("log_prob_micro_batch_size_per_gpu", ref_cfg)
        self.assertNotIn("log_prob_max_token_len_per_gpu", ref_cfg)


if __name__ == "__main__":
    unittest.main()
