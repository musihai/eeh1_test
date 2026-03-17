import asyncio
import unittest

import numpy as np
import torch
from omegaconf import OmegaConf

from arft.ray_agent_trainer import evaluate_validation_reward_manager
from verl import DataProto
from verl.experimental.reward_loop.reward_manager.naive import NaiveRewardManager


class _AsyncRewardManager:
    async def run_single(self, data: DataProto):
        uid = str(data.non_tensor_batch["uid"][0])
        if uid.endswith("0"):
            return {"reward_score": 0.25, "reward_extra_info": {"acc": 0.25, "tag": "a"}}
        return {"reward_score": 0.5, "reward_extra_info": {"acc": 0.5}}


class _Tokenizer:
    def decode(self, ids, skip_special_tokens=True):
        return "2016-01-01 00:00:00 1.0000"


class TestValidationRewardManager(unittest.TestCase):
    def _build_batch(self) -> DataProto:
        return DataProto.from_dict(
            tensors={
                "prompts": torch.tensor([[11, 12], [21, 22]], dtype=torch.long),
                "responses": torch.tensor([[31, 32, 33], [41, 42, 0]], dtype=torch.long),
                "attention_mask": torch.tensor(
                    [
                        [1, 1, 1, 1, 1],
                        [1, 1, 1, 1, 0],
                    ],
                    dtype=torch.long,
                ),
            },
            non_tensors={
                "uid": np.array(["sample-0", "sample-1"], dtype=object),
                "reward_model": np.array(
                    [
                        {"ground_truth": "2016-01-01 00:00:00 1.0", "style": "rule"},
                        {"ground_truth": "2016-01-01 00:00:00 2.0", "style": "rule"},
                    ],
                    dtype=object,
                ),
                "data_source": np.array(["ETTh1", "ETTh1"], dtype=object),
            },
        )

    def test_evaluate_validation_reward_manager_supports_async_reward_loop_manager(self) -> None:
        batch = self._build_batch()
        result = evaluate_validation_reward_manager(_AsyncRewardManager(), batch)
        reward_tensor = result["reward_tensor"]
        reward_extra_info = result["reward_extra_info"]

        self.assertEqual(tuple(reward_tensor.shape), (2, 3))
        self.assertAlmostEqual(float(reward_tensor[0, 2].item()), 0.25, places=6)
        self.assertAlmostEqual(float(reward_tensor[1, 1].item()), 0.5, places=6)
        self.assertEqual(reward_extra_info["acc"], [0.25, 0.5])
        self.assertEqual(reward_extra_info["tag"], ["a", None])

    def test_evaluate_validation_reward_manager_handles_reward_manager_with_stale_loop(self) -> None:
        batch = self._build_batch()
        manager = NaiveRewardManager(
            config=OmegaConf.create({}),
            tokenizer=_Tokenizer(),
            compute_score=lambda **kwargs: 0.3,
        )
        stale_loop = asyncio.new_event_loop()
        manager.loop = stale_loop

        try:
            result = evaluate_validation_reward_manager(manager, batch)
            reward_tensor = result["reward_tensor"]

            self.assertAlmostEqual(float(reward_tensor[0, 2].item()), 0.3, places=6)
            self.assertAlmostEqual(float(reward_tensor[1, 1].item()), 0.3, places=6)
        finally:
            stale_loop.close()


if __name__ == "__main__":
    unittest.main()
