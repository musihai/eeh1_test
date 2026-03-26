import unittest

import numpy as np
import torch

from arft.core_algos import compute_grpo_outcome_advantage


class GrpoTerminalOutcomeTests(unittest.TestCase):
    def test_grpo_matches_paper_style_g8_terminal_reward_grouping(self) -> None:
        terminal_rewards = torch.tensor([0.10, 0.25, 0.30, 0.55, 0.60, 0.80, 0.95, 1.20], dtype=torch.float32)
        token_level_rewards = []
        response_mask = []
        index = []
        trajectory_uids = []
        step_indices = []

        for traj_idx, terminal_reward in enumerate(terminal_rewards.tolist()):
            for step_idx, reward in enumerate((7.0 + traj_idx, -3.0 - traj_idx, terminal_reward)):
                token_level_rewards.append([reward, 0.0])
                response_mask.append([1.0, 0.0])
                index.append("sample-a")
                trajectory_uids.append(f"traj-{traj_idx}")
                step_indices.append(step_idx)

        advantages, returns = compute_grpo_outcome_advantage(
            token_level_rewards=torch.tensor(token_level_rewards, dtype=torch.float32),
            response_mask=torch.tensor(response_mask, dtype=torch.float32),
            index=np.array(index, dtype=object),
            trajectory_uids=np.array(trajectory_uids, dtype=object),
            step_indices=np.array(step_indices, dtype=np.int32),
            norm_adv_by_std_in_grpo=True,
        )

        expected = (terminal_rewards - terminal_rewards.mean()) / terminal_rewards.std()
        for traj_idx, expected_advantage in enumerate(expected.tolist()):
            row_slice = slice(traj_idx * 3, traj_idx * 3 + 3)
            self.assertTrue(
                torch.allclose(
                    advantages[row_slice, 0],
                    torch.full((3,), float(expected_advantage), dtype=torch.float32),
                    atol=1e-6,
                )
            )
        self.assertTrue(torch.equal(advantages, returns))

    def test_grpo_uses_terminal_step_reward_for_each_trajectory(self) -> None:
        token_level_rewards = torch.tensor(
            [
                [5.0, 0.0],
                [1.0, 0.0],
                [-5.0, 0.0],
                [3.0, 0.0],
            ],
            dtype=torch.float32,
        )
        response_mask = torch.tensor(
            [
                [1.0, 0.0],
                [1.0, 0.0],
                [1.0, 0.0],
                [1.0, 0.0],
            ],
            dtype=torch.float32,
        )
        index = np.array(["sample-a", "sample-a", "sample-a", "sample-a"], dtype=object)
        trajectory_uids = np.array(["traj-1", "traj-1", "traj-2", "traj-2"], dtype=object)
        step_indices = np.array([0, 1, 0, 1], dtype=np.int32)

        advantages, returns = compute_grpo_outcome_advantage(
            token_level_rewards=token_level_rewards,
            response_mask=response_mask,
            index=index,
            trajectory_uids=trajectory_uids,
            step_indices=step_indices,
            norm_adv_by_std_in_grpo=True,
        )

        expected = 1.0 / torch.sqrt(torch.tensor(2.0))
        self.assertAlmostEqual(float(advantages[0, 0].item()), -float(expected.item()), places=6)
        self.assertAlmostEqual(float(advantages[1, 0].item()), -float(expected.item()), places=6)
        self.assertAlmostEqual(float(advantages[2, 0].item()), float(expected.item()), places=6)
        self.assertAlmostEqual(float(advantages[3, 0].item()), float(expected.item()), places=6)
        self.assertTrue(torch.equal(advantages, returns))

    def test_single_trajectory_group_has_zero_centered_advantage(self) -> None:
        token_level_rewards = torch.tensor([[0.0, 0.0], [2.5, 0.0]], dtype=torch.float32)
        response_mask = torch.tensor([[1.0, 0.0], [1.0, 0.0]], dtype=torch.float32)
        index = np.array(["sample-a", "sample-a"], dtype=object)
        trajectory_uids = np.array(["traj-1", "traj-1"], dtype=object)
        step_indices = np.array([0, 1], dtype=np.int32)

        advantages, returns = compute_grpo_outcome_advantage(
            token_level_rewards=token_level_rewards,
            response_mask=response_mask,
            index=index,
            trajectory_uids=trajectory_uids,
            step_indices=step_indices,
            norm_adv_by_std_in_grpo=True,
        )

        self.assertTrue(torch.equal(advantages, returns))
        self.assertTrue(torch.allclose(advantages, torch.zeros_like(advantages)))

    def test_grpo_ignores_nonterminal_reward_magnitude_when_forming_group_scores(self) -> None:
        terminal_rewards = torch.tensor([0.2, 0.5], dtype=torch.float32)
        baseline_rewards = torch.tensor(
            [
                [0.0, 0.0],
                [0.0, 0.0],
                [terminal_rewards[0].item(), 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [terminal_rewards[1].item(), 0.0],
            ],
            dtype=torch.float32,
        )
        perturbed_rewards = torch.tensor(
            [
                [100.0, 0.0],
                [-250.0, 0.0],
                [terminal_rewards[0].item(), 0.0],
                [-999.0, 0.0],
                [777.0, 0.0],
                [terminal_rewards[1].item(), 0.0],
            ],
            dtype=torch.float32,
        )
        response_mask = torch.tensor([[1.0, 0.0]] * 6, dtype=torch.float32)
        index = np.array(["sample-a"] * 6, dtype=object)
        trajectory_uids = np.array(["traj-1"] * 3 + ["traj-2"] * 3, dtype=object)
        step_indices = np.array([0, 1, 2, 0, 1, 2], dtype=np.int32)

        baseline_advantages, _ = compute_grpo_outcome_advantage(
            token_level_rewards=baseline_rewards,
            response_mask=response_mask,
            index=index,
            trajectory_uids=trajectory_uids,
            step_indices=step_indices,
            norm_adv_by_std_in_grpo=True,
        )
        perturbed_advantages, _ = compute_grpo_outcome_advantage(
            token_level_rewards=perturbed_rewards,
            response_mask=response_mask,
            index=index,
            trajectory_uids=trajectory_uids,
            step_indices=step_indices,
            norm_adv_by_std_in_grpo=True,
        )

        self.assertTrue(torch.allclose(baseline_advantages, perturbed_advantages, atol=1e-6))


if __name__ == "__main__":
    unittest.main()
