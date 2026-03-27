# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2022 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Core functions to implement PPO algorithms.
The function implemented in this file should be used by trainer with different distributed strategies to
implement PPO-like algorithms.
"""

import torch
import numpy as np
from collections import defaultdict

import verl.utils.torch_functional as verl_F

def compute_gae_advantage_return(
    token_level_rewards: torch.Tensor,
    values: torch.Tensor,
    response_mask: torch.Tensor,
    trajectory_uids: np.ndarray,
    step_indices: np.ndarray,
    gamma: torch.Tensor,
    lam: torch.Tensor,
):
    """Adapted from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape is (bs, response_length)
        values: `(torch.Tensor)`
            shape is (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape is (bs, response_length). [EOS] mask. The token after [EOS] have mask zero.
        gamma is `(float)`
            discounted factor used in RL
        lam: `(float)`
            lambda value when computing Generalized Advantage Estimation (https://arxiv.org/abs/1506.02438)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)

    """
    device = token_level_rewards.device

    with torch.no_grad():
        # Step-level reward: sum of token rewards inside the step.
        rewards = token_level_rewards.sum(dim=1)
        values = values.gather(1, response_mask.sum(1).unsqueeze(1) - 1).squeeze(1)

        # Map trajectories to contiguous ids for compact padding.
        # Use numpy's unique to handle both object and numeric types
        unique_traj_np, traj_inv_np = np.unique(trajectory_uids, return_inverse=True)
        num_traj = len(unique_traj_np)
        traj_inv = torch.as_tensor(traj_inv_np, dtype=torch.long, device=device)
        step_ids = torch.as_tensor(step_indices, device=device)
        max_step = int(step_ids.max().item()) + 1

        # reshape to (num_traj, max_step).
        # Use the same dtype as rewards and values to avoid type mismatch
        rewards_map = torch.zeros((num_traj, max_step), dtype=rewards.dtype, device=device)
        values_map = torch.zeros((num_traj, max_step), dtype=values.dtype, device=device)

        rewards_map[traj_inv, step_ids] = rewards
        values_map[traj_inv, step_ids] = values

        lastgaelam = 0
        advantages_reversed = []

        for t in reversed(range(max_step)):
            nextvalues = values_map[:, t + 1] if t < max_step - 1 else 0.0
            delta = rewards_map[:, t] + gamma * nextvalues - values_map[:, t]
            lastgaelam = delta + gamma * lam * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages_map = torch.stack(advantages_reversed[::-1], dim=1)

        # Map back to batch rows and then to token level.
        advantages = advantages_map[traj_inv, step_ids]
        returns = advantages + values

        advantages = advantages.unsqueeze(1) * response_mask
        returns = returns.unsqueeze(1) * response_mask
        advantages = verl_F.masked_whiten(advantages, response_mask)

    return advantages, returns

def compute_grpo_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    trajectory_uids: np.ndarray,
    step_indices: np.ndarray,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for GRPO, operating only on Outcome reward
    (with only one scalar reward for each response).

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape is (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape is (bs, response_length)
        index: `(np.ndarray)`
            index array for grouping
        epsilon: `(float)`
            small value to avoid division by zero
        norm_adv_by_std_in_grpo: `(bool)`
            whether to scale the GRPO advantage

    Note:
        If norm_adv_by_std_in_grpo is True, the advantage is scaled by the std, as in the original GRPO.
        If False, the advantage is not scaled, as in Dr.GRPO (https://arxiv.org/abs/2503.20783).

    Returns:
        advantages: `(torch.Tensor)`
            shape is (bs, response_length)
        Returns: `(torch.Tensor)`
            shape is (bs, response_length)
    """
    terminal_rewards = token_level_rewards.sum(dim=-1)
    group_scores = defaultdict(list)
    group_means = {}
    group_stds = {}
    terminal_score_by_trajectory = {}
    terminal_row_by_trajectory = {}

    with torch.no_grad():
        bsz = terminal_rewards.shape[0]
        trajectory_group_ids = {}

        for i in range(bsz):
            trajectory_uid = trajectory_uids[i]
            group_id = index[i]
            previous_group_id = trajectory_group_ids.get(trajectory_uid)
            if previous_group_id is None:
                trajectory_group_ids[trajectory_uid] = group_id
            elif previous_group_id != group_id:
                raise ValueError(
                    "GRPO grouping invariant violated: the same trajectory_uid "
                    f"{trajectory_uid!r} spans multiple group ids "
                    f"{previous_group_id!r} and {group_id!r}."
                )

            step_index = int(step_indices[i])
            previous_terminal = terminal_row_by_trajectory.get(trajectory_uid)
            if previous_terminal is None or step_index > previous_terminal[0]:
                terminal_row_by_trajectory[trajectory_uid] = (step_index, i)

        for trajectory_uid, (_, row_index) in terminal_row_by_trajectory.items():
            # Clone the scalar terminal reward so later in-place writes to the
            # output tensor cannot corrupt subsequent rows of the same
            # trajectory after batch reordering/balancing.
            terminal_score = terminal_rewards[row_index].clone()
            terminal_score_by_trajectory[trajectory_uid] = terminal_score
            group_scores[index[row_index]].append(terminal_score)

        for group_id, group_score_list in group_scores.items():
            if len(group_score_list) == 1:
                group_means[group_id] = group_score_list[0]
                group_stds[group_id] = torch.tensor(1.0, device=group_score_list[0].device)
            elif len(group_score_list) > 1:
                scores_tensor = torch.stack(group_score_list)
                group_means[group_id] = torch.mean(scores_tensor)
                group_stds[group_id] = torch.std(scores_tensor)
            else:
                raise ValueError(f"no terminal trajectory score in prompt index: {group_id}")

        advantages = torch.empty_like(terminal_rewards)
        for i in range(bsz):
            terminal_score = terminal_score_by_trajectory[trajectory_uids[i]]
            if norm_adv_by_std_in_grpo:
                advantages[i] = (terminal_score - group_means[index[i]]) / (group_stds[index[i]] + epsilon)
            else:
                advantages[i] = terminal_score - group_means[index[i]]
        advantages = advantages.unsqueeze(-1) * response_mask

    return advantages, advantages
