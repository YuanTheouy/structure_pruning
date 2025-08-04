from dataclasses import dataclass
from typing import Optional, Any

import torch

from .recurrent import *


class RolloutStorage:
    @dataclass
    class Transition:
        actor_obs: Any = None
        critic_obs: Any = None
        next_critic_obs: Any = None
        values: Any = None
        next_values: Any = None
        actions: Any = None
        mu: Any = None
        sigma: Any = None
        rewards: Any = None
        dones: Any = None
        timeout: Any = None
        actions_logprob: Any = None
        actor_hidden: Any = None
        critic_hidden: Any = None

    @dataclass
    class Batch:
        actor_obs: torch.Tensor = None
        critic_obs: torch.Tensor = None
        actions: torch.Tensor = None
        sigma: torch.Tensor = None
        mu: torch.Tensor = None
        values: torch.Tensor = None
        advantages: torch.Tensor = None
        returns: torch.Tensor = None
        actions_logprob: torch.Tensor = None
        # assigned every ppo update
        curr_mu: torch.Tensor = None
        curr_sigma: torch.Tensor = None
        curr_values: torch.Tensor = None
        # RNNs only
        actor_hidden: torch.Tensor = None
        critic_hidden: torch.Tensor = None
        trajectory_masks: torch.Tensor = None
        hidden_masks: torch.Tensor = None
        trajectory_lengths: torch.Tensor = None

    def __init__(
        self,
        num_envs,
        length,
        actor_obs_shape,
        critic_obs_shape,
        actions_shape,
        device,
    ):
        self.device = device
        self.length = length
        self.num_envs = num_envs

        # Core
        self.critic_obs = torch.zeros([length, num_envs, *critic_obs_shape], device=self.device)
        self.actor_obs = torch.zeros([length, num_envs, *actor_obs_shape], device=self.device)
        self.next_critic_obs = torch.zeros([length, num_envs, *critic_obs_shape], device=self.device)
        self.rewards = torch.zeros([length, num_envs, 1], device=self.device)
        self.actions = torch.zeros([length, num_envs, *actions_shape], device=self.device)
        self.dones = torch.zeros([length, num_envs, 1], dtype=torch.bool, device=self.device)
        self.timeout = torch.zeros([length, num_envs, 1], dtype=torch.bool, device=self.device)

        # For PPO
        self.actions_logprob = torch.zeros([length, num_envs, 1], device=self.device)
        self.values = torch.zeros([length, num_envs, 1], device=self.device)
        self.next_values = torch.zeros([length, num_envs, 1], device=self.device)
        self.returns = torch.zeros([length, num_envs, 1], device=self.device)
        self.advantages = torch.zeros([length, num_envs, 1], device=self.device)
        self.mu = torch.zeros([length, num_envs, *actions_shape], device=self.device)
        self.sigma = torch.zeros([length, num_envs, *actions_shape], device=self.device)

        # rnn
        self.actor_hidden: Optional[list[torch.Tensor]] = None
        self.critic_hidden: Optional[list[torch.Tensor]] = None

        self.step = 0

    def add_transition(self, t: Transition):
        if self.step >= self.length:
            raise AssertionError("Rollout buffer overflow")
        self.critic_obs[self.step] = torch.as_tensor(t.critic_obs, device=self.device)
        self.actor_obs[self.step] = torch.as_tensor(t.actor_obs, device=self.device)
        self.next_critic_obs[self.step] = torch.as_tensor(t.next_critic_obs, device=self.device)
        self.values[self.step] = torch.as_tensor(t.values, device=self.device)
        self.next_values[self.step] = torch.as_tensor(t.next_values, device=self.device)
        self.actions[self.step] = torch.as_tensor(t.actions, device=self.device)
        self.mu[self.step] = torch.as_tensor(t.mu, device=self.device)
        self.sigma[self.step] = torch.as_tensor(t.sigma, device=self.device)
        self.timeout[self.step] = torch.as_tensor(t.timeout, device=self.device).unsqueeze_(-1)
        self.rewards[self.step] = torch.as_tensor(t.rewards, device=self.device).unsqueeze_(-1)
        self.dones[self.step] = torch.as_tensor(t.dones, device=self.device).unsqueeze_(-1)
        self.actions_logprob[self.step] = torch.as_tensor(t.actions_logprob, device=self.device).unsqueeze_(-1)
        if t.actor_hidden is not None:
            if self.actor_hidden is None:
                self.actor_hidden = self._init_hidden_buffer_like(t.actor_hidden)
            else:
                self._save_hidden_states(t.actor_hidden, self.actor_hidden)
        if t.critic_hidden is not None:
            if self.critic_hidden is None:
                self.critic_hidden = self._init_hidden_buffer_like(t.critic_hidden)
            else:
                self._save_hidden_states(t.critic_hidden, self.critic_hidden)
        self.step += 1

    def _save_hidden_states(self, hidden, buffer):
        if isinstance(hidden, torch.Tensor):
            buffer[0][self.step] = hidden
        else:
            for i in range(len(hidden)):
                buffer[i][self.step] = hidden[i]

    def _init_hidden_buffer_like(self, example):
        if isinstance(example, torch.Tensor):
            return [
                torch.zeros(self.length, *example.shape, device=self.device)
            ]
        return [
            torch.zeros(self.length, *hidden.shape, device=self.device)
            for hidden in example
        ]

    def clear(self):
        self.step = 0

    def compute_returns(self, gamma, lamda, normalize_adv=True):
        advantage = 0

        for step in reversed(range(self.length)):
            mask = self.dones[step].logical_not().logical_or(self.timeout[step]) * gamma
            delta = self.rewards[step] + mask * self.next_values[step] - self.values[step]
            advantage = delta + mask * lamda * advantage
            self.returns[step] = advantage + self.values[step]

        # Compute and normalize the advantages
        self.advantages = self.returns - self.values
        if normalize_adv:
            std, mean = torch.std_mean(self.advantages)
            self.advantages = (self.advantages - mean) / (std + 1e-8)

    @property
    def data_sampler(self):
        if self.actor_hidden is None and self.critic_hidden is None:
            return self._data_sampler
        else:
            return self._recurrent_data_sampler

    def _data_sampler(self, num_mini_batches, num_repetitions, batch: Batch = None):
        batch_size = self.num_envs * self.length
        mini_batch_size = batch_size // num_mini_batches
        collections = (
            self.actor_obs.flatten(0, 1),
            self.critic_obs.flatten(0, 1),
            self.actions.flatten(0, 1),
            self.sigma.flatten(0, 1),
            self.mu.flatten(0, 1),
            self.values.flatten(0, 1),
            self.advantages.flatten(0, 1),
            self.returns.flatten(0, 1),
            self.actions_logprob.flatten(0, 1),
        )

        for _ in range(num_repetitions):
            rand_indices = torch.randperm(num_mini_batches * mini_batch_size, device=self.device)
            for i in range(num_mini_batches):
                indices = rand_indices[i * mini_batch_size: (i + 1) * mini_batch_size]
                if batch is None:
                    batch = self.Batch(*[item[indices] for item in collections])
                else:
                    batch.__init__(*[item[indices] for item in collections])
                yield batch

    def _recurrent_data_sampler(self, num_mini_batches, num_repetitions):
        mini_batch_size = self.num_envs // num_mini_batches
        batches = []

        for i in range(num_mini_batches):
            start = i * mini_batch_size
            stop = (i + 1) * mini_batch_size
            batches.append(
                self._recurrent_minibatch_slicer(
                    slice(start, stop)
                )
            )

        yield from batches * num_repetitions

    def _recurrent_data_sampler_with_shuffle(self, num_mini_batches, num_repetitions):
        mini_batch_size = self.num_envs // num_mini_batches
        for ep in range(num_repetitions):
            rand_indices = torch.randperm(
                num_mini_batches * mini_batch_size, device=self.device
            ).reshape(num_mini_batches, -1)
            for i in range(num_mini_batches):
                yield self._recurrent_minibatch_slicer(rand_indices[i])

    def _get_hidden_mask(self, dones):
        dones = dones.squeeze(-1)
        last_was_done = torch.zeros_like(dones, dtype=torch.bool)
        last_was_done[1:] = dones[:-1]
        last_was_done[0] = True
        return last_was_done.permute(1, 0)

    def _recurrent_minibatch_slicer(self, indices, batch=None):
        if batch is None:
            batch = self.Batch()
        # reshape to [num_envs, time, num layers, hidden dim]
        # (original shape: [time, num_layers, num_envs, hidden_dim])
        # then take only time steps after dones (flattens num envs and time dimensions),
        # take a batch of trajectories and finally reshape back to [num_layers, batch, hidden_dim]
        batch.hidden_masks = self._get_hidden_mask(self.dones[:, indices])
        batch.trajectory_lengths = get_trajectory_lengths(self.dones[:, indices])
        batch.actor_obs, batch.actor_hidden = self._slice_hidden_relevant(
            self.actor_obs, self.actor_hidden, indices,
            batch.hidden_masks, batch.trajectory_lengths
        )
        batch.critic_obs, batch.critic_hidden = self._slice_hidden_relevant(
            self.critic_obs, self.critic_hidden, indices,
            batch.hidden_masks, batch.trajectory_lengths
        )

        batch.actions = self.actions[:, indices]
        batch.sigma = self.sigma[:, indices]
        batch.mu = self.mu[:, indices]
        batch.values = self.values[:, indices]
        batch.advantages = self.advantages[:, indices]
        batch.returns = self.returns[:, indices]
        batch.actions_logprob = self.actions_logprob[:, indices]
        batch.trajectory_masks = get_trajectory_mask(batch.trajectory_lengths)
        return batch

    @staticmethod
    def _slice_hidden_relevant(obs, hidden, indices, mask, traj_len):
        # TODO: Simplify pipeline by saving trajectory lengths and init hidden
        if hidden is None:
            return obs[:, indices], None

        obs_batch = split_trajectory(obs[:, indices], traj_len)
        hidden_batch = [
            hid[:, :, indices].permute(2, 0, 1, 3)[mask].transpose(1, 0).contiguous()
            for hid in hidden
        ]
        if len(hidden_batch) == 1:
            hidden_batch = hidden_batch[0]
        return obs_batch, hidden_batch
