from dataclasses import dataclass
from typing import Optional, Any

import torch

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
        self.timeout[self.step] = torch.as_tensor(t.timeout, device=self.device).unsqueeze(-1)
        self.rewards[self.step] = torch.as_tensor(t.rewards, device=self.device).unsqueeze(-1)
        self.dones[self.step] = torch.as_tensor(t.dones, device=self.device).unsqueeze(-1)
        self.actions_logprob[self.step] = torch.as_tensor(t.actions_logprob, device=self.device).unsqueeze(-1)
        self.step += 1

    def clear(self):
        self.step = 0

    def compute_returns(self, gamma, lam, normalize_adv=True):
        advantage = 0

        for step in reversed(range(self.length)):
            mask = self.dones[step].logical_not().logical_or(self.timeout[step]) * gamma
            delta = self.rewards[step] + mask * self.next_values[step] - self.values[step]
            advantage = delta + mask * lam * advantage
            self.returns[step] = advantage + self.values[step]

        # Compute and normalize the advantages
        self.advantages = self.returns - self.values
        if normalize_adv:
            std, mean = torch.std_mean(self.advantages)
            self.advantages = (self.advantages - mean) / (std + 1e-8)

    def data_sampler(self, num_mini_batches, num_repetitions):
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
                yield self.Batch(*[item[indices] for item in collections])
