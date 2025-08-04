import itertools
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim

from .ac import Actor, Critic
from .recurrent import split_trajectory, unpad_trajectory
from .storage import RolloutStorage

GIGA = 2 ** 30


class PPO:
    def __init__(
        self,
        actor: Actor,
        critic: Critic,
        num_envs,
        num_collects,
        num_learning_epochs,
        num_mini_batches,
        clip_param=0.2,
        gamma=0.998,
        lamda=0.95,
        value_loss_coef=0.5,
        entropy_coef=0.0,
        learning_rate=5e-4,
        max_grad_norm=0.5,
        clip_value_loss=True,
        init_std=None,
        min_std=None,
        max_std=None,
        device='cpu',
        **kwargs,
    ):

        if kwargs:
            print('PPO: Ignored kwargs: ', *[f'  - {k}: {v}' for k, v in kwargs.items()], sep='\n')
        # PPO components
        self.actor = actor
        self.critic = critic
        self.num_envs = num_envs
        self.num_collects = num_collects
        self.optimizer = optim.Adam(
            [*self.actor.parameters(), *self.critic.parameters()],
            lr=learning_rate
        )
        self.device = device

        # PPO parameters
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lamda = lamda
        self.max_grad_norm = max_grad_norm
        self.clip_value_loss = clip_value_loss
        self.min_std = min_std
        self.max_std = max_std
        if init_std is not None:
            self.actor.distribution.set_std(init_std)

        # ADAM
        self.learning_rate = learning_rate

        # storage
        self.storage = None
        self.transition: Optional[RolloutStorage.Transition] = None
        self.last_actor_hidden = None
        self.last_critic_hidden = None
        self.init_storage()

    def init_storage(self):
        self.storage = RolloutStorage(
            self.num_envs, self.num_collects,
            self.actor.obs_shape, self.critic.obs_shape,
            self.actor.action_shape, self.device
        )
        self.transition = self.storage.Transition()

    def act(self, actor_obs, critic_obs=None):
        actor_obs = self._to_torch(actor_obs)
        critic_obs = actor_obs if critic_obs is None else self._to_torch(critic_obs)
        self.transition.actor_obs = actor_obs
        self.transition.critic_obs = critic_obs
        self.transition.actor_hidden = self.last_actor_hidden
        self.transition.critic_hidden = self.last_critic_hidden
        actions, self.transition.actions_logprob, actor_hidden = self.actor.sample(
            actor_obs, self.last_actor_hidden)
        self.transition.actions = actions
        self.last_actor_hidden = actor_hidden

        self.transition.values, critic_hidden = self.critic.evaluate(
            critic_obs, self.last_critic_hidden)
        self.last_critic_hidden = critic_hidden

        self.transition.mu = self.actor.action_mean
        self.transition.sigma = self.actor.distribution.std
        return actions.cpu().numpy()

    def step(self, next_value_obs, rews, dones, timeout):
        self.transition.next_critic_obs = self._to_torch(next_value_obs)
        self.transition.rewards = rews
        self.transition.dones = dones
        self.transition.timeout = timeout
        # duplicated inference for convenience and performance
        self.transition.next_values, _ = self.critic.evaluate(
            self.transition.next_critic_obs, self.last_critic_hidden  # updated
        )
        self.storage.add_transition(self.transition)
        self.transition.__init__()
        self.clear_hidden_states(dones)

    def clear_hidden_states(self, dones):
        if self.last_actor_hidden is not None:
            if isinstance(self.last_actor_hidden, torch.Tensor):  # gru
                self.last_actor_hidden[..., dones, :] = 0.0
            else:
                for hidden in self.last_actor_hidden:  # lstm
                    hidden[..., dones, :] = 0.0

        if self.last_critic_hidden is not None:
            if isinstance(self.last_critic_hidden, torch.Tensor):  # gru
                self.last_critic_hidden[..., dones, :] = 0.0
            else:
                for hidden in self.last_critic_hidden:  # lstm
                    hidden[..., dones, :] = 0.0

    def _to_torch(self, tensor, dtype=None):
        return torch.as_tensor(tensor, dtype=dtype, device=self.device)

    def update(self, warmup=False):
        # Learning step
        self.storage.compute_returns(self.gamma, self.lamda)
        summary = self._warmup() if warmup else self._train_step()
        if torch.cuda.memory_reserved() > 4 * GIGA:
            # TODO: FIGURE OUT WHY TORCH IS CONSTANTLY
            #  RESERVING MEMORY DURING RNN TRAINING
            torch.cuda.empty_cache()
        self.storage.clear()
        summary.update({
            'PPO/mean_noise_std': self.actor.exploration.mean(),
            'PPO/learning_rate': self.learning_rate,
            'PPO/entropy': self.entropy_coef,
        })
        return summary

    def _warmup(self):
        mean_value_loss = torch.zeros(1, device=self.device)
        for batch in self.storage.data_sampler(self.num_mini_batches, self.num_learning_epochs):
            curr_value, _ = self.critic(batch.critic_obs, batch.critic_hidden)
            if curr_value.shape[:-1] != batch.actions.shape[:-1]:
                curr_value = unpad_trajectory(curr_value, batch.trajectory_masks, self.num_collects)

            # Value function loss
            if self.clip_value_loss:
                value_clipped = batch.values + (curr_value - batch.values).clamp(-self.clip_param, self.clip_param)
                value_losses = (curr_value - batch.returns).pow(2)
                value_losses_clipped = (value_clipped - batch.returns).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (batch.returns - curr_value).pow(2).mean()

            loss = self.value_loss_coef * value_loss
            # TODO: ADD EXTRA LOSS TO WARMUP

            # Gradient step
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                itertools.chain(
                    self.actor.parameters(),
                    self.critic.parameters(),
                ), self.max_grad_norm
            )
            self.optimizer.step()

            mean_value_loss += value_loss.detach()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        return {
            'PPO/value_function': mean_value_loss.item(),
            'PPO/surrogate': 0.,
            'PPO/ratio': 0.,
        }

    def _train_step(self):
        mean_value_loss = torch.zeros(1, device=self.device)
        mean_surrogate_loss = torch.zeros(1, device=self.device)
        mean_ratio = torch.zeros(1, device=self.device)
        mean_extra_losses = None
        for batch in self.storage.data_sampler(self.num_mini_batches, self.num_learning_epochs):
            batch.curr_mu, _ = self.actor(batch.actor_obs, batch.actor_hidden)
            if batch.curr_mu.shape[:-1] != batch.actions.shape[:-1]:
                batch.curr_mu = unpad_trajectory(batch.curr_mu, batch.trajectory_masks, self.num_collects)
            curr_actions_logprob, curr_entropy = self.actor.calc_logprob_entropy(batch.curr_mu, batch.actions)
            batch.curr_values, _ = self.critic(batch.critic_obs, batch.critic_hidden)
            if batch.curr_values.shape[:-1] != batch.actions.shape[:-1]:
                batch.curr_values = unpad_trajectory(batch.curr_values, batch.trajectory_masks, self.num_collects)
            batch.curr_sigma = self.actor.distribution.std

            # Surrogate loss
            ratio = torch.exp(curr_actions_logprob - batch.actions_logprob.squeeze())
            surrogate = -torch.squeeze(batch.advantages) * ratio
            surrogate_clipped = -batch.advantages.squeeze() * ratio.clamp(1.0 - self.clip_param,
                                                                          1.0 + self.clip_param)
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            # Value function loss
            if self.clip_value_loss:
                value_clipped = batch.values + (batch.curr_values - batch.values).clamp(-self.clip_param,
                                                                                        self.clip_param)
                value_losses = (batch.curr_values - batch.returns).pow(2)
                value_losses_clipped = (value_clipped - batch.returns).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (batch.returns - batch.curr_values).pow(2).mean()

            loss = (
                surrogate_loss
                + self.value_loss_coef * value_loss
                - self.entropy_coef * curr_entropy.mean()
            )

            # Gradient step
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                itertools.chain(
                    self.actor.parameters(),
                    self.critic.parameters(),
                ), self.max_grad_norm
            )
            self.optimizer.step()

            with torch.inference_mode():
                mean_value_loss += value_loss
                mean_surrogate_loss += surrogate_loss
                mean_ratio += torch.abs(ratio - 1.0).mean()

        self.actor.distribution.clamp_std(min=self.min_std, max=self.max_std)

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_ratio /= num_updates

        data = {
            'PPO/value_function': mean_value_loss.item(),
            'PPO/surrogate': mean_surrogate_loss.item(),
            'PPO/ratio': mean_ratio.item(),
        }
        if mean_extra_losses is not None:
            mean_extra_losses = (mean_extra_losses / num_updates).cpu().numpy()
            data.update(self._get_extra_loss_info(mean_extra_losses))

        return data

    def load_state_dict(self, state_dict):
        version = state_dict.get('version', 'v0')
        if version == 'v0':
            self.actor.architecture.load_state_dict(state_dict['actor_architecture_state_dict'])
            self.actor.distribution.load_state_dict(state_dict['actor_distribution_state_dict'])
            self.critic.architecture.load_state_dict(state_dict['critic_architecture_state_dict'])
            self.optimizer.load_state_dict(state_dict['optimizer_state_dict'])
        elif version == 'v1':
            self.actor.load_state_dict(state_dict['actor'])
            self.critic.load_state_dict(state_dict['critic'])
            self.optimizer.load_state_dict(state_dict['optimizer'])

    def state_dict(self):
        return {
            'version': 'v1',
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }
    def save_model(self, output):
        torch.save(self.state_dict(),"{}/rl.pth.tar".format(output))

    @classmethod
    def restore_policy(cls, policy, state_dict):
        version = state_dict.get('version', 'v0')
        if version == 'v0':
            policy_state_dict = state_dict['actor_architecture_state_dict']
        elif version == 'v1':
            policy_state_dict = state_dict['actor']['architecture']
            for k in tuple(policy_state_dict):
                if k.startswith('memory.'):
                    policy_state_dict[k.removeprefix('memory.')] = policy_state_dict.pop(k)
        policy.load_state_dict(policy_state_dict)


@torch.jit.script
def kl_divergence(mu1, sigma1, mu2, sigma2):
    kl = torch.sum(
        torch.log(sigma2 / sigma1 + 1.e-5) + (
            torch.square(sigma1) + torch.square(mu1 - mu2)) / (
            2.0 * torch.square(sigma2)) - 0.5, dim=-1)
    return kl.mean()
