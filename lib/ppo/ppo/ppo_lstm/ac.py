import numpy as np
import torch
from torch import nn

from .module import *
from .recurrent import RMlp, is_recurrent

__all__ = ['Actor', 'Critic']


class Actor(nn.Module):
    def __init__(
        self,
        architecture,
        distribution: Gaussian,
    ):
        super(Actor, self).__init__()

        self.architecture = architecture
        self.distribution = distribution
        self.action_mean = None

        self.is_recurrent = is_recurrent(architecture)
        if self.is_recurrent:
            self.forward = self.forward_recurrent

    def sample(self, obs, *args, **kwargs):
        with torch.inference_mode():
            self.action_mean, hidden = self(obs, *args, **kwargs)
        actions, logprob = self.distribution.sample(self.action_mean)
        return actions, logprob, hidden

    def calc_logprob_entropy(self, logits, actions):
        return self.distribution.calc_logprob_entropy(logits, actions)

    def forward(self, obs, *args, **kwargs):
        return self.architecture(obs), None

    def forward_recurrent(self, obs, *args, **kwargs):
        return self.architecture(obs, *args, **kwargs)

    @property
    def obs_shape(self):
        return self.architecture.input_shape

    @property
    def action_shape(self):
        return self.architecture.output_shape

    @property
    def exploration(self) -> np.ndarray:
        return self.distribution.std.detach().cpu().numpy()

    def state_dict(self):
        return {
            'architecture': self.architecture.state_dict(),
            'distribution': self.distribution.state_dict(),
        }

    def load_state_dict(self, state_dict, strict: bool = True):
        if strict:
            self.architecture.load_state_dict(state_dict['architecture'], strict)
            self.distribution.load_state_dict(state_dict['distribution'], strict)
        else:
            raise ValueError('Strict mode is allowed only.')


class Critic(nn.Module):
    def __init__(self, architecture):
        super(Critic, self).__init__()
        self.architecture = architecture

        self.is_recurrent = is_recurrent(architecture)
        if self.is_recurrent:
            self.forward = self.forward_recurrent

    def evaluate(self, obs, *args, **kwargs):
        with torch.inference_mode():
            return self(obs, *args, **kwargs)

    def forward(self, obs, *args, **kwargs):
        return self.architecture(obs), None

    def forward_recurrent(self, obs, *args, **kwargs):
        return self.architecture(obs, *args, **kwargs)

    @property
    def obs_shape(self):
        return self.architecture.input_shape
