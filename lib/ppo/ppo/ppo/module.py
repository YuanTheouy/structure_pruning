import numpy as np
import torch
from torch import nn

__all__ = ['Actor', 'Critic', 'MLP', 'Normalization', 'Gaussian']


class Actor(nn.Module):
    def __init__(
        self,
        architecture: 'MLP',
        distribution: 'Gaussian',
    ):
        super(Actor, self).__init__()

        self.architecture = architecture
        self.distribution = distribution
        self.action_mean = None

    def sample(self, obs):
        with torch.inference_mode():
            self.action_mean = self(obs)
        actions, logprob = self.distribution.sample(self.action_mean)
        return actions, logprob

    def calc_logprob_entropy(self, logits, actions):
        return self.distribution.calc_logprob_entropy(logits, actions)

    def forward(self, obs):
        return self.architecture(obs)

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

    def evaluate(self, obs):
        with torch.inference_mode():
            return self(obs)

    def forward(self, obs):
        return self.architecture(obs)

    @property
    def obs_shape(self):
        return self.architecture.input_shape


class MLP(nn.Module):
    def __init__(self, shape, activation_fn, input_size, output_size):
        super(MLP, self).__init__()
        self.activation_fn = activation_fn

        modules = [nn.Linear(input_size, shape[0]), self.activation_fn()]
        for idx in range(len(shape) - 1):
            modules.append(nn.Linear(shape[idx], shape[idx + 1]))
            modules.append(self.activation_fn())
        modules.append(nn.Linear(shape[-1], output_size))
        self.architecture = nn.Sequential(*modules)

        self.shape = shape
        self.input_shape = [input_size]
        self.output_shape = [output_size]

    def forward(self, x):
        return self.architecture(x)

    @staticmethod
    def init_weights(modules, scale=np.sqrt(2)):
        for m in modules:
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.orthogonal_(m.weight, gain=scale)
                # torch.nn.init.zeros_(m.bias)
            elif isinstance(m, torch.nn.LSTM):
                torch.nn.init.orthogonal_(m.weight_hh_l0, gain=scale)
                torch.nn.init.orthogonal_(m.weight_ih_l0, gain=scale)
                torch.nn.init.zeros_(m.bias_hh_l0)
                torch.nn.init.zeros_(m.bias_ih_l0)

    @staticmethod
    def rescale_layer(module, scale=0.01):
        module.weight.data.copy_(scale * module.weight.data)


class Normalization(nn.Module):
    def __init__(self, mean, var, eps=1e-8):
        super().__init__()
        self.mean = nn.Parameter(
            torch.as_tensor(mean), requires_grad=False
        )
        self.std = nn.Parameter(
            torch.sqrt(torch.as_tensor(var) + eps), requires_grad=False
        )

    def forward(self, x):
        return (x - self.mean) / self.std

    def denormalize(self, x):
        return x * self.std + self.mean


LOG_SQRT_2PI = np.log(np.sqrt(2 * np.pi))
LOG_2PI = np.log(2 * np.pi)
ENTROPY_BIAS = 0.5 + 0.5 * LOG_2PI


def calc_logprob(mean, std, sample):
    return -((sample - mean) ** 2) / (2 * std ** 2) - std.log() - LOG_SQRT_2PI


def calc_entropy(std):
    return ENTROPY_BIAS + std.log()


class Gaussian(nn.Module):
    # pure pytorch implemented
    # gpu version is faster than cpu multiprocessing version
    def __init__(self, dim, init_std):
        super().__init__()
        self.std = nn.Parameter(init_std * torch.ones(dim))

    def sample(self, logits):
        with torch.inference_mode():
            sample = torch.normal(logits, self.std)
            logprob = calc_logprob(logits, self.std, sample).sum(dim=-1)
            return sample, logprob

    def calc_logprob_entropy(self, logits, sample):
        logprob = calc_logprob(logits, self.std, sample).sum(dim=-1)
        entropy = calc_entropy(self.std).sum(dim=-1).repeat(logprob.shape)
        return logprob, entropy

    def set_std(self, std):
        self.std.data[:] = std

    def clamp_std(self, min=None, max=None):
        if min is not None or max is not None:
            self.std.data.clamp_(min, max)

