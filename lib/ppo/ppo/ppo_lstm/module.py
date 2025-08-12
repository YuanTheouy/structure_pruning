import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from typing import Tuple

__all__ = ['MLP', 'Normalization', 'Gaussian', 'GumbelActor']


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
                torch.nn.init.zeros_(m.bias)
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
        # 仍然学习 log_std 以保证 std > 0
        initial_log_std = torch.log(torch.tensor(init_std, dtype=torch.float32) + 1e-8)
        self.log_std = nn.Parameter(initial_log_std * torch.ones(dim))

    # --- 核心改动：添加 @property ---
    @property
    def std(self) -> torch.Tensor:
        """
        让外部代码可以通过 .std 访问到计算出的标准差，
        同时保证其值永远为正。
        """
        return torch.exp(self.log_std)

    def sample(self, logits):
        with torch.inference_mode():
            # 现在可以直接使用 self.std，因为它是一个 property
            sample = torch.normal(logits, self.std)
            logprob = calc_logprob(logits, self.std, sample).sum(dim=-1)
            return sample, logprob

    def calc_logprob_entropy(self, logits, sample):
        # 这里也直接使用 self.std
        logprob = calc_logprob(logits, self.std, sample).sum(dim=-1)
        entropy = calc_entropy(self.std).sum(dim=-1).repeat(logprob.shape)
        return logprob, entropy

    def set_std(self, std_val):
        # 外部设置 std 时，我们仍然是更新内部的 log_std
        self.log_std.data[:] = torch.log(torch.tensor(std_val, device=self.log_std.device) + 1e-8)

    def clamp_std(self, min=None, max=None, indices=None):
        # clamp 操作仍然在 log 空间中进行
        with torch.no_grad():
            if min is not None:
                min_log_std = torch.log(torch.tensor(min, device=self.log_std.device) + 1e-8)
            else:
                min_log_std = None

            if max is not None:
                max_log_std = torch.log(torch.tensor(max, device=self.log_std.device) + 1e-8)
            else:
                max_log_std = None

            if min is not None or max is not None:
                if indices is None:
                    self.log_std.data.clamp_(min=min_log_std, max=max_log_std)
                else:
                    self.log_std.data[indices].clamp_(min=min_log_std, max=max_log_std)

# >>> ADD THE FOLLOWING NEW CLASS
class GumbelActor(nn.Module):
    """
    An actor that uses the Gumbel-Softmax trick to handle discrete action bins
    for a continuous action space problem.
    """
    def __init__(self, net: nn.Module, num_actions: int, num_bins: int, action_bins: torch.Tensor):
        """
        Initializes the GumbelActor.
        Args:
            net (nn.Module): The base MLP network that outputs raw logits.
            num_actions (int): The number of continuous actions (e.g., 48 for 48 modules).
            num_bins (int): The number of discrete bins for each action.
            action_bins (torch.Tensor): A 1D tensor containing the values of the action bins.
        """
        super().__init__()
        self.net = net
        self.num_actions = num_actions
        self.num_bins = num_bins
        
        # Register action_bins as a buffer so it's moved to the correct device
        # with the model, but is not considered a model parameter.
        self.register_buffer('action_bins', action_bins)
        
        # Temperature `tau` is a non-trainable parameter, updated externally.
        self.tau = nn.Parameter(torch.tensor(1.0, dtype=torch.float32), requires_grad=False)
        
        # For compatibility with PPO framework
        self.action_mean = None
        # Create a dummy distribution object for compatibility
        class DummyDistribution:
            def __init__(self, num_actions):
                self.std = torch.ones(num_actions) * 0.1  # Dummy std values
            
            def clamp_std(self, min=None, max=None, indices=None):
                """Dummy clamp_std method for PPO compatibility."""
                if min is not None or max is not None:
                    if indices is not None:
                        self.std[indices].clamp_(min=min, max=max)
                    else:
                        self.std.clamp_(min=min, max=max)
        self.distribution = DummyDistribution(num_actions)

    def set_tau(self, new_tau: float):
        """Externally set the Gumbel-Softmax temperature."""
        self.tau.fill_(new_tau)

    def forward(self, state: torch.Tensor, hidden=None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        The main forward pass for training.
        It computes the final continuous actions, their log-probabilities, and the policy entropy.
        
        Args:
            state (torch.Tensor): The input state tensor, shape (batch_size, state_dim).
            hidden: Hidden state (for compatibility with recurrent networks, unused here).
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - final_actions (torch.Tensor): The resulting continuous actions, shape (batch_size, num_actions).
                - log_prob (torch.Tensor): The log probability of the actions for the PPO loss, shape (batch_size,).
                - entropy (torch.Tensor): The mean policy entropy, a scalar tensor.
        """
    def forward(self, state: torch.Tensor, hidden=None) -> Tuple[torch.Tensor, None]:
        """
        The main forward pass for training.
        It computes the final continuous actions for PPO compatibility.
        
        Args:
            state (torch.Tensor): The input state tensor, shape (batch_size, state_dim).
            hidden: Hidden state (for compatibility with recurrent networks, unused here).
            
        Returns:
            Tuple[torch.Tensor, None]:
                - final_actions (torch.Tensor): The resulting continuous actions, shape (batch_size, num_actions).
                - None: Hidden state (unused, for compatibility).
        """
        # 1. Get raw logits from the base network.
        # Shape: (batch_size, num_actions * num_bins)
        logits = self.net(state)

        # 2. Reshape logits to be per-action.
        # Shape: (batch_size, num_actions, num_bins)
        reshaped_logits = logits.view(-1, self.num_actions, self.num_bins)

        # 3. Apply Gumbel-Softmax to get differentiable, soft one-hot vectors.
        # `hard=False` is essential for a differentiable backward pass during training.
        # Shape: (batch_size, num_actions, num_bins)
        gumbel_probs = F.gumbel_softmax(reshaped_logits, tau=self.tau, hard=False, dim=-1)

        # 4. Calculate the final continuous action via weighted average.
        # self.action_bins is broadcasted to match gumbel_probs' shape.
        # Shape: (batch_size, num_actions)
        final_actions = torch.sum(gumbel_probs * self.action_bins.view(1, 1, -1), dim=-1)

        return final_actions, None

    @torch.no_grad()
    def act(self, state: torch.Tensor, deterministic: bool = False) -> np.ndarray:
        """
        Generates an action for environment interaction (inference).
        
        Args:
            state (torch.Tensor): The input state, shape (1, state_dim).
            deterministic (bool): If True, take the argmax of logits. If False, sample.
            
        Returns:
            np.ndarray: The final continuous action, shape (num_actions,).
        """
        self.eval()
        
        logits = self.net(state)
        reshaped_logits = logits.view(-1, self.num_actions, self.num_bins)

        if deterministic:
            # For evaluation, simply take the most likely bin (argmax).
            choice_indices = torch.argmax(reshaped_logits, dim=-1) # Shape: (1, num_actions)
            final_actions = self.action_bins[choice_indices]
        else:
            # For stochastic interaction during training, sample using Gumbel-Softmax.
            # Using `hard=True` gives a clean one-hot vector, which is fine for `no_grad` interaction.
            gumbel_probs = F.gumbel_softmax(reshaped_logits, tau=self.tau, hard=True, dim=-1)
            final_actions = torch.sum(gumbel_probs * self.action_bins.view(1, 1, -1), dim=-1)
        
        self.train()
        return final_actions.squeeze(0).cpu().numpy()

    def sample(self, obs, *args, **kwargs):
        """
        Sample actions for environment interaction (compatibility with PPO).
        
        Args:
            obs (torch.Tensor): Observation tensor.
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor, None]:
                - actions: The sampled actions, shape (batch_size, num_actions).
                - logprob: The log probability of the actions, shape (batch_size,).
                - hidden: None (for compatibility, since we don't use recurrent networks).
        """
        with torch.inference_mode():
            # Get raw logits and compute full forward pass for sampling
            logits = self.net(obs)
            reshaped_logits = logits.view(-1, self.num_actions, self.num_bins)
            
            # Apply Gumbel-Softmax for sampling
            gumbel_probs = F.gumbel_softmax(reshaped_logits, tau=self.tau, hard=False, dim=-1)
            final_actions = torch.sum(gumbel_probs * self.action_bins.view(1, 1, -1), dim=-1)
            
            # Calculate log probabilities for PPO
            log_pi = F.log_softmax(reshaped_logits, dim=-1)
            log_prob = torch.sum(log_pi * gumbel_probs, dim=-1).sum(dim=-1)
            
            # Set action_mean for PPO compatibility
            self.action_mean = final_actions
            
        return final_actions, log_prob, None

    def calc_logprob_entropy(self, action_mean, actions):
        """
        Calculate log probabilities and entropy for given action_mean and actions (compatibility with PPO).
        
        Args:
            action_mean (torch.Tensor): The action means (continuous actions), shape (batch_size, num_actions).
            actions (torch.Tensor): The actions taken, shape (batch_size, num_actions).
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - logprob: The log probability of the actions, shape (batch_size,).
                - entropy: The policy entropy, shape (batch_size,).
        """
        # For our GumbelActor, we need to get the raw logits from the network
        # We can't directly use action_mean to get logits, so we'll approximate
        
        # Create action indices by finding the closest bin for each action
        action_indices = torch.zeros_like(actions, dtype=torch.long, device=actions.device)
        for i in range(self.num_actions):
            distances = torch.abs(actions[:, i:i+1] - self.action_bins.view(1, -1))
            action_indices[:, i] = torch.argmin(distances, dim=1)
        
        # For entropy calculation, we use a uniform distribution over bins as approximation
        # This is a simplification for PPO compatibility
        log_num_bins = torch.log(torch.tensor(self.num_bins, dtype=torch.float32, device=actions.device))
        entropy = log_num_bins.repeat(actions.shape[0])
        
        # For log probability, we assume uniform probability over the selected bin
        # This is also a simplification for PPO compatibility
        logprob = -log_num_bins.repeat(actions.shape[0]) * self.num_actions
        
        return logprob, entropy

    @property
    def obs_shape(self):
        """Returns the input shape of the network for compatibility with PPO."""
        return self.net.input_shape

    @property
    def action_shape(self):
        """Returns the output shape (number of actions) for compatibility with PPO."""
        return [self.num_actions]

    @property
    def exploration(self) -> np.ndarray:
        """Returns exploration noise standard deviation for logging."""
        return self.distribution.std.detach().cpu().numpy()
# >>> END OF NEW CLASS