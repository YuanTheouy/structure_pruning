import copy

import torch
import torch.nn as nn

from .module import MLP

__all__ = ['get_trajectory_lengths', 'split_trajectory', 'get_trajectory_mask',
           'unpad_trajectory', 'is_recurrent', 'RMlp']


@torch.jit.script
def get_trajectory_lengths(dones: torch.Tensor) -> torch.Tensor:
    dones = dones.clone()
    dones[-1] = 1
    # Permute the buffers to have order (num_envs, num_collects, ...), for correct reshaping
    flat_dones = dones.transpose(1, 0).flatten()

    # Get length of trajectory by counting the number of successive not done elements
    done_indices = torch.cat((
        torch.tensor([-1], dtype=torch.int64, device=dones.device),
        flat_dones.nonzero().squeeze()
    ))

    # done_indices = torch.nn.functional.pad(flat_dones.nonzero().squeeze(), (1, 0), value=-1)
    trajectory_lengths = done_indices[1:] - done_indices[:-1]
    return trajectory_lengths


@torch.jit.script
def split_trajectory(tensor: torch.Tensor, lengths: torch.Tensor):
    # Extract the individual trajectories
    lengths_list: list[int] = lengths.tolist()
    trajectories = torch.split(tensor.transpose(1, 0).flatten(0, 1), lengths_list)
    padded_trajectories = torch.nn.utils.rnn.pad_sequence(trajectories)
    return padded_trajectories


@torch.jit.script
def get_trajectory_mask(lengths: torch.Tensor):
    trajectory_masks = lengths > torch.arange(0, lengths.max(), device=lengths.device).unsqueeze_(1)
    return trajectory_masks


@torch.jit.script
def unpad_trajectory(trajectories: torch.Tensor, masks: torch.Tensor, sequence_len: int):
    """
    Does the inverse operation of split_trajectory()
    """
    # Need to transpose before and after the masking to have proper reshaping
    return trajectories.transpose(1, 0)[masks.transpose(1, 0)].reshape(
        -1, sequence_len, trajectories.shape[-1]).transpose(1, 0)


def is_recurrent(*modules):
    return any(
        getattr(m, 'is_recurrent', False) for m in modules
    )


class RMlp(nn.Module):
    is_recurrent = True

    def __init__(
            self,
            rnn_type,
            rnn_num_layers,
            rnn_hidden_dim,
            mlp_shape,
            activation_fn,
            input_size,
            output_size
    ):
        super(RMlp, self).__init__()
        rnn_type_ = rnn_type.lower()
        if rnn_type_ == 'lstm':
            self.rnn = nn.LSTM(
                input_size=input_size,
                num_layers=rnn_num_layers,
                hidden_size=rnn_hidden_dim,
            )
        elif rnn_type_ == 'gru':
            self.rnn = nn.GRU(
                input_size=input_size,
                num_layers=rnn_num_layers,
                hidden_size=rnn_hidden_dim,
            )
        else:
            raise ValueError(f"Unknown RNN Type {rnn_type}")

        self.mlp = MLP(mlp_shape, activation_fn, rnn_hidden_dim, output_size)
        self.input_shape = [input_size]
        self.output_shape = [output_size]

    def forward(self, x, h=None):
        # for x.dim() == 2, the 1st dim is seen as batch instead of time
        if x.dim() < 3:
            x = x.unsqueeze(0)
        f, hn = self.rnn(x, h)
        return self.mlp(f.squeeze(0)), hn

    def __deepcopy__(self, memodict={}):
        cls = self.__class__
        result = cls.__new__(cls)
        memodict[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, copy.deepcopy(v, memodict))
        result.rnn.flatten_parameters()
        return result

