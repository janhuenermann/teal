import numpy as np
from numba import njit
from torch import Tensor
import torch
from torch.nn import functional as F


@njit
def compute_generalised_advantage(
    reward: np.ndarray,
    value: np.ndarray,
    is_last: np.ndarray,
    gamma: float = 0.99,
    gae_lambda: float = 0.9,
):
    """
    Computes the generalised advantage estimate (GAE). Please refer
    to the following paper for more details:

    HIGH-DIMENSIONAL CONTINUOUS CONTROL USING GENERALIZED ADVANTAGE ESTIMATION
    https://arxiv.org/abs/1506.02438
    """
    advantage = np.zeros((len(reward),), dtype=reward.dtype)

    # Compute deltas
    for i in range(len(reward) - 1, -1, -1):
        if is_last[i]:
            advantage[i] = reward[i] - value[i]
        else:
            advantage[i] = reward[i] + gamma * value[i + 1] - value[i]

    # Compute advantages
    gamma_lambda = gamma * gae_lambda
    for i in range(len(reward) - 2, -1, -1):
        if is_last[i]:
            continue
        advantage[i] += gamma_lambda * advantage[i + 1]

    return advantage


def sample_categorical(probs: Tensor):
    y = torch.rand(probs.shape[:-1], device=probs.device)
    return (probs.cumsum(-1) > y.unsqueeze(-1)).byte().argmax(-1)


def encode_two_hot(x: Tensor, bins: Tensor):
    """
    Twohot encoding is a generalization of onehot encoding to continuous values.
    It produces a vector of length |B| where all elements are 0 except for the
    two entries closest to the encoded continuous number, at positions k and k + 1.
    These two entries sum up to 1, with more weight given to the entry that is closer
    to the encoded number

    Args:
        x (Tensor): continuous number to encode. Decimals will be used for the twohot encoding
        num_bins (int): length of the encoded vector, i.e. max is |B| - 1
    """
    assert bins.ndim == 1
    num_bins = len(bins)
    left_count = (bins <= x[..., None]).long().sum(-1) - 1
    below_index = torch.clamp(left_count, 0, num_bins - 1)
    above_index = torch.clamp(left_count + 1, 0, num_bins - 1)
    equal = below_index == above_index
    below_dist = torch.where(equal, 1.0, torch.abs(bins[below_index] - x))
    above_dist = torch.where(equal, 1.0, torch.abs(bins[above_index] - x))
    total_dist = below_dist + above_dist
    weight_below = above_dist / total_dist
    weight_above = below_dist / total_dist
    out = torch.zeros(x.shape + (num_bins,), device=x.device)
    out.scatter_add_(-1, below_index[..., None], weight_below[..., None])
    out.scatter_add_(-1, above_index[..., None], weight_above[..., None])
    return out


def symexp(x: Tensor):
    """
    Symmetric exponential function, i.e. exp(x) for x <= 0 and -exp(-x) for x > 0.
    """
    return torch.sign(x) * (torch.exp(x.abs()) - 1.0)


def symlog(x: Tensor):
    """
    Symmetric logarithm function, i.e. log(x) for x > 0 and -log(-x) for x <= 0.
    """
    return torch.sign(x) * torch.log(1.0 + x.abs())


def probs_to_logits(probs: Tensor, eps: float = 1e-10) -> Tensor:
    return torch.log(probs.clamp(min=eps, max=1.0 - eps))
