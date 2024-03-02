from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor
from torch.jit import script, interface  # type: ignore
from torch.nn import functional as F

from teal.utils.math import (
    encode_two_hot,
    probs_to_logits,
    sample_categorical,
    symexp,
    symlog,
)


@script
@dataclass
class Bernoulli:
    logit: Tensor

    def log_prob(self, x: Tensor) -> Tensor:
        assert x.dtype == torch.bool
        # -softplus(x) = log(1 - p)
        # -softplus(-x) = log(p)
        # where p = sigmoid(x)
        return -F.softplus(torch.where(x, -self.logit, self.logit))

    def mean(self) -> Tensor:
        return torch.sigmoid(self.logit)

    def mode(self) -> Tensor:
        return self.logit > 0.0


@script
@dataclass
class SymlogCategorical:
    """
    Represent a scalar value using a categorical distribution over a
    symlog-transformed space. Proposed in the DreamerV3 paper.
    """

    logits: Tensor
    low: float = -5.0
    high: float = 5.0

    def make_bins(self) -> Tensor:
        return torch.linspace(
            self.low, self.high, self.logits.shape[-1], device=self.logits.device
        )

    def log_prob(self, x: Tensor) -> Tensor:
        x = symlog(x)
        two_hot = encode_two_hot(x, self.make_bins())
        log_probs = torch.log_softmax(self.logits, dim=-1)
        return (two_hot * log_probs).sum(dim=-1)

    def mean(self) -> Tensor:
        probs = torch.softmax(self.logits, dim=-1)
        return symexp((probs * self.make_bins()).sum(dim=-1))

    def mode(self) -> Tensor:
        mode_probs = F.one_hot(self.logits.argmax(dim=-1), self.logits.shape[-1])
        return symexp((mode_probs.float() * self.make_bins()).sum(dim=-1))


@script
@dataclass
class MSE:
    value: Tensor

    def log_prob(self, x: Tensor) -> Tensor:
        return -((x - self.value) ** 2)

    def mean(self) -> Tensor:
        return self.value

    def mode(self) -> Tensor:
        return self.value


@script
@dataclass
class FlatMultivariateOneHot:
    # [..., N, D] -> [..., N * D]
    probs: Tensor

    def sample(self) -> Tensor:
        return self.straight_through_one_hot(sample_categorical(self.probs))

    def mean(self) -> Tensor:
        return self.probs

    def mode(self) -> Tensor:
        return self.straight_through_one_hot(self.probs.argmax(dim=-1))

    def log_prob(self, x: Tensor) -> Tensor:
        logits = probs_to_logits(self.probs)
        return torch.sum(logits * x.view(self.probs.shape), dim=(-1, -2))

    def entropy(self) -> Tensor:
        logits = probs_to_logits(self.probs)
        return -torch.sum(self.probs * logits, dim=(-1, -2))

    def kl_div(self, other: "FlatMultivariateOneHot") -> Tensor:
        return torch.sum(
            self.probs * (probs_to_logits(self.probs) - probs_to_logits(other.probs)),
            dim=(-1, -2),
        )

    def straight_through_one_hot(self, indices: Tensor) -> Tensor:
        one_hot = F.one_hot(indices, self.probs.shape[-1]).float()
        one_hot = one_hot + self.probs - self.probs.detach()
        return one_hot.flatten(-2)

    def detach(self) -> "FlatMultivariateOneHot":
        return FlatMultivariateOneHot(self.probs.detach())


@interface
class Distribution:
    def log_prob(self, x: Tensor) -> Tensor:  # type: ignore
        pass

    def mean(self) -> Tensor:  # type: ignore
        pass

    def mode(self) -> Tensor:  # type: ignore
        pass


# TODO: remove when torch.jit.interface returns proper type
# Distribution: type[_Distribution] = interface(_Distribution)  # type: ignore


@script
def log_prob(dist: Distribution, x: Tensor):
    return dist.log_prob(x)  # type: ignore


@script
def mean(dist: Distribution):
    return dist.mean()  # type: ignore


@script
def mode(dist: Distribution):
    return dist.mode()
