import torch
from teal.utils.distributions import SymlogCategorical
from teal.utils.math import symexp, symlog, encode_two_hot


def test_two_hot_encoding():
    bins = torch.tensor([0.0, 1.0, 2.0])
    torch.testing.assert_close(
        encode_two_hot(torch.tensor(1.2), bins),
        torch.tensor([0.0, 0.8, 0.2]),
    )

    torch.testing.assert_close(
        encode_two_hot(torch.tensor(4.0), bins),
        torch.tensor([0.0, 0.0, 1.0]),
    )

    torch.testing.assert_close(
        encode_two_hot(torch.tensor(0.5), bins),
        torch.tensor([0.5, 0.5, 0.0]),
    )

    torch.testing.assert_close(
        encode_two_hot(torch.tensor(0.9), bins),
        torch.tensor([0.1, 0.9, 0.0]),
    )

    torch.testing.assert_close(
        encode_two_hot(torch.tensor(-1.0), bins),
        torch.tensor([1.0, 0.0, 0.0]),
    )

    # Mean of two hot encoding should be the original value:
    random_vals = (4.0 * torch.randn(20)).clamp_(min=-5.0, max=5.0)
    bins = torch.linspace(-5, 5, 100)
    torch.testing.assert_close(
        torch.sum(bins * encode_two_hot(random_vals, bins), -1),
        random_vals,
    )


def test_symlog():
    # Symlog should be the inverse of symexp:
    xs = torch.linspace(-5.0, 5.0, 100)
    torch.testing.assert_close(symlog(symexp(xs)), xs, rtol=1e-4, atol=1e-4)

    # Symlog categorical mean should be the original value:
    lo = -5.0
    hi = 5.0
    value = lo + (hi - lo) * torch.rand(20)
    bins = torch.linspace(lo, hi, 100)
    probs = encode_two_hot(symlog(value), bins)

    dist = SymlogCategorical(probs.log().nan_to_num(neginf=-1e6), lo, hi)
    torch.testing.assert_close(dist.mean(), value, rtol=1e-4, atol=1e-4)
    dist.log_prob(value)
