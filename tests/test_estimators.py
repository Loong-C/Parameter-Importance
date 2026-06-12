import torch

from param_importance.estimators import (
    double_estimate,
    microbatch_estimate,
    naive_estimate,
    single_direct_estimate,
)
from param_importance.simulate import theoretical_gaussian_variances


def test_m2_microbatch_equals_symmetric_double() -> None:
    u = torch.tensor([[1.0, 2.0], [3.0, 5.0]])
    v = torch.tensor([[[2.0, 1.0], [7.0, 4.0]]])
    micro, _, _ = microbatch_estimate(u, v, gamma=0.3)
    double = double_estimate(
        u[:1],
        v[:, :1],
        u[1:],
        v[:, 1:],
        gamma=0.3,
        symmetric=True,
    )
    torch.testing.assert_close(micro, double)


def test_zero_variance_needs_no_correction() -> None:
    u = torch.full((8, 3), 2.0)
    v = torch.full((1, 8, 3), 4.0)
    naive = naive_estimate(u, v, gamma=0.1)
    corrected, raw, correction = single_direct_estimate(u, v, gamma=0.1)
    torch.testing.assert_close(raw, naive)
    torch.testing.assert_close(corrected, naive)
    torch.testing.assert_close(correction, torch.zeros_like(correction))


def test_direct_correction_is_monte_carlo_unbiased() -> None:
    generator = torch.Generator().manual_seed(3)
    repetitions = 20_000
    batch_size = 8
    samples = 0.5 + torch.randn((repetitions, batch_size), generator=generator)
    means = samples.mean(dim=1)
    variances = samples.var(dim=1, unbiased=True)
    naive = means.square()
    direct = means.square() - variances / batch_size
    assert abs(float(direct.mean()) - 0.25) < 0.01
    assert float(naive.mean()) > float(direct.mean()) + 0.1


def test_independent_batch_changes_double_not_naive() -> None:
    u_a = torch.tensor([[1.0], [2.0]])
    v_a = u_a.unsqueeze(0)
    u_b = torch.tensor([[10.0], [20.0]])
    v_b = u_b.unsqueeze(0)
    naive = naive_estimate(u_a, v_a, gamma=1.0)
    double = double_estimate(u_a, v_a, u_b, v_b, gamma=1.0)
    assert not torch.equal(naive, double)


def test_matched_budget_variance_is_equal_at_m2_and_larger_afterward() -> None:
    m2 = theoretical_gaussian_variances(0.5, 1.0, 32, 2, 0.01)
    assert m2["single_micro"] == m2["double_matched"]
    m8 = theoretical_gaussian_variances(0.5, 1.0, 32, 8, 0.01)
    assert m8["single_micro"] < m8["double_matched"]
