from __future__ import annotations

from collections.abc import Sequence

import torch


def _weights_tensor(
    weights: Sequence[float] | torch.Tensor | None,
    reference: torch.Tensor,
) -> torch.Tensor:
    node_count = reference.shape[0]
    if weights is None:
        return torch.full(
            (node_count,),
            1.0 / node_count,
            dtype=reference.dtype,
            device=reference.device,
        )
    result = torch.as_tensor(weights, dtype=reference.dtype, device=reference.device)
    if result.shape != (node_count,):
        raise ValueError(f"Expected {node_count} quadrature weights, got {tuple(result.shape)}")
    return result


def _integrate(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    return torch.einsum("q,q...->...", weights, values)


def oracle_estimate(
    mean_u: torch.Tensor,
    mean_v: torch.Tensor,
    gamma: float,
    weights: Sequence[float] | torch.Tensor | None = None,
) -> torch.Tensor:
    if mean_v.ndim != mean_u.ndim + 1:
        raise ValueError("mean_v must have one leading quadrature-node dimension")
    node_weights = _weights_tensor(weights, mean_v)
    return gamma * mean_u * _integrate(mean_v, node_weights)


def naive_estimate(
    u_samples: torch.Tensor,
    v_samples: torch.Tensor,
    gamma: float,
    weights: Sequence[float] | torch.Tensor | None = None,
) -> torch.Tensor:
    if u_samples.ndim < 1 or v_samples.ndim != u_samples.ndim + 1:
        raise ValueError("Expected u=[B,...] and v=[Q,B,...]")
    if u_samples.shape[0] != v_samples.shape[1]:
        raise ValueError("u and v sample counts differ")
    return oracle_estimate(u_samples.mean(0), v_samples.mean(1), gamma, weights)


def single_direct_from_moments(
    mean_u: torch.Tensor,
    mean_v: torch.Tensor,
    sample_cross_covariance: torch.Tensor,
    sample_count: int,
    gamma: float,
    weights: Sequence[float] | torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if sample_count < 2:
        raise ValueError("Single-sample correction requires at least two samples")
    node_weights = _weights_tensor(weights, mean_v)
    raw = gamma * mean_u * _integrate(mean_v, node_weights)
    correction = gamma * _integrate(sample_cross_covariance, node_weights) / sample_count
    return raw - correction, raw, correction


def single_direct_estimate(
    u_samples: torch.Tensor,
    v_samples: torch.Tensor,
    gamma: float,
    weights: Sequence[float] | torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size = u_samples.shape[0]
    mean_u = u_samples.mean(0)
    mean_v = v_samples.mean(1)
    centered_u = u_samples - mean_u
    centered_v = v_samples - mean_v[:, None, ...]
    reduce_dims = (1,)
    covariance = (centered_v * centered_u.unsqueeze(0)).sum(dim=reduce_dims) / (batch_size - 1)
    return single_direct_from_moments(
        mean_u,
        mean_v,
        covariance,
        batch_size,
        gamma,
        weights,
    )


def microbatch_estimate(
    u_group_means: torch.Tensor,
    v_group_means: torch.Tensor,
    gamma: float,
    weights: Sequence[float] | torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Correct a shared-batch estimate using covariance of microbatch means.

    u_group_means has shape [M,...], v_group_means has shape [Q,M,...].
    """

    group_count = u_group_means.shape[0]
    if group_count < 2:
        raise ValueError("Microbatch correction requires at least two groups")
    if v_group_means.shape[1] != group_count:
        raise ValueError("u and v microbatch counts differ")
    mean_u = u_group_means.mean(0)
    mean_v = v_group_means.mean(1)
    centered_u = u_group_means - mean_u
    centered_v = v_group_means - mean_v[:, None, ...]
    covariance = (centered_v * centered_u.unsqueeze(0)).sum(dim=1) / (group_count - 1)
    return single_direct_from_moments(
        mean_u,
        mean_v,
        covariance,
        group_count,
        gamma,
        weights,
    )


def double_estimate(
    u_a: torch.Tensor,
    v_a: torch.Tensor,
    u_b: torch.Tensor,
    v_b: torch.Tensor,
    gamma: float,
    weights: Sequence[float] | torch.Tensor | None = None,
    symmetric: bool = True,
) -> torch.Tensor:
    """Estimate with independent halves.

    The symmetric form averages A-update/B-evaluation and B-update/A-evaluation.
    It is exactly the M=2 microbatch U-statistic under a shared total sample budget.
    """

    mean_u_a = u_a.mean(0)
    mean_u_b = u_b.mean(0)
    mean_v_a = v_a.mean(1)
    mean_v_b = v_b.mean(1)
    a_to_b = oracle_estimate(mean_u_a, mean_v_b, gamma, weights)
    if not symmetric:
        return a_to_b
    b_to_a = oracle_estimate(mean_u_b, mean_v_a, gamma, weights)
    return (a_to_b + b_to_a) / 2


def ppt_variance_only_ablation(
    mean_u: torch.Tensor,
    mean_v: torch.Tensor,
    variance_u: torch.Tensor,
    sample_count: int,
    gamma: float,
    weights: Sequence[float] | torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """PPT ablation: subtract update-point variance at every path node."""

    node_weights = _weights_tensor(weights, mean_v)
    raw = gamma * mean_u * _integrate(mean_v, node_weights)
    correction = gamma * variance_u / sample_count
    return raw - correction, raw, correction

