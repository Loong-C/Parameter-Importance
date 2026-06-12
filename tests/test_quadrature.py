import numpy as np

from param_importance.quadrature import (
    adaptive_vector_integral,
    composite_trapezoid_vector,
    get_rule,
)


def test_gauss_three_integrates_degree_five_polynomial() -> None:
    rule = get_rule("gauss3")
    estimate = sum(weight * node**5 for node, weight in zip(rule.nodes, rule.weights))
    assert np.isclose(estimate, 1 / 6)


def test_adaptive_vector_integral_matches_polynomial_analytic_value() -> None:
    result = adaptive_vector_integral(
        lambda value: np.asarray([value**2, value**5]),
        epsabs=1e-12,
        epsrel=1e-12,
    )
    assert result.success
    assert np.allclose(result.value, [1 / 3, 1 / 6], atol=1e-11)


def test_adaptive_and_refined_trapezoid_agree_on_kink() -> None:
    function = lambda value: np.asarray([abs(value - 0.37), value])
    adaptive = adaptive_vector_integral(
        function,
        epsabs=1e-11,
        epsrel=1e-11,
        points=[0.37],
    )
    refined = composite_trapezoid_vector(function, 10_000)
    exact = np.asarray([(0.37**2 + 0.63**2) / 2, 0.5])
    assert np.allclose(adaptive.value, exact, atol=1e-10)
    assert np.allclose(refined, exact, atol=1e-8)
