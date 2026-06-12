import pytest

from param_importance.checkpoint import _gradient_group_counts


def test_double_group_is_computed_when_m2_is_not_requested() -> None:
    requested, computed = _gradient_group_counts(64, [4, 8])
    assert requested == [4, 8]
    assert computed == [2, 4, 8]


def test_double_sampling_rejects_odd_batch_size() -> None:
    with pytest.raises(ValueError, match="even checkpoint batch sizes"):
        _gradient_group_counts(63, [3, 7])
