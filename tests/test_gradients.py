import torch

from param_importance.gradients import FunctionalGradientComputer
from param_importance.models import MLP


def test_per_sample_mean_matches_batch_gradient() -> None:
    torch.manual_seed(1)
    model = MLP((1, 2, 2), 3, hidden_sizes=(5,))
    computer = FunctionalGradientComputer(model, "cpu")
    inputs = torch.randn(6, 1, 2, 2)
    targets = torch.tensor([0, 1, 2, 0, 1, 2])
    batch = computer.mean_gradient(inputs, targets)
    per_sample = computer.per_sample_gradient(inputs, targets).mean(0)
    torch.testing.assert_close(batch, per_sample, rtol=1e-5, atol=1e-6)


def test_microbatch_gradient_mean_matches_full_batch() -> None:
    torch.manual_seed(2)
    model = MLP((1, 2, 2), 2, hidden_sizes=(4,))
    computer = FunctionalGradientComputer(model, "cpu")
    inputs = torch.randn(8, 1, 2, 2)
    targets = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1])
    full = computer.mean_gradient(inputs, targets)
    groups = torch.stack(
        [
            computer.mean_gradient(inputs[:4], targets[:4]),
            computer.mean_gradient(inputs[4:], targets[4:]),
        ]
    )
    torch.testing.assert_close(full, groups.mean(0), rtol=1e-5, atol=1e-6)


def test_streamed_group_means_match_explicit_microbatches() -> None:
    torch.manual_seed(3)
    model = MLP((1, 2, 2), 2, hidden_sizes=(4,))
    computer = FunctionalGradientComputer(model, "cpu")
    inputs = torch.randn(8, 1, 2, 2)
    targets = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1])
    _, grouped = computer.paired_cross_moments_with_groups(
        inputs,
        targets,
        [computer.params],
        chunk_size=3,
        group_counts=[2, 4],
    )
    for count in [2, 4]:
        group_size = inputs.shape[0] // count
        explicit = torch.stack(
            [
                computer.mean_gradient(
                    inputs[index * group_size : (index + 1) * group_size],
                    targets[index * group_size : (index + 1) * group_size],
                )
                for index in range(count)
            ]
        )
        update_means, node_means = grouped[count]
        torch.testing.assert_close(update_means, explicit, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(node_means[0], explicit, rtol=1e-5, atol=1e-6)
