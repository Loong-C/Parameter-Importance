import torch

from param_importance.online import RunningCrossMoments


def test_chunked_cross_moments_match_direct_calculation() -> None:
    generator = torch.Generator().manual_seed(7)
    x = torch.randn((13, 5), generator=generator)
    y = torch.randn((3, 13, 5), generator=generator)
    moments = RunningCrossMoments()
    moments.update(x[:4], y[:, :4])
    moments.update(x[4:9], y[:, 4:9])
    moments.update(x[9:], y[:, 9:])

    centered_x = x - x.mean(0)
    centered_y = y - y.mean(1, keepdim=True)
    expected = (centered_y * centered_x.unsqueeze(0)).sum(1) / (x.shape[0] - 1)
    torch.testing.assert_close(moments.mean_x, x.mean(0))
    torch.testing.assert_close(moments.mean_y, y.mean(1))
    torch.testing.assert_close(moments.sample_cross_covariance, expected)

