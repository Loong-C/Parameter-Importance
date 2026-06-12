from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class RunningCrossMoments:
    """Elementwise Welford moments for paired tensors.

    x batches have shape [N, D] and y batches have shape [Q, N, D].
    """

    count: int = 0
    mean_x: torch.Tensor | None = None
    mean_y: torch.Tensor | None = None
    cross_deviation: torch.Tensor | None = None

    def update(self, x: torch.Tensor, y: torch.Tensor) -> None:
        if x.ndim != 2 or y.ndim != 3:
            raise ValueError("Expected x=[N,D] and y=[Q,N,D]")
        if x.shape[0] != y.shape[1] or x.shape[1] != y.shape[2]:
            raise ValueError("x and y batch dimensions do not match")
        batch_count = x.shape[0]
        if batch_count == 0:
            return

        batch_mean_x = x.mean(dim=0)
        batch_mean_y = y.mean(dim=1)
        centered_x = x - batch_mean_x
        centered_y = y - batch_mean_y[:, None, :]
        batch_cross = (centered_y * centered_x[None, :, :]).sum(dim=1)

        if self.count == 0:
            self.count = batch_count
            self.mean_x = batch_mean_x
            self.mean_y = batch_mean_y
            self.cross_deviation = batch_cross
            return

        assert self.mean_x is not None
        assert self.mean_y is not None
        assert self.cross_deviation is not None
        total = self.count + batch_count
        delta_x = batch_mean_x - self.mean_x
        delta_y = batch_mean_y - self.mean_y
        adjustment = delta_y * delta_x[None, :] * (self.count * batch_count / total)
        self.cross_deviation = self.cross_deviation + batch_cross + adjustment
        self.mean_x = self.mean_x + delta_x * (batch_count / total)
        self.mean_y = self.mean_y + delta_y * (batch_count / total)
        self.count = total

    @property
    def sample_cross_covariance(self) -> torch.Tensor:
        if self.count < 2 or self.cross_deviation is None:
            if self.mean_y is None:
                raise ValueError("No observations have been added")
            return torch.zeros_like(self.mean_y)
        return self.cross_deviation / (self.count - 1)

