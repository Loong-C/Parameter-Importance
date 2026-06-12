from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn
from torch.func import functional_call, grad, vmap
from torch.utils.data import DataLoader, Dataset

from .online import RunningCrossMoments


@dataclass(frozen=True, slots=True)
class ParameterSlice:
    name: str
    group: str
    start: int
    stop: int
    shape: torch.Size


class FunctionalGradientComputer:
    def __init__(
        self,
        model: nn.Module,
        device: torch.device | str,
        forward_args: tuple[object, ...] = (),
    ) -> None:
        self.model = model.to(device)
        self.device = torch.device(device)
        self.forward_args = forward_args
        self.params = OrderedDict(
            (name, parameter.detach().clone().requires_grad_(True))
            for name, parameter in self.model.named_parameters()
            if parameter.requires_grad
        )
        self.buffers = OrderedDict(
            (name, buffer.detach().clone())
            for name, buffer in self.model.named_buffers()
        )
        self.layout: list[ParameterSlice] = []
        cursor = 0
        for name, parameter in self.params.items():
            count = parameter.numel()
            self.layout.append(
                ParameterSlice(
                    name=name,
                    group=name.rsplit(".", 1)[0] if "." in name else name,
                    start=cursor,
                    stop=cursor + count,
                    shape=parameter.shape,
                )
            )
            cursor += count
        self.parameter_count = cursor

        def loss_function(
            params: Mapping[str, torch.Tensor],
            buffers: Mapping[str, torch.Tensor],
            inputs: torch.Tensor,
            targets: torch.Tensor,
        ) -> torch.Tensor:
            logits = functional_call(
                self.model,
                (params, buffers),
                (inputs, *self.forward_args),
            )
            return F.cross_entropy(logits, targets)

        def single_loss(
            params: Mapping[str, torch.Tensor],
            buffers: Mapping[str, torch.Tensor],
            inputs: torch.Tensor,
            target: torch.Tensor,
        ) -> torch.Tensor:
            logits = functional_call(
                self.model,
                (params, buffers),
                (inputs.unsqueeze(0), *self.forward_args),
            )
            return F.cross_entropy(logits, target.unsqueeze(0))

        self._loss = loss_function
        self._mean_grad = grad(loss_function)
        self._single_grad = grad(single_loss)

    def flatten(self, values: Mapping[str, torch.Tensor]) -> torch.Tensor:
        return torch.cat([values[name].reshape(-1) for name in self.params])

    def unflatten(self, vector: torch.Tensor) -> OrderedDict[str, torch.Tensor]:
        if vector.numel() != self.parameter_count:
            raise ValueError(
                f"Expected {self.parameter_count} values, got {vector.numel()}"
            )
        result: OrderedDict[str, torch.Tensor] = OrderedDict()
        for item in self.layout:
            result[item.name] = vector[item.start : item.stop].reshape(item.shape)
        return result

    def shifted_params(self, delta: torch.Tensor, alpha: float) -> OrderedDict[str, torch.Tensor]:
        delta_dict = self.unflatten(delta)
        return OrderedDict(
            (
                name,
                (parameter + alpha * delta_dict[name]).detach().requires_grad_(True),
            )
            for name, parameter in self.params.items()
        )

    def mean_gradient(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        params: Mapping[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        gradient = self._mean_grad(
            params or self.params,
            self.buffers,
            inputs,
            targets,
        )
        return self.flatten(gradient).detach()

    def per_sample_gradient(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        params: Mapping[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        gradients = vmap(
            self._single_grad,
            in_dims=(None, None, 0, 0),
            randomness="different",
        )(params or self.params, self.buffers, inputs, targets)
        return torch.cat(
            [gradients[name].reshape(inputs.shape[0], -1) for name in self.params],
            dim=1,
        ).detach()

    def loss(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        params: Mapping[str, torch.Tensor] | None = None,
    ) -> float:
        with torch.no_grad():
            value = self._loss(params or self.params, self.buffers, inputs, targets)
        return float(value.item())

    def full_dataset_gradient(
        self,
        dataset: Dataset,
        batch_size: int,
        params: Mapping[str, torch.Tensor] | None = None,
        workers: int = 0,
    ) -> torch.Tensor:
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers)
        total = torch.zeros(self.parameter_count, device=self.device)
        count = 0
        for inputs, targets in loader:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            current = inputs.shape[0]
            total += self.mean_gradient(inputs, targets, params) * current
            count += current
        return total / count

    def full_dataset_loss(
        self,
        dataset: Dataset,
        batch_size: int,
        params: Mapping[str, torch.Tensor] | None = None,
        workers: int = 0,
    ) -> float:
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers)
        weighted_loss = 0.0
        count = 0
        for inputs, targets in loader:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            current = inputs.shape[0]
            weighted_loss += self.loss(inputs, targets, params) * current
            count += current
        return weighted_loss / count

    def paired_cross_moments(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        node_params: Iterable[Mapping[str, torch.Tensor]],
        chunk_size: int,
    ) -> RunningCrossMoments:
        nodes = list(node_params)
        moments = RunningCrossMoments()
        for start in range(0, inputs.shape[0], chunk_size):
            stop = min(start + chunk_size, inputs.shape[0])
            chunk_inputs = inputs[start:stop]
            chunk_targets = targets[start:stop]
            update_gradients = self.per_sample_gradient(
                chunk_inputs,
                chunk_targets,
                self.params,
            )
            node_gradients = torch.stack(
                [
                    self.per_sample_gradient(chunk_inputs, chunk_targets, params)
                    for params in nodes
                ]
            )
            moments.update(update_gradients, node_gradients)
        return moments

    def paired_cross_moments_with_groups(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        node_params: Iterable[Mapping[str, torch.Tensor]],
        chunk_size: int,
        group_counts: Iterable[int],
    ) -> tuple[
        RunningCrossMoments,
        dict[int, tuple[torch.Tensor, torch.Tensor]],
    ]:
        nodes = list(node_params)
        counts = sorted(set(int(value) for value in group_counts))
        for count in counts:
            if count < 2 or inputs.shape[0] % count:
                raise ValueError(
                    f"Batch size {inputs.shape[0]} is not divisible by group count {count}"
                )
        moments = RunningCrossMoments()
        update_sums = {
            count: torch.zeros(
                (count, self.parameter_count),
                device=self.device,
            )
            for count in counts
        }
        node_sums = {
            count: torch.zeros(
                (len(nodes), count, self.parameter_count),
                device=self.device,
            )
            for count in counts
        }
        group_sizes = {count: inputs.shape[0] // count for count in counts}

        for start in range(0, inputs.shape[0], chunk_size):
            stop = min(start + chunk_size, inputs.shape[0])
            chunk_inputs = inputs[start:stop]
            chunk_targets = targets[start:stop]
            update_gradients = self.per_sample_gradient(
                chunk_inputs,
                chunk_targets,
                self.params,
            )
            node_gradients = torch.stack(
                [
                    self.per_sample_gradient(chunk_inputs, chunk_targets, params)
                    for params in nodes
                ]
            )
            moments.update(update_gradients, node_gradients)
            global_indices = torch.arange(start, stop, device=self.device)
            for count in counts:
                group_ids = torch.div(
                    global_indices,
                    group_sizes[count],
                    rounding_mode="floor",
                )
                update_sums[count].index_add_(0, group_ids, update_gradients)
                for node_id in range(len(nodes)):
                    node_sums[count][node_id].index_add_(
                        0,
                        group_ids,
                        node_gradients[node_id],
                    )

        group_means = {
            count: (
                update_sums[count] / group_sizes[count],
                node_sums[count] / group_sizes[count],
            )
            for count in counts
        }
        return moments, group_means


def group_slices(layout: list[ParameterSlice]) -> dict[str, list[slice]]:
    groups: dict[str, list[slice]] = {"all": [slice(0, layout[-1].stop)]}
    for item in layout:
        groups.setdefault(item.group, []).append(slice(item.start, item.stop))
    return groups


def select_slices(vector: torch.Tensor, slices: list[slice]) -> torch.Tensor:
    return torch.cat([vector[current] for current in slices])
