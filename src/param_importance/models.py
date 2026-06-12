from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn
from torchvision.models import resnet18


class MLP(nn.Module):
    def __init__(
        self,
        input_shape: Sequence[int],
        num_classes: int,
        hidden_sizes: Sequence[int] = (256, 256),
    ) -> None:
        super().__init__()
        input_size = 1
        for value in input_shape:
            input_size *= int(value)
        layers: list[nn.Module] = [nn.Flatten()]
        previous = input_size
        for hidden in hidden_sizes:
            layers.extend([nn.Linear(previous, int(hidden)), nn.ReLU()])
            previous = int(hidden)
        layers.append(nn.Linear(previous, num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int = 10, smooth: bool = False) -> None:
        super().__init__()
        activation = nn.Softplus if smooth else nn.ReLU
        pooling: nn.Module = nn.AvgPool2d(2) if smooth else nn.MaxPool2d(2)
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            activation(),
            pooling,
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            activation(),
            pooling,
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            activation(),
            nn.AdaptiveAvgPool2d((2, 2)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 2 * 2, 256),
            activation(),
            nn.Linear(256, num_classes),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(inputs))


class TaskAwareResNet18(nn.Module):
    def __init__(self, classes_per_task: int, task_count: int) -> None:
        super().__init__()
        backbone = resnet18(weights=None)
        feature_count = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.heads = nn.ModuleList(
            nn.Linear(feature_count, classes_per_task) for _ in range(task_count)
        )

    def forward(self, inputs: torch.Tensor, task_id: int = 0) -> torch.Tensor:
        return self.heads[task_id](self.backbone(inputs))


class VisionTransformerTiny(nn.Module):
    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        num_classes: int = 100,
        embed_dim: int = 192,
        depth: int = 12,
        heads: int = 3,
    ) -> None:
        super().__init__()
        if image_size % patch_size:
            raise ValueError("image_size must be divisible by patch_size")
        self.patch_embed = nn.Conv2d(
            3,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
        patch_count = (image_size // patch_size) ** 2
        self.class_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.position = nn.Parameter(torch.zeros(1, patch_count + 1, embed_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        nn.init.trunc_normal_(self.position, std=0.02)
        nn.init.trunc_normal_(self.class_token, std=0.02)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        patches = self.patch_embed(inputs).flatten(2).transpose(1, 2)
        token = self.class_token.expand(inputs.shape[0], -1, -1)
        encoded = self.encoder(torch.cat([token, patches], dim=1) + self.position)
        return self.head(self.norm(encoded[:, 0]))


def build_model(
    name: str,
    input_shape: Sequence[int],
    num_classes: int,
    **kwargs: object,
) -> nn.Module:
    normalized = name.lower()
    if normalized in {"mlp", "mnist_mlp"}:
        return MLP(
            input_shape,
            num_classes,
            hidden_sizes=kwargs.get("hidden_sizes", (256, 256)),
        )
    if normalized in {"simple_cnn", "relu_cnn"}:
        return SimpleCNN(num_classes=num_classes, smooth=False)
    if normalized in {"smooth_cnn", "softplus_cnn"}:
        return SimpleCNN(num_classes=num_classes, smooth=True)
    if normalized in {"vit_tiny", "vision_transformer_tiny"}:
        return VisionTransformerTiny(
            image_size=int(kwargs.get("image_size", input_shape[-1])),
            patch_size=int(kwargs.get("patch_size", 16)),
            num_classes=num_classes,
            embed_dim=int(kwargs.get("embed_dim", 192)),
            depth=int(kwargs.get("depth", 12)),
            heads=int(kwargs.get("heads", 3)),
        )
    raise ValueError(f"Unknown model: {name}")
