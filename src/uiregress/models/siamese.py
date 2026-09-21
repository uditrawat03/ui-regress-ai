from __future__ import annotations

from typing import NamedTuple

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18


class SiameseOutput(NamedTuple):
    binary_logits: torch.Tensor
    class_logits: torch.Tensor


class SiameseRegressionClassifier(nn.Module):
    """Shared ResNet-18 encoder with binary and regression-type heads."""

    architecture_name = "resnet18-siamese-dual-head-v1"

    def __init__(self, num_classes: int, pretrained: bool = True) -> None:
        super().__init__()
        if num_classes < 2:
            raise ValueError("num_classes must be at least 2")

        weights = ResNet18_Weights.DEFAULT if pretrained else None
        backbone = resnet18(weights=weights)
        feature_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.encoder = backbone
        self.num_classes = num_classes

        comparison_dim = feature_dim * 3
        self.comparison = nn.Sequential(
            nn.Linear(comparison_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
        )
        self.binary_head = nn.Linear(512, 1)
        self.class_head = nn.Linear(512, num_classes)

    def forward(self, baseline: torch.Tensor, current: torch.Tensor) -> SiameseOutput:
        baseline_features = self.encoder(baseline)
        current_features = self.encoder(current)
        absolute_difference = torch.abs(baseline_features - current_features)
        pair_features = torch.cat(
            [baseline_features, current_features, absolute_difference],
            dim=1,
        )
        shared = self.comparison(pair_features)
        return SiameseOutput(
            binary_logits=self.binary_head(shared).squeeze(1),
            class_logits=self.class_head(shared),
        )
