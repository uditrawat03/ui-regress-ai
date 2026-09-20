from __future__ import annotations

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18


class SiameseRegressionClassifier(nn.Module):
    """Starter pairwise classifier.

    This is deliberately small and debuggable. It is a scaffold for Milestone 3,
    not a trained production model.
    """

    def __init__(self, num_classes: int = 9, pretrained: bool = True) -> None:
        super().__init__()
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        backbone = resnet18(weights=weights)
        feature_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.encoder = backbone

        self.classifier = nn.Sequential(
            nn.Linear(feature_dim * 3, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(512, num_classes),
        )

    def forward(self, baseline: torch.Tensor, current: torch.Tensor) -> torch.Tensor:
        baseline_features = self.encoder(baseline)
        current_features = self.encoder(current)
        absolute_difference = torch.abs(baseline_features - current_features)
        pair_features = torch.cat(
            [baseline_features, current_features, absolute_difference], dim=1
        )
        return self.classifier(pair_features)
