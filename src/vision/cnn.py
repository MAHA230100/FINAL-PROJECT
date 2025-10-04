from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as models


class SimpleCNN(nn.Module):
	def __init__(self, num_classes: int = 2, pretrained: bool = True):
		super().__init__()
		backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
		in_features = backbone.fc.in_features
		backbone.fc = nn.Linear(in_features, num_classes)
		self.model = backbone

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		return self.model(x)
