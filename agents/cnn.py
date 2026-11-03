"""Small convolutional feature extractor for stacked NES frames."""

import torch
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn


class MarioCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=128):
        super().__init__(observation_space, features_dim)
        self.conv = nn.Sequential(
            nn.Conv2d(observation_space.shape[0], 16, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            size = self.conv(torch.zeros(1, *observation_space.shape)).shape[1]
        self.project = nn.Sequential(nn.Linear(size, features_dim), nn.ReLU())

    def forward(self, observations):
        return self.project(self.conv(observations))
