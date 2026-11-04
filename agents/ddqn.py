"""Inference adapter for MarioPlayingAgent Double DQN checkpoints."""

import torch
from torch import nn


class MarioDDQN(nn.Module):
    def __init__(self, action_count):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(4, 32, 8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1), nn.ReLU(),
            nn.Flatten(),
        )
        self.head = nn.Sequential(
            nn.Linear(64 * 7 * 7, 512), nn.ReLU(),
            nn.Linear(512, action_count),
        )

    @classmethod
    def load(cls, path, action_count):
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        policy = cls(action_count)
        policy.load_state_dict(checkpoint.get("online_net_state_dict", checkpoint))
        return policy.eval()

    @torch.inference_mode()
    def predict(self, observation, deterministic=True):
        pixels = torch.from_numpy(observation).unsqueeze(0).float() / 255
        values = self.head(self.features(pixels))
        return values.argmax(dim=1), None
