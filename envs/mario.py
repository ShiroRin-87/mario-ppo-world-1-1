"""Gymnasium-facing environment for the legacy NES emulator."""

from collections import deque

import gymnasium as gym
import gym_super_mario_bros
import numpy as np
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace
from PIL import Image


class MarioEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, render_mode="rgb_array", max_steps=1200,
                 progress_weight=0.0, death_penalty=0.0):
        if render_mode not in self.metadata["render_modes"]:
            raise ValueError(f"Invalid render_mode: {render_mode}")
        if max_steps < 1:
            raise ValueError("max_steps must be positive")
        self.render_mode = render_mode
        self.max_steps = max_steps
        self.progress_weight = progress_weight
        self.death_penalty = death_penalty
        self.frame_skip = 4
        self.action_space = gym.spaces.Discrete(len(SIMPLE_MOVEMENT))
        self.observation_space = gym.spaces.Box(0, 255, (4, 84, 84), np.uint8)
        self._frames = deque(maxlen=4)
        raw = gym_super_mario_bros.make(
            "SuperMarioBros-1-1-v0", disable_env_checker=True
        ).unwrapped
        self._env = JoypadSpace(raw, SIMPLE_MOVEMENT)
        self._rgb = None
        self._steps = 0
        self._last_x = None

    @staticmethod
    def _process(rgb):
        gray = Image.fromarray(rgb).convert("L")
        return np.asarray(gray.resize((84, 84), Image.Resampling.BOX))

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._env.seed(seed)
        self._steps = 0
        self._last_x = None
        self._rgb = self._env.reset().copy()
        frame = self._process(self._rgb)
        self._frames.clear()
        self._frames.extend([frame] * self._frames.maxlen)
        return np.stack(self._frames), {}

    def step(self, action):
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid Mario action: {action}")
        self._steps += 1
        base_reward = 0.0
        terminated = False
        for _ in range(self.frame_skip):
            rgb, reward, done, info = self._env.step(int(action))
            base_reward += reward
            terminated = bool(done or info.get("flag_get", False))
            if terminated:
                break
        self._rgb = rgb.copy()
        self._frames.append(self._process(self._rgb))
        x_pos = info["x_pos"]
        delta = 0 if self._last_x is None else x_pos - self._last_x
        self._last_x = x_pos
        reward = base_reward + self.progress_weight * max(delta, 0)
        if terminated and not info.get("flag_get", False):
            reward -= self.death_penalty
        truncated = not terminated and self._steps >= self.max_steps
        if self.render_mode == "human":
            self.render()
        return np.stack(self._frames), float(reward), terminated, truncated, {
            **info,
            "base_reward": float(base_reward),
        }

    def render(self):
        if self.render_mode == "human":
            return self._env.render(mode="human")
        if self._rgb is None:
            raise RuntimeError("Call reset before rendering")
        return self._rgb.copy()

    def close(self):
        self._env.close()

