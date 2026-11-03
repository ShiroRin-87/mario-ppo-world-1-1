"""Run a deterministic 100-action smoke check against the NES backend."""

import numpy as np
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace


def main():
    raw = gym_super_mario_bros.make(
        "SuperMarioBros-1-1-v0", disable_env_checker=True
    ).unwrapped
    env = JoypadSpace(raw, SIMPLE_MOVEMENT)
    rng = np.random.default_rng(42)
    max_x = 0
    total_reward = 0.0
    try:
        obs = env.reset()
        if env.action_space.n != 7 or obs.shape != (240, 256, 3):
            raise RuntimeError("Unexpected NES action space or frame size")
        for _ in range(100):
            obs, reward, done, info = env.step(int(rng.integers(env.action_space.n)))
            if obs.dtype != np.uint8:
                raise RuntimeError(f"Unexpected NES frame dtype: {obs.dtype}")
            total_reward += reward
            max_x = max(max_x, info["x_pos"])
            if done:
                env.reset()
        frame = env.render(mode="rgb_array")
        if frame is None or frame.shape != obs.shape:
            raise RuntimeError("NES did not return an RGB frame")
        print(f"PASS steps=100 actions=7 max_x_pos={max_x} total_reward={total_reward:.1f}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
