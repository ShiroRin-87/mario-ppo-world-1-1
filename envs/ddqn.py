"""Double DQN observations on the project's World 1-1 emulator."""

import cv2
import numpy as np

from envs.mario import MarioEnv


class DDQNMarioEnv(MarioEnv):
    @staticmethod
    def _process(rgb):
        # Match the checkpoint's Gymnasium grayscale and area resize.
        gray = np.sum(rgb * np.array([0.2125, 0.7154, 0.0721]), axis=-1).astype(np.uint8)
        return cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
