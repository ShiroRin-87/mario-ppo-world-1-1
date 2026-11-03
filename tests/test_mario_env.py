import unittest

import numpy as np
from gymnasium.utils.env_checker import check_env

from envs.mario import MarioEnv


class MarioEnvironmentTests(unittest.TestCase):
    def setUp(self):
        self.env = MarioEnv(max_steps=10)
        self.addCleanup(self.env.close)

    def test_gymnasium_contract(self):
        check_env(self.env, skip_render_check=True)

    def test_observation_and_render(self):
        observation, _ = self.env.reset(seed=42)
        self.assertEqual(observation.shape, (4, 84, 84))
        self.assertEqual(observation.dtype, np.uint8)
        np.testing.assert_array_equal(observation[0], observation[3])
        self.assertEqual(self.env.render().shape, (240, 256, 3))

    def test_actions(self):
        self.env.reset(seed=42)
        self.assertEqual(self.env.action_space.n, 7)
        for action in range(7):
            observation, _, _, _, info = self.env.step(action)
            self.assertTrue(self.env.observation_space.contains(observation))
            self.assertIn("x_pos", info)
        with self.assertRaises(ValueError):
            self.env.step(7)

    def test_step_limit_truncates(self):
        self.env.reset(seed=42)
        for _ in range(10):
            _, _, terminated, truncated, _ = self.env.step(0)
        self.assertFalse(terminated)
        self.assertTrue(truncated)

    def test_render_owns_frame(self):
        self.env.reset(seed=42)
        self.env.render().fill(0)
        self.assertTrue(self.env.render().any())

