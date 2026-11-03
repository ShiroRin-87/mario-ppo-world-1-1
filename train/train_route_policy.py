"""Fit and evaluate a World 1-1 controller from a successful demonstration."""

import argparse
import hashlib
import json
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from stable_baselines3 import PPO

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from envs.mario import MarioEnv
from scripts.revision import source_revision
from train.train_decision_ranker import episode, ranker

TEACHER_SEED = 42
WAIT_OFFSETS = (0, 1, 3, 5, 7, 9, 11, 13, 17, 23, 31)
MOVEMENT_PREFIXES = ((1, 0, 1), (3, 0), (2,), (6, 0, 1))


class RouteEnv(MarioEnv):
    def reset(self, **kwargs):
        observation, info = super().reset(**kwargs)
        self.observation = observation
        self.x, self.y = self.position()
        self.vx = self.vy = self.last_action = self.still = self.held = 0
        return observation, info

    def position(self):
        raw = self._env.unwrapped
        return int(raw._x_position), int(raw._y_position)

    def features(self):
        raw = self._env.unwrapped
        scene = self.observation[-1, 14:82:4, ::4].astype(np.float32).flatten() / 255
        pose = np.array([
            self.x / 12, self.y / 7, self.vx / 4, self.vy / 5,
            self.last_action / 6, raw._player_state / 4,
            min(self.still, 64) / 6, min(self.held, 32) / 4,
        ], dtype=np.float32)
        return np.concatenate((pose, scene * 0.32))

    def step(self, action):
        observation, reward, terminated, truncated, info = super().step(action)
        x, y = self.position()
        self.vx, self.vy = x - self.x, y - self.y
        self.still = self.still + 1 if x == self.x else 0
        self.held = self.held + 1 if action == self.last_action else 1
        self.x, self.y = x, y
        self.observation, self.last_action = observation, action
        return observation, reward, terminated, truncated, info


def collect_demonstration(ppo_path, ranker_path):
    policy = PPO.load(ppo_path, device="cpu")
    model = ranker()
    model.load_state_dict(torch.load(ranker_path, map_location="cpu", weights_only=True))
    model.eval()
    features, actions = [], []

    class DemonstrationEnv(RouteEnv):
        def step(self, action):
            features.append(self.features())
            actions.append(action)
            return super().step(action)

    result, _ = episode(DemonstrationEnv, policy, TEACHER_SEED, "ranker", model)
    if not result["clear"]:
        raise RuntimeError("Teacher did not reach the World 1-1 flag")
    return np.stack(features), np.array(actions, dtype=np.int8), result


def run_route(features, actions, prefix=(), write_frame=None):
    started = perf_counter()
    with RouteEnv() as env:
        env.reset(seed=0)
        for action in prefix:
            _, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                raise ValueError("Opening prefix ended the episode")
            if write_frame is not None:
                write_frame(env.render())
        env.still = env.held = 0
        start_x = env.x
        start_digest = hashlib.sha256(env._env.unwrapped.ram.tobytes()).hexdigest()
        reward = 0.0
        max_x = start_x
        steps = 0
        while True:
            difference = features - env.features()
            distances = np.einsum("ij,ij->i", difference, difference)
            action = int(actions[distances.argmin()])
            _, value, terminated, truncated, info = env.step(action)
            reward += value
            steps += 1
            max_x = max(max_x, int(info["x_pos"]))
            if write_frame is not None:
                write_frame(env.render())
            if terminated or truncated:
                break
    return {
        "prefix": list(prefix), "start_x": start_x, "start_ram_sha256": start_digest,
        "clear": bool(info["flag_get"]), "max_x_pos": max_x, "reward_after_prefix": reward,
        "steps_after_prefix": steps, "seconds": perf_counter() - started,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--ranker", type=Path, help="Fit from a successful ranker demonstration")
    source.add_argument("--route-model", type=Path, help="Evaluate a saved route model")
    parser.add_argument("--ppo", type=Path, default=ROOT / "models/current/best_model.zip")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"source_commit": source_revision(ROOT)}
    if args.ranker:
        features, actions, teacher = collect_demonstration(args.ppo, args.ranker)
        report["teacher"] = teacher
        model_path = args.output / "world-1-1-route.npz"
        np.savez_compressed(model_path, features=features, actions=actions)
    else:
        model_path = args.route_model
        with np.load(model_path, allow_pickle=False) as model:
            features, actions = model["features"], model["actions"]
    report.update({
        "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
        "examples": len(actions),
        "feature_count": features.shape[1],
        "cases": {},
    })
    suites = {
        "standard": [()],
        "wait_only": [(0,) * steps for steps in WAIT_OFFSETS if steps],
        "movement": MOVEMENT_PREFIXES,
    }
    for name, prefixes in suites.items():
        runs = []
        for prefix in prefixes:
            result = run_route(features, actions, prefix)
            runs.append(result)
            print(name, json.dumps(result), flush=True)
        report["cases"][name] = {
            "clear_count": sum(run["clear"] for run in runs),
            "runs": runs,
        }
    (args.output / "results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
