"""Train and compare a small ranker for Super Mario Bros. action sequences."""

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from stable_baselines3 import PPO
from torch import nn

LOOKAHEAD = 16
PREVIEW = 6
INTERVAL = 12
FIRST_PROBE = 8
ROOT = Path(__file__).resolve().parents[1]

CANDIDATES = (
    (1,) * LOOKAHEAD,
    (3,) * LOOKAHEAD,
    (2,) * 9 + (1,) * 7,
    (4,) * 9 + (3,) * 7,
    (3,) * 4 + (4,) * 9 + (3,) * 3,
    (0,) * 2 + (4,) * 9 + (3,) * 5,
)
FEATURES = len(CANDIDATES) + 11
CANDIDATE_IDS = np.eye(len(CANDIDATES), dtype=np.float32)


@dataclass
class Preview:
    features: np.ndarray
    short_utility: np.ndarray
    long_utility: np.ndarray


class Probe:
    """Maintain a separate NES instance for counterfactuals."""

    def __init__(self, env_type, seed):
        self.env = env_type()
        self.env.reset(seed=seed)
        self.joypad = self.env._env
        self.raw = self.joypad.unwrapped

    def close(self):
        self.env.close()

    def step(self, action):
        reward = 0.0
        for _ in range(self.env.frame_skip):
            _, value, done, info = self.joypad.step(action)
            reward += value
            if done or info["flag_get"]:
                return reward, True, info
        return reward, False, info

    def inspect(self, x, y, vx, labels):
        self.raw._backup()
        python_state = (self.raw.done, self.raw._time_last, self.raw._x_position_last)
        features, short, long = [], [], []
        try:
            horizon = LOOKAHEAD if labels else PREVIEW
            for index, candidate in enumerate(CANDIDATES):
                self.restore(python_state)
                score = 0.0
                peak = x
                ended = False
                flag = False
                end_x, end_y = x, y
                for step, action in enumerate(candidate[:horizon], 1):
                    if not ended:
                        value, ended, info = self.step(action)
                        score += value
                        end_x, end_y = info["x_pos"], info["y_pos"]
                        peak = max(peak, end_x)
                        flag = bool(info["flag_get"])
                    if step == PREVIEW:
                        identity = CANDIDATE_IDS[index]
                        values = [
                            x / 3200, y / 256, vx / 16,
                            (end_x - x) / 100, (peak - x) / 100, score / 100,
                            (end_y - y) / 100, float(ended), float(flag),
                            end_y / 256, index / (len(CANDIDATES) - 1),
                        ]
                        features.append(np.concatenate((identity, values)))
                        short.append(self.utility(x, end_x, peak, ended, flag))
                long.append(self.utility(x, end_x, peak, ended, flag))
        finally:
            self.restore(python_state)
        return Preview(np.array(features, dtype=np.float32), np.array(short), np.array(long))

    def restore(self, python_state):
        self.raw._restore()
        self.raw.done, self.raw._time_last, self.raw._x_position_last = python_state

    @staticmethod
    def utility(x, end_x, peak, ended, flag):
        return 2000 * flag - 300 * (ended and not flag) + end_x - x + 0.25 * (peak - x)


def ranker():
    return nn.Sequential(nn.Linear(FEATURES, 32), nn.ReLU(), nn.Linear(32, 1))


def episode(env_type, policy, seed, mode, model=None):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    examples = []
    total_reward = 0.0
    max_x = 0
    steps = 0
    decisions = 0
    action_queue = []
    info = None
    vx = 0
    started = perf_counter()
    with env_type() as env:
        obs, _ = env.reset(seed=seed)
        probe = Probe(env_type, seed) if mode != "ppo" else None
        try:
            while True:
                if probe is not None and steps >= FIRST_PROBE and (
                    (steps - FIRST_PROBE) % INTERVAL == 0
                ):
                    preview = probe.inspect(info["x_pos"], info["y_pos"], vx, labels=mode == "collect")
                    decisions += 1
                    if mode == "collect":
                        examples.append((preview.features, preview.short_utility, preview.long_utility))
                        selection = rng.random()
                        if selection < 0.65:
                            index = int(preview.long_utility.argmax())
                        elif selection < 0.85:
                            index = int(preview.short_utility.argmax())
                        else:
                            index = None
                    elif mode == "rules":
                        index = int(preview.short_utility.argmax())
                    else:
                        with torch.inference_mode():
                            index = int(model(torch.from_numpy(preview.features)).argmax())
                    if index is not None:
                        action_queue = list(CANDIDATES[index][:PREVIEW])
                if action_queue:
                    action = action_queue.pop(0)
                else:
                    predicted, _ = policy.predict(obs, deterministic=False)
                    action = int(predicted.item())
                prev_x = info["x_pos"] if info else None
                obs, reward, terminated, truncated, info = env.step(action)
                if probe is not None:
                    _, probe_done, probe_info = probe.step(action)
                    actual = (info["x_pos"], info["y_pos"], terminated, info["flag_get"])
                    mirrored = (probe_info["x_pos"], probe_info["y_pos"], probe_done, probe_info["flag_get"])
                    if actual != mirrored:
                        raise RuntimeError(f"Probe state diverged at step {steps}: {actual} != {mirrored}")
                vx = 0 if prev_x is None else info["x_pos"] - prev_x
                steps += 1
                total_reward += reward
                max_x = max(max_x, int(info["x_pos"]))
                if terminated or truncated:
                    break
        finally:
            if probe is not None:
                probe.close()
    return {
        "seed": seed, "max_x_pos": max_x, "reward": total_reward, "steps": steps,
        "clear": bool(info["flag_get"]), "lookahead_calls": decisions, "seconds": perf_counter() - started,
    }, examples


def fit(training, validation):
    x = torch.tensor(np.stack([row[0] for row in training]))
    y = torch.tensor(np.stack([row[2] for row in training]), dtype=torch.float32)
    y = ((y - y.max(dim=1, keepdim=True).values) / 35).softmax(dim=1)
    torch.manual_seed(27)
    model = ranker()
    optim = torch.optim.AdamW(model.parameters(), lr=0.003)
    for _ in range(120):
        for inds in torch.randperm(len(x)).split(64):
            logits = model(x[inds]).squeeze(-1)
            loss = -(y[inds] * logits.log_softmax(dim=1)).sum(dim=1).mean()
            optim.zero_grad()
            loss.backward()
            optim.step()
    model.eval()
    features = torch.tensor(np.stack([row[0] for row in validation]))
    scores = np.stack([row[2] for row in validation])
    rule_indices = np.stack([row[1] for row in validation]).argmax(axis=1)
    with torch.inference_mode():
        model_indices = model(features).squeeze(-1).argmax(dim=1).numpy()
    index = np.arange(len(scores))
    best = scores.max(axis=1)
    return model, {
        name: {
            "mean_regret": float((best - scores[index, indices]).mean()),
            "optimal_rate": float(np.isclose(best, scores[index, indices]).mean()),
        }
        for name, indices in [("rules", rule_indices), ("ranker", model_indices)]
    }


def evaluate(env_type, policy, mode, model, seeds):
    runs = []
    for seed in seeds:
        run, _ = episode(env_type, policy, seed, mode, model)
        runs.append(run)
        print(f"{mode} seed={seed} x={run['max_x_pos']} reward={run['reward']:.1f}", flush=True)
    return {
        "runs": runs,
        "max_x_pos": max(run["max_x_pos"] for run in runs),
        "mean_x_pos": float(np.mean([run["max_x_pos"] for run in runs])),
        "mean_reward": float(np.mean([run["reward"] for run in runs])),
        "clear_rate": float(np.mean([run["clear"] for run in runs])),
        "steps_per_second": sum(run["steps"] for run in runs) / sum(run["seconds"] for run in runs),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=ROOT)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--ranker", type=Path, help="Evaluate saved ranker weights without training")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-episodes", type=int, default=18)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--eval-seed", type=int, default=42)
    args = parser.parse_args()
    if not 1 <= args.train_episodes <= 1000 or args.eval_episodes < 1:
        parser.error("Use 1–1000 training episodes and at least one evaluation episode")
    sys.path.insert(0, str(args.repo.resolve()))
    from envs.mario import MarioEnv
    from scripts.revision import source_revision

    torch.set_num_threads(2)
    policy_path = args.model or args.repo / "models/current/best_model.zip"
    if not policy_path.is_file():
        parser.error(f"Missing PPO checkpoint: {policy_path}")
    args.output.mkdir(parents=True, exist_ok=False)
    policy = PPO.load(policy_path, device="cpu")
    if args.ranker:
        model = ranker()
        model.load_state_dict(torch.load(args.ranker, map_location="cpu", weights_only=True))
        model.eval()
        training_report = {"loaded_ranker_sha256": hashlib.sha256(args.ranker.read_bytes()).hexdigest()}
    else:
        train_seeds = range(1000, 1000 + args.train_episodes)
        validation_seeds = range(2000, 2005)
        training, validation = [], []
        for split, seeds in [(training, train_seeds), (validation, validation_seeds)]:
            for seed in seeds:
                run, samples = episode(MarioEnv, policy, seed, "collect")
                split.extend(samples)
                print(f"collect seed={seed} states={len(samples)} x={run['max_x_pos']}", flush=True)
        started = perf_counter()
        model, ranking = fit(training, validation)
        training_report = {
            "train_seeds": list(train_seeds),
            "validation_seeds": list(validation_seeds),
            "fit_seconds": perf_counter() - started,
            "train_states": len(training),
            "validation_states": len(validation),
            "ranking": ranking,
        }
        torch.save(model.state_dict(), args.output / "decision-ranker.pt")
    eval_seeds = range(args.eval_seed, args.eval_seed + args.eval_episodes)
    report = {
        "source_commit": source_revision(args.repo),
        "ppo_model_sha256": hashlib.sha256(policy_path.read_bytes()).hexdigest(),
        "evaluation_seeds": list(eval_seeds),
        "candidates": len(CANDIDATES),
        "parameters": sum(p.numel() for p in model.parameters()),
        **training_report,
        "results": {},
    }
    for mode in ("ppo", "rules", "ranker"):
        report["results"][mode] = evaluate(MarioEnv, policy, mode, model, eval_seeds)
        (args.output / "results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: {k: v for k, v in value.items() if k != "runs"}
                      for key, value in report["results"].items()}, indent=2))


if __name__ == "__main__":
    main()
