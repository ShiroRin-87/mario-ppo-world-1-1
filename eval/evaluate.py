"""Evaluate a Mario PPO policy or a seeded uniform random baseline."""

import argparse
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import psutil
import yaml

from envs.mario import MarioEnv
from scripts.revision import source_revision


def run_episode(env, policy, seed, write_frame=None, stochastic=False):
    obs, _ = env.reset(seed=seed)
    rng = np.random.default_rng(seed)
    if stochastic and policy is not None:
        import torch

        torch.manual_seed(seed)
    reward = 0.0
    max_x = 0
    steps = 0
    while True:
        if policy is None:
            action = int(rng.integers(env.action_space.n))
        else:
            prediction, _ = policy.predict(obs, deterministic=not stochastic)
            action = int(prediction.item())
        obs, value, terminated, truncated, info = env.step(action)
        max_x = max(max_x, int(info["x_pos"]))
        reward += value
        steps += 1
        if write_frame is not None:
            write_frame(env.render().tobytes())
        if terminated or truncated:
            return {
                "seed": seed,
                "reward": reward,
                "max_x_pos": max_x,
                "cleared": bool(info.get("flag_get")),
                "steps": steps,
                "ended_by": "terminated" if terminated else "truncated",
            }


def record_episode(env, policy, seed, target, stochastic=False):
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("FFmpeg is required for --video")
    target.parent.mkdir(parents=True, exist_ok=True)
    pending = target.with_name(f".{target.stem}.{uuid4().hex}.tmp.mp4")
    command = [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "256x240",
        "-r", str(env.metadata["render_fps"] / env.frame_skip), "-i", "-",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(pending),
    ]
    try:
        with subprocess.Popen(command, stdin=subprocess.PIPE) as process:
            run_episode(env, policy, seed, process.stdin.write, stochastic=stochastic)
            process.stdin.close()
            if process.wait() != 0:
                raise RuntimeError("FFmpeg could not encode the replay")
        pending.replace(target)
    finally:
        pending.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--model", type=Path)
    source.add_argument("--random", action="store_true")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--exp-id")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "baseline.yaml")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--video", type=Path)
    parser.add_argument("--stochastic", action="store_true")
    args = parser.parse_args()
    label = "random" if args.random else args.model.stem
    exp_id = args.exp_id or datetime.now().strftime(f"exp_%Y%m%d_{label}_%H%M%S")
    if args.episodes < 1:
        parser.error("--episodes must be positive")
    if args.model and not args.model.is_file():
        parser.error(f"Model does not exist: {args.model}")
    config = yaml.safe_load(args.config.read_text(encoding="utf-8-sig"))
    seed = args.seed if args.seed is not None else config["seed"]
    policy = None
    if args.model:
        import torch
        from stable_baselines3 import PPO

        torch.set_num_threads(config["torch_threads"])
        policy = PPO.load(args.model, device="cpu")
    env = MarioEnv(max_steps=config["env"]["max_steps"])
    process = psutil.Process()
    max_rss = 0
    try:
        started = time.perf_counter()
        results = []
        for index in range(args.episodes):
            results.append(run_episode(env, policy, seed + index, stochastic=args.stochastic))
            memory = process.memory_info()
            max_rss = max(max_rss, getattr(memory, "peak_wset", memory.rss))
        elapsed = time.perf_counter() - started

        rewards = [episode["reward"] for episode in results]
        total_steps = sum(episode["steps"] for episode in results)
        result = {
            "exp_id": exp_id,
            "seed": seed,
            "git_commit": source_revision(ROOT),
            "episodes": args.episodes,
            "mean_reward": float(np.mean(rewards)),
            "std_reward": float(np.std(rewards)),
            "max_x_pos": max(episode["max_x_pos"] for episode in results),
            "clear_rate": sum(episode["cleared"] for episode in results) / args.episodes,
            "fps": total_steps / elapsed,
            "model_path": str(args.model) if args.model else None,
        }
        output = json.dumps(result, indent=2)
        path = args.output or ROOT / "logs" / exp_id / "evaluation.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(output + "\n", encoding="utf-8")
        detail = {
            "seconds": elapsed,
            "decision_mode": "random" if policy is None else ("sampled" if args.stochastic else "greedy"),
            "evaluation_steps": total_steps,
            "peak_rss_bytes": max_rss,
            "episodes": results,
        }
        path.with_suffix(".details.json").write_text(
            json.dumps(detail, indent=2) + "\n", encoding="utf-8"
        )
        print(output)
        if args.video:
            best = max(results, key=lambda episode: episode["max_x_pos"])
            record_episode(env, policy, best["seed"], args.video, stochastic=args.stochastic)
    finally:
        env.close()


if __name__ == "__main__":
    main()
