"""Train or resume PPO using a YAML experiment configuration."""

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from functools import partial
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import CSVOutputFormat, HumanOutputFormat, Logger, TensorBoardOutputFormat
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from agents.cnn import MarioCNN
from envs.mario import MarioEnv


class Checkpoints(BaseCallback):
    def __init__(self, folder, interval):
        super().__init__()
        self.folder = folder
        self.interval = interval

    def _on_training_start(self):
        self.next_step = (self.num_timesteps // self.interval + 1) * self.interval

    def _on_step(self):
        if self.num_timesteps >= self.next_step:
            save_model(self.model, self.folder / f"ppo_{self.num_timesteps}.zip")
            self.next_step += self.interval
        return True


def save_model(model, path):
    temporary = path.with_suffix(".tmp.zip")
    model.save(temporary)
    temporary.replace(path)


def git_value(*args):
    if shutil.which("git") is None:
        return ""
    return subprocess.run(
        ["git", *args], cwd=ROOT, text=True, capture_output=True, check=False
    ).stdout.strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--exp-id")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--no-publish", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8-sig"))
    if not isinstance(config, dict):
        parser.error("Expected a YAML mapping")
    exp_id = args.exp_id or datetime.now().strftime("exp_%Y%m%d_%H%M%S_01")
    if not re.fullmatch(r"exp_[0-9]{8}_[a-zA-Z0-9_-]+_[0-9]+", exp_id):
        parser.error("Expected exp_YYYYMMDD_label_number")
    if args.resume and not args.resume.is_file():
        parser.error(f"Missing checkpoint: {args.resume}")
    vec_cls = {"dummy": DummyVecEnv, "subprocess": SubprocVecEnv}.get(config["vec_env"])
    if vec_cls is None:
        parser.error("vec_env must be dummy or subprocess")
    n_envs = config["n_envs"]
    for key in ("n_envs", "total_timesteps", "checkpoint_steps", "eval_steps", "eval_episodes", "torch_threads"):
        if config[key] < 1:
            parser.error(f"{key} must be positive")

    folder = ROOT / "logs" / exp_id
    folder.mkdir(parents=True, exist_ok=False)
    (folder / "checkpoints").mkdir()
    shutil.copyfile(args.config, folder / "config.yaml")
    snapshot = ROOT / "SOURCE_COMMIT"
    revision = snapshot.read_text(encoding="utf-8").strip() if snapshot.is_file() else git_value("rev-parse", "HEAD")
    metadata = {
        "exp_id": exp_id,
        "seed": config["seed"],
        "git_commit": revision or None,
        "git_dirty": None if snapshot.is_file() else bool(git_value("status", "--porcelain")),
        "config": "config.yaml",
        "resume": str(args.resume) if args.resume else None,
    }
    manifest = folder / "experiment.json"
    manifest.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    torch.set_num_threads(config["torch_threads"])
    train_env = make_vec_env(
        partial(MarioEnv, **config["env"]),
        n_envs=n_envs,
        seed=config["seed"],
        vec_env_cls=vec_cls,
        monitor_dir=str(folder / "monitor"),
    )
    eval_env = None
    logger = None
    started = time.perf_counter()
    try:
        eval_env = make_vec_env(
            partial(MarioEnv, max_steps=config["env"]["max_steps"]),
            n_envs=1,
            seed=config["seed"] + 1,
        )
        if args.resume:
            model = PPO.load(args.resume, env=train_env, device=config["device"])
        else:
            model = PPO(
                "CnnPolicy",
                train_env,
                seed=config["seed"],
                device=config["device"],
                policy_kwargs={
                    "features_extractor_class": MarioCNN,
                    "features_extractor_kwargs": {"features_dim": config["features_dim"]},
                },
                **config["ppo"],
            )
        logger = Logger(str(folder), [
            HumanOutputFormat(sys.stdout),
            HumanOutputFormat(str(folder / "train.log")),
            CSVOutputFormat(str(folder / "progress.csv")),
            TensorBoardOutputFormat(str(folder / "tb")),
        ])
        model.set_logger(logger)
        logger.log(json.dumps(metadata, ensure_ascii=False))
        initial_steps = model.num_timesteps
        model.learn(
            total_timesteps=config["total_timesteps"],
            reset_num_timesteps=not args.resume,
            callback=[
                Checkpoints(folder / "checkpoints", config["checkpoint_steps"]),
                EvalCallback(
                    eval_env,
                    best_model_save_path=str(folder),
                    log_path=str(folder),
                    eval_freq=math.ceil(config["eval_steps"] / n_envs),
                    n_eval_episodes=config["eval_episodes"],
                    deterministic=True,
                ),
            ],
        )
        save_model(model, folder / "final_model.zip")
        best = folder / "best_model.zip"
        if not best.is_file():
            shutil.copyfile(folder / "final_model.zip", best)
        metadata.update(
            initial_steps=initial_steps,
            total_timesteps=model.num_timesteps,
            elapsed_seconds=round(time.perf_counter() - started, 3),
            model="best_model.zip",
        )
        manifest.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        if not args.no_publish:
            destination = ROOT / "models" / "current"
            destination.mkdir(parents=True, exist_ok=True)
            pending = destination / f".best_model.{uuid4().hex}.tmp.zip"
            shutil.copyfile(best, pending)
            os.replace(pending, destination / "best_model.zip")
        logger.log(json.dumps(metadata, ensure_ascii=False))
    finally:
        train_env.close()
        if eval_env is not None:
            eval_env.close()
        if logger is not None:
            logger.close()


if __name__ == "__main__":
    main()
