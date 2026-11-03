"""Evaluate checkpoint recovery from injected controller inputs in World 1-1."""

import argparse
import hashlib
import json
import sys
from pathlib import Path
from time import perf_counter

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.revision import source_revision
from train.train_route_policy import MOVEMENT_PREFIXES, RouteEnv, select_route_action


OPENING_CASES = tuple({0: prefix} for prefix in MOVEMENT_PREFIXES)
MIDCOURSE_CASES = (
    {48: (0,)}, {96: (1,)}, {144: (0, 0)}, {192: (6,)},
    {240: (3,)}, {288: (2,)}, {336: (0, 0)}, {360: (4,)},
)
HELD_OUT_CASES = (
    {0: (4, 3, 0, 6, 1)}, {0: (3,) * 8}, {0: (2,) * 6 + (0, 6)},
    {72: (0,) * 4}, {137: (6, 6, 0, 2)}, {269: (4, 0, 0, 6, 6)},
    {0: (6, 0, 3), 101: (0, 2), 251: (6, 0, 0), 374: (3,)},
    {62: (4, 4, 4, 0), 188: (6, 6, 6), 310: (0,) * 8},
)


class Checkpoint:
    """Snapshot the NES and its Python wrapper before an input override."""

    fields = (
        "_steps", "_last_x", "_rgb", "observation",
        "x", "y", "vx", "vy", "last_action", "still", "held",
    )

    def __init__(self, env):
        self.env = env
        self.nes = env._env.unwrapped
        self.nes._backup()
        self.ram = self.nes.ram.copy()
        self.nes_state = (self.nes.done, self.nes._time_last, self.nes._x_position_last)
        self.wrapper = {field: getattr(env, field) for field in self.fields}
        self.frames = tuple(env._frames)

    def restore(self):
        self.nes._restore()
        self.nes.done, self.nes._time_last, self.nes._x_position_last = self.nes_state
        for field, value in self.wrapper.items():
            setattr(self.env, field, value)
        self.env._frames.clear()
        self.env._frames.extend(self.frames)
        if not np.array_equal(self.ram, self.nes.ram):
            raise RuntimeError("NES RAM does not match the saved checkpoint")


def run_case(features, actions, overrides, recover, write_frame=None):
    """Apply real emulator inputs at the listed decision steps, then optionally restore."""

    started = perf_counter()
    events = []
    max_x = 0
    decision = 0
    trajectory = hashlib.sha256()
    with RouteEnv() as env:
        env.reset(seed=0)
        while True:
            action = select_route_action(env, features, actions)
            pulse = overrides.get(decision)
            if pulse:
                event_started = perf_counter()
                checkpoint = Checkpoint(env) if recover else None
                before = env.x
                applied = 0
                for override in pulse:
                    _, _, terminated, truncated, info = env.step(override)
                    applied += 1
                    if write_frame is not None:
                        write_frame(env.render())
                    if terminated or truncated:
                        break
                event = {
                    "at_decision": decision, "requested_action": action,
                    "injected_actions": list(pulse[:applied]),
                    "x_before": before, "x_after": env.x,
                    "ended_during_override": bool(terminated or truncated),
                }
                if checkpoint is not None:
                    checkpoint.restore()
                    event["restored_x"] = env.x
                    event["restore_seconds"] = perf_counter() - event_started
                    if write_frame is not None:
                        write_frame(env.render())
                elif terminated or truncated:
                    events.append(event)
                    max_x = max(max_x, int(info["x_pos"]))
                    break
                else:
                    action = select_route_action(env, features, actions)
                events.append(event)
            _, _, terminated, truncated, info = env.step(action)
            decision += 1
            trajectory.update(env._env.unwrapped.ram.tobytes())
            max_x = max(max_x, int(info["x_pos"]))
            if write_frame is not None:
                write_frame(env.render())
            if terminated or truncated:
                break
    return {
        "clear": bool(info["flag_get"]), "max_x_pos": max_x,
        "policy_decisions": decision, "elapsed_seconds": perf_counter() - started,
        "trajectory_sha256": trajectory.hexdigest(), "override_events": events,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--route-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with np.load(args.route_model, allow_pickle=False) as model:
        features, actions = model["features"], model["actions"]
    args.output.mkdir(parents=True, exist_ok=False)
    baseline = run_case(features, actions, {}, recover=False)
    report = {
        "source_commit": source_revision(ROOT),
        "model_sha256": hashlib.sha256(args.route_model.read_bytes()).hexdigest(),
        "standard": baseline,
        "groups": {},
    }
    for name, cases in (
        ("opening", OPENING_CASES), ("midcourse", MIDCOURSE_CASES),
        ("held_out", HELD_OUT_CASES),
    ):
        rows = []
        for overrides in cases:
            before = run_case(features, actions, overrides, recover=False)
            after = run_case(features, actions, overrides, recover=True)
            if len(after["override_events"]) != len(overrides):
                raise RuntimeError("A scheduled override was not applied")
            if after["trajectory_sha256"] != baseline["trajectory_sha256"]:
                raise RuntimeError("The restored run diverged from the standard run")
            rows.append({"overrides": overrides, "without_recovery": before, "checkpoint_recovery": after})
            print(name, overrides, before["clear"], after["clear"], flush=True)
        report["groups"][name] = rows
        (args.output / "results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
