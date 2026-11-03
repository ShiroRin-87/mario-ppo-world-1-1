"""Plot episode reward against collected timesteps from an SB3 run."""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    with (args.experiment / "progress.csv").open(encoding="utf-8", newline="") as source:
        rows = list(csv.DictReader(source))
    data = [
        (int(row["time/total_timesteps"]), float(row["rollout/ep_rew_mean"]))
        for row in rows if row.get("time/total_timesteps") and row.get("rollout/ep_rew_mean")
    ]
    evaluation = [
        (int(row["time/total_timesteps"]), float(row["eval/mean_reward"]))
        for row in rows if row.get("time/total_timesteps") and row.get("eval/mean_reward")
    ]
    if not data and not evaluation:
        parser.error("No completed episodes in progress.csv")
    fig, ax = plt.subplots(figsize=(9, 4.5), layout="constrained")
    for points, label in [(data, "Training mean"), (evaluation, "Evaluation mean")]:
        if points:
            ax.plot(*zip(*points), marker="." if len(points) < 20 else None, label=label)
    ax.set(title=args.experiment.name, xlabel="Environment steps", ylabel="Mean episode reward")
    ax.legend()
    ax.grid(alpha=0.25)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
