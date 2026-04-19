"""Aggregate per-checkpoint training stats into a wide CSV.

One row per (env, residual) combination. For each checkpoint subdirectory
({env}_online_{N}steps[_residual]/step_{xxxxxx}/) the row gets a block of
columns: success rates, average steps conditional on success, standard error
of those averages, edges added, and cumulative training time.

Usage:
    python experiments/extract_stats.py [--outputs-dir PATH] [--csv OUT.csv]
"""

import argparse
import csv
import math
import re
import statistics
from pathlib import Path

import torch


DIR_RE  = re.compile(r"^(?P<env>[a-z_]+?)_online_(?P<steps>\d+)steps(?P<residual>_residual)?$")
STEP_RE = re.compile(r"^step_(\d+)$")

# Penalty step count assigned to failed rollouts when computing avg_steps.
# This makes avg_steps a unified performance metric that doesn't get confused
# by changing success rates (longer successful trials inflating the average).
FAIL_PENALTY = 1000


def parse_results_txt(path: Path) -> dict:
    out = {}
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        k, _, v = line.partition(":")
        out[k.strip()] = v.strip()
    return out


def safe_float(x, default=""):
    try:
        return float(x)
    except (ValueError, TypeError):
        return default


def stats_from_trajectories(path: Path) -> dict:
    """Compute mean/se of trajectory step counts over ALL rollouts.

    For each success type (success / env_success / graph_success), failed
    rollouts are assigned FAIL_PENALTY steps so the average stays comparable
    across runs with different success rates.

    The .pt file is a list of dicts saved by online_pipeline.py, each containing
    {episode, goal_node, success, env_success, steps, [graph_success]}.
    """
    if not path.exists():
        return {}
    data = torch.load(path, map_location="cpu", weights_only=False)
    if not data:
        return {}

    success_keys = ["success", "env_success"]
    if any("graph_success" in r for r in data):
        success_keys.append("graph_success")

    out = {}
    for key in success_keys:
        print(f"Computing stats for {path.name} success key '{key}'...")
        xs = []
        for r in data:
            actual = r.get("steps", 0)
            ok     = r.get(key, False)
            if key == 'graph_success':
                print(f"  episode {r.get('episode', '?')}, goal_node {r.get('goal_node', '?')}: {key}={ok}, steps={actual}")
            xs.append(actual if ok else FAIL_PENALTY)
        mean = statistics.mean(xs)
        sd   = statistics.stdev(xs) if len(xs) >= 2 else 0.0
        se   = sd / math.sqrt(len(xs))
        out[f"avg_steps_{key}"] = f"{mean:.1f}"
        out[f"se_steps_{key}"]  = f"{se:.2f}"
        out[f"n_{key}"]         = len(xs)
    return out


# Metrics extracted per checkpoint, in order:
SUCCESS_KEYS = [
    "n2n_success_rate",
    "n2n_env_success_rate",
    "s2s_success_rate",
    "s2s_graph_success_rate",
    "s2s_env_success_rate",
]
N2N_STEP_KEYS = ["avg_steps_success", "se_steps_success",
                 "avg_steps_env_success", "se_steps_env_success"]
S2S_STEP_KEYS = ["avg_steps_success", "se_steps_success",
                 "avg_steps_graph_success", "se_steps_graph_success",
                 "avg_steps_env_success", "se_steps_env_success"]
EXTRA_KEYS = ["edges_added", "train_time_s"]


def collect_checkpoint(step_dir: Path) -> dict:
    """Parse one step_NNNNNN subdirectory."""
    out = {}
    results = parse_results_txt(step_dir / "online_results.txt")
    if not results:
        return out

    for k in SUCCESS_KEYS:
        out[k] = results.get(k, "")
    out["train_time_s"] = results.get("train_time_s", "")
    try:
        out["edges_added"] = int(results["final_graph_edges"]) - int(results["initial_graph_edges"])
    except (KeyError, ValueError):
        out["edges_added"] = ""

    n2n_stats = stats_from_trajectories(step_dir / "n2n_trajectories.pt")
    for k in N2N_STEP_KEYS:
        out[f"n2n_{k}"] = n2n_stats.get(k, "")

    s2s_stats = stats_from_trajectories(step_dir / "s2s_trajectories.pt")
    for k in S2S_STEP_KEYS:
        out[f"s2s_{k}"] = s2s_stats.get(k, "")

    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--outputs-dir",
        default="/scratch/gpfs/TSILVER/de7281/shortcuts_from_demos/outputs",
        type=Path,
    )
    parser.add_argument(
        "--csv",
        default=Path(__file__).resolve().parent / "online_stats.csv",
        type=Path,
    )
    args = parser.parse_args()

    # rows[(env, residual)] = {step_int: checkpoint_dict}
    rows: dict[tuple[str, bool], dict[int, dict]] = {}
    all_steps: set[int] = set()

    for d in sorted(args.outputs_dir.iterdir()):
        if not d.is_dir():
            continue
        m = DIR_RE.match(d.name)
        if not m:
            continue
        last = d.name.rsplit("_", 1)[-1]
        if last.isdigit():
            continue  # jobid suffix

        env       = m.group("env")
        residual  = bool(m.group("residual"))
        key       = (env, residual)
        rows.setdefault(key, {})

        # Iterate step subdirectories in this run.
        for sub in sorted(d.iterdir()):
            if not sub.is_dir():
                continue
            sm = STEP_RE.match(sub.name)
            if not sm:
                continue
            step = int(sm.group(1))
            cp = collect_checkpoint(sub)
            if cp:
                rows[key][step] = cp
                all_steps.add(step)

    if not rows:
        print("No data found.")
        return

    sorted_steps = sorted(all_steps)
    per_step_keys = (
        SUCCESS_KEYS
        + [f"n2n_{k}" for k in N2N_STEP_KEYS]
        + [f"s2s_{k}" for k in S2S_STEP_KEYS]
        + EXTRA_KEYS
    )

    fieldnames = ["env", "residual"]
    for step in sorted_steps:
        for k in per_step_keys:
            fieldnames.append(f"step{step}_{k}")

    with open(args.csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for (env, residual) in sorted(rows.keys()):
            row = {"env": env, "residual": residual}
            for step in sorted_steps:
                cp = rows[(env, residual)].get(step, {})
                for k in per_step_keys:
                    row[f"step{step}_{k}"] = cp.get(k, "")
            writer.writerow(row)

    print(f"Wrote {len(rows)} (env, residual) rows × {len(sorted_steps)} checkpoints to {args.csv}")
    print(f"Checkpoints: {sorted_steps}")


if __name__ == "__main__":
    main()
