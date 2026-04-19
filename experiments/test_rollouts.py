"""Quick rollout stress test: loads a saved learning_system.pt and runs
repeated rollouts to reproduce OOM / exit-137 failures.

Usage:
    python experiments/test_rollouts.py /path/to/pipeline_jobXXX [--num-rollouts 50] [--max-steps 200]
"""

import argparse
import os
import sys
import tracemalloc
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from omegaconf import OmegaConf

from src.learning_system import LearningSystem


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--num-rollouts", type=int, default=50)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.output_dir / "config.yaml")
    hdf5_path = cfg.dataset.get("path", None)

    if hdf5_path is not None:
        from src.robomimic_dataset import RobomimicDataset
        from src.robomimic_env import RobomimicEnvironment
        obs_keys = list(cfg.dataset.obs_keys)
        dataset = RobomimicDataset(hdf5_path, obs_keys=obs_keys)
        env = RobomimicEnvironment(
            hdf5_path, obs_keys=obs_keys,
            **{k: v for k, v in cfg.env.items()},
        )
    else:
        from src.simple_gridworld import GridworldConfig, SimpleDataset, SimpleEnvironment
        gw_config = GridworldConfig()
        dataset = SimpleDataset(gw_config)
        env = SimpleEnvironment(dataset, gw_config)

    ls = LearningSystem.load(
        args.output_dir / "learning_system.pt",
        dataset=dataset, env=env, device=args.device,
    )
    print(f"Loaded LearningSystem (step {ls.training_step})")

    tracemalloc.start()
    pid = os.getpid()

    for i in range(args.num_rollouts):
        result = ls.rollout(max_steps=args.max_steps)

        # Memory diagnostics
        current, peak = tracemalloc.get_traced_memory()
        rss_kb = 0
        try:
            with open(f"/proc/{pid}/status") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        rss_kb = int(line.split()[1])
                        break
        except FileNotFoundError:
            pass

        print(
            f"rollout {i+1:3d}/{args.num_rollouts}  "
            f"success={result['success']}  steps={len(result['actions'])}  "
            f"frames={len(result['frames'])}  "
            f"traced={current / 1e6:.1f}MB  peak={peak / 1e6:.1f}MB  "
            f"RSS={rss_kb / 1024:.0f}MB"
        )

    tracemalloc.stop()
    print("\nDone — no crash.")


if __name__ == "__main__":
    main()
