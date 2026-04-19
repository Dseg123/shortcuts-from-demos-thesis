"""Check whether any demo actions fall outside [-1, 1].

Usage:
    python check_action_bounds.py /path/to/low_dim.hdf5
"""

import sys
from pathlib import Path

import h5py
import numpy as np


def main():
    if len(sys.argv) < 2:
        print("Usage: python check_action_bounds.py <hdf5_path>")
        sys.exit(1)

    path = Path(sys.argv[1])
    with h5py.File(path, "r") as f:
        demos = sorted(f["data"].keys())
        global_min = np.inf
        global_max = -np.inf
        violations = 0
        total_steps = 0

        for key in demos:
            actions = f[f"data/{key}/actions"][:]
            total_steps += len(actions)
            global_min = min(global_min, actions.min())
            global_max = max(global_max, actions.max())
            n_outside = np.sum(np.abs(actions) > 1.0)
            if n_outside > 0:
                violations += n_outside
                print(f"  {key}: {n_outside} values outside [-1, 1]  "
                      f"min={actions.min():.4f} max={actions.max():.4f}")

    print(f"\n{path.name}: {len(demos)} demos, {total_steps} steps")
    print(f"  global min: {global_min:.6f}")
    print(f"  global max: {global_max:.6f}")
    if violations:
        print(f"  {violations} total values outside [-1, 1]")
    else:
        print("  All actions within [-1, 1]")


if __name__ == "__main__":
    main()
