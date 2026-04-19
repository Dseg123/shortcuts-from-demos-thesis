"""Check obs keys and action dimensions in robomimic HDF5 files.

Usage:
    python check_obs_keys.py /path/to/dir1 /path/to/dir2 ...
    python check_obs_keys.py /scratch/gpfs/TSILVER/de7281/robomimic/*/ph/
"""

import sys
from pathlib import Path
import h5py


def inspect(path: Path) -> None:
    with h5py.File(path, "r") as f:
        demos = sorted(f["data"].keys())
        demo  = f[f"data/{demos[0]}"]
        demo_keys = list(demo.keys())

        print(f"\n{'='*60}")
        print(f"File: {path}")
        print(f"Demos: {len(demos)}  (showing demo_0)")
        print(f"Demo keys: {demo_keys}")

        if "obs" in demo:
            print(f"obs keys: {sorted(demo['obs'].keys())}")
            for k in sorted(demo["obs"].keys()):
                print(f"  {k}: {demo['obs'][k].shape}")
        else:
            print("obs: <not present — raw demo, needs conversion>")

        if "actions" in demo:
            print(f"actions shape: {demo['actions'].shape}  (action_dim={demo['actions'].shape[1]})")

        if "states" in demo:
            print(f"states shape:  {demo['states'].shape}")


def main():
    paths = []
    for arg in sys.argv[1:]:
        p = Path(arg)
        if p.is_dir():
            paths.extend(sorted(p.glob("*.hdf5")))
        elif p.suffix == ".hdf5":
            paths.append(p)
        else:
            print(f"Skipping {arg} (not a .hdf5 file or directory)")

    if not paths:
        print("Usage: python check_obs_keys.py <path_or_dir> [...]")
        sys.exit(1)

    for p in paths:
        try:
            inspect(p)
        except Exception as e:
            print(f"\nERROR reading {p}: {e}")


if __name__ == "__main__":
    main()
