"""Visualization suite for an online_pipeline output directory.

Loads the offline artifacts (perceiver, offline graph, LearningSystem) from the
offline_dir pointer stored in online_config.yaml, then overlays the expanded
online graph and plots online-specific diagnostics.

Usage:
    python experiments/visualize_online.py OUTPUT_DIR [options]

    python experiments/visualize_online.py /path/to/outputs/online_job123
    python experiments/visualize_online.py /path/to/run --n-obs 2000
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from omegaconf import OmegaConf

from src.learning_system import LearningSystem
from src.visualization import (
    load_pipeline_artifacts,
    plot_dataset_nodes,
    plot_online_loss_curves,
    plot_online_edge_status,
    plot_graph_networkx,
)


def main():
    parser = argparse.ArgumentParser(
        description="Visualize an online_pipeline output directory."
    )
    parser.add_argument("output_dir", type=Path, help="Path to the online pipeline output directory.")
    parser.add_argument("--n-obs",   type=int, default=1000, help="Observations for scatter plot.")
    parser.add_argument("--device",  type=str, default="cpu", help="Device for model inference.")
    args = parser.parse_args()

    output_dir = args.output_dir
    if not output_dir.is_dir():
        print(f"Error: {output_dir} is not a directory.")
        sys.exit(1)

    # Load online config to find the offline artifacts.
    online_cfg  = OmegaConf.load(output_dir / "online_config.yaml")
    offline_dir = Path(online_cfg.offline_dir)
    offline_cfg = OmegaConf.load(offline_dir / "config.yaml")

    # Reconstruct dataset and env from the offline config.
    hdf5_path = offline_cfg.dataset.get("path", None)
    if hdf5_path is not None:
        from src.robomimic_dataset import RobomimicDataset
        from src.robomimic_env import RobomimicEnvironment
        obs_keys = list(offline_cfg.dataset.obs_keys)
        dataset  = RobomimicDataset(hdf5_path, obs_keys=obs_keys)
        env      = RobomimicEnvironment(
            hdf5_path, obs_keys=obs_keys,
            **{k: v for k, v in offline_cfg.env.items()},
        )
        print(f"Loaded dataset: {hdf5_path}")
    else:
        from src.simple_gridworld import GridworldConfig, SimpleDataset, SimpleEnvironment
        gw_config = GridworldConfig()
        dataset   = SimpleDataset(gw_config)
        env       = SimpleEnvironment(dataset, gw_config)
        print("Using SimpleGridworld dataset.")

    # Load LearningSystem from the offline directory.
    ls = LearningSystem.load(
        offline_dir / "learning_system.pt",
        dataset=dataset, env=env, device=args.device,
    )

    # Load perceiver, offline graph, and node_state_pool.
    perceiver, offline_graph, node_state_pool, _ = load_pipeline_artifacts(offline_dir, ls)
    K = len(perceiver.node_obs)
    print(f"Offline artifacts loaded: K={K}  offline edges={len(offline_graph)}")

    # Load the expanded online graph topology from online_system.pt.
    online_ckpt  = torch.load(output_dir / "online_system.pt", map_location="cpu", weights_only=False)
    online_edges = online_ckpt.get("graph_edges", {})
    # Reconstruct graph dict in the same format as _rebuild_graph: (pi=None, avg=None, cost).
    online_graph = {
        eval(k): (None, None, v) for k, v in online_edges.items()
    }
    n_new = len(online_graph) - len(offline_graph)
    print(f"Online graph: {len(online_graph)} total edges  ({n_new:+d} vs offline)")

    # --- Online loss curves ---
    loss_history_path = output_dir / "online_loss_history.pt"
    if loss_history_path.exists():
        loss_history = torch.load(loss_history_path, map_location="cpu", weights_only=False)
        print(f"\nPlotting online loss curves ({len(loss_history)} steps)...")
        plot_online_loss_curves(
            loss_history,
            save_path=str(output_dir / "online_loss_curves.png"),
        )
    else:
        print("No online_loss_history.pt found — skipping loss curves.")

    # --- PCA scatter with offline + online edges highlighted ---
    print("\nPlotting dataset nodes with online edges highlighted...")
    plot_dataset_nodes(
        ls, perceiver,
        n_obs=args.n_obs,
        graph=offline_graph,
        online_graph=online_graph,
        node_state_pool=node_state_pool,
        save_path=str(output_dir / "online_dataset_nodes.png"),
    )

    print("\nPlotting online edge status heatmap...")
    plot_online_edge_status(
        K=K,
        offline_graph=offline_graph,
        online_graph=online_graph,
        save_path=str(output_dir / "online_edge_status.png"),
    )

    print("\nPlotting graph diagram...")
    plot_graph_networkx(
        K=K,
        offline_graph=offline_graph,
        online_graph=online_graph,
        save_path=str(output_dir / "online_graph.png"),
    )

    print(f"\nAll plots saved to {output_dir}")


if __name__ == "__main__":
    main()
