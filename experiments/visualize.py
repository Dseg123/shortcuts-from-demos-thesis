"""Visualization suite for a pipeline output directory.

Loads a trained LearningSystem together with its perceiver and graph artifacts,
then generates the full set of diagnostic plots and saves them alongside the
other output files.

Usage:
    python experiments/visualize.py OUTPUT_DIR [options]

    python experiments/visualize.py /path/to/outputs/2026-03-26/12-00-00
    python experiments/visualize.py /path/to/run --n-obs 2000 --n-samples 8
    python experiments/visualize.py /path/to/run --no-rollout
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from omegaconf import OmegaConf

from src.learning_system import LearningSystem
from src.visualization import (
    load_pipeline_artifacts,
    plot_dataset_nodes,
    plot_demo_nodes,
    plot_edge_success_rates,
    plot_loss_curves,
    plot_node_samples,
    plot_rollout_nodes,
)


def main():
    parser = argparse.ArgumentParser(description="Visualize a pipeline output directory.")
    parser.add_argument("output_dir", type=Path, help="Path to the pipeline output directory.")
    parser.add_argument("--n-obs",     type=int, default=1000, help="Observations for scatter plot.")
    parser.add_argument("--n-samples", type=int, default=5,    help="Samples per node in node grid.")
    parser.add_argument("--no-rollout", action="store_true",   help="Skip rollout visualization.")
    parser.add_argument("--num-rollouts", type=int, default=5, help="Number of rollouts to visualize.")
    parser.add_argument("--episode",   type=int, default=0,    help="Demo episode index to plot.")
    parser.add_argument("--device",    type=str, default="cpu", help="Device for model inference (default: cpu).")
    args = parser.parse_args()

    output_dir = args.output_dir
    if not output_dir.is_dir():
        print(f"Error: {output_dir} is not a directory.")
        sys.exit(1)

    # --- Load config & reconstruct dataset / env ---
    cfg = OmegaConf.load(output_dir / "config.yaml")
    hdf5_path = cfg.dataset.get("path", None)

    if hdf5_path is not None:
        from src.robomimic_dataset import RobomimicDataset
        from src.robomimic_env import RobomimicEnvironment
        obs_keys = list(cfg.dataset.obs_keys)
        dataset  = RobomimicDataset(hdf5_path, obs_keys=obs_keys)
        env      = RobomimicEnvironment(
            hdf5_path, obs_keys=obs_keys,
            **{k: v for k, v in cfg.env.items()},
        )
        print(f"Loaded dataset: {hdf5_path}")
    else:
        from src.simple_gridworld import GridworldConfig, SimpleDataset, SimpleEnvironment
        gw_config = GridworldConfig()
        dataset   = SimpleDataset(gw_config)
        env       = SimpleEnvironment(dataset, gw_config)
        print("Using SimpleGridworld dataset.")

    # --- Load LearningSystem ---
    ls = LearningSystem.load(output_dir / "learning_system.pt", dataset=dataset, env=env, device=args.device)

    # --- Load perceiver / graph / node_state_pool ---
    perceiver, graph, node_state_pool, edge_stats = load_pipeline_artifacts(output_dir, ls)
    K = len(perceiver.node_obs)
    print(f"Perceiver loaded: K={K}  |  graph edges: {len(graph)}")

    # --- Generate plots ---
    loss_history_path = output_dir / "loss_history.pt"
    if loss_history_path.exists():
        import torch as _t
        loss_history = _t.load(loss_history_path, map_location='cpu', weights_only=False)
        print("\nPlotting loss curves...")
        plot_loss_curves(loss_history, save_path=str(output_dir / "loss_curves.png"))

    if edge_stats is not None:
        print("\nPlotting edge success rates...")
        plot_edge_success_rates(
            edge_stats,
            success_threshold=cfg.perceiver.success_threshold,
            save_path=str(output_dir / "edge_success_rates.png"),
        )
    print("\nPlotting dataset nodes...")
    pca = plot_dataset_nodes(
        ls, perceiver,
        n_obs=args.n_obs,
        graph=graph,
        node_state_pool=node_state_pool,
        save_path=str(output_dir / "dataset_nodes.png"),
    )

    print("\nPlotting node samples...")
    plot_node_samples(
        ls, perceiver,
        n_samples=args.n_samples,
        node_state_pool=node_state_pool,
        env=env,
        save_path=str(output_dir / "node_samples.png"),
    )

    print(f"\nPlotting demo #{args.episode}...")
    plot_demo_nodes(
        ls, perceiver,
        episode_idx=args.episode,
        pca=pca,
        save_path=str(output_dir / "demo_nodes.png"),
    )

    if not args.no_rollout:
        from PIL import Image

        num_vis_rollouts = args.num_rollouts
        print(f"\nRunning {num_vis_rollouts} rollouts...")
        for i in range(num_vis_rollouts):
            result = ls.rollout(max_steps=cfg.eval.max_steps)
            label = "ok" if result['success'] else "fail"
            print(f"  rollout {i+1}/{num_vis_rollouts}: "
                  f"success={result['success']}  steps={len(result['actions'])}")

            if result['frames']:
                frames = [Image.fromarray(f) for f in result['frames']]
                gif_path = output_dir / f"rollout_{i:03d}_{label}.gif"
                frames[0].save(
                    gif_path, save_all=True, append_images=frames[1:], duration=50, loop=0,
                )
                print(f"    saved {gif_path.name}")

            plot_rollout_nodes(
                ls, perceiver,
                result,
                pca=pca,
                save_path=str(output_dir / f"rollout_{i:03d}_{label}_nodes.png"),
            )

    print(f"\nAll plots saved to {output_dir}")


if __name__ == "__main__":
    main()
