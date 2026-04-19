"""Hydra pipeline for offline CRL on robomimic datasets.

Constructs a RobomimicDataset, RobomimicEnvironment, and LearningSystem from
a Hydra config, trains, then evaluates with rollouts.

Run locally (unit test config):
    python experiments/pipeline.py

Override individual values:
    python experiments/pipeline.py training.num_steps=500

Use a different config:
    python experiments/pipeline.py --config-name lift_ph_lowdim
"""

import sys
import time
from dataclasses import fields
from pathlib import Path

import torch
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.learning_system import LearningSystem, LearningSystemConfig
from src.robomimic_dataset import RobomimicDataset
from src.robomimic_env import RobomimicEnvironment
from src.simple_gridworld import GridworldConfig, SimpleDataset, SimpleEnvironment
from src.visualization import save_pipeline_artifacts


def dataclass_from_cfg(dataclass_type, cfg_section):
    """Instantiate a dataclass from a Hydra config section, ignoring unknown keys."""
    field_names = {f.name for f in fields(dataclass_type)}
    kwargs = {k: v for k, v in cfg_section.items() if k in field_names}
    return dataclass_type(**kwargs)


@hydra.main(version_base=None, config_path="configs", config_name="unit_test")
def main(cfg: DictConfig) -> float:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)

    print("=" * 80)
    print("Offline CRL Pipeline — robomimic")
    print("=" * 80)
    print(OmegaConf.to_yaml(cfg))
    print("=" * 80)

    output_dir = Path(HydraConfig.get().runtime.output_dir)
    hdf5_path = cfg.dataset.get("path", None)

    print("Path:", hdf5_path)

    # --- Dataset & Environment ---
    if hdf5_path is None:
        # No HDF5 path provided — fall back to simple gridworld with default config.
        gw_config = GridworldConfig()
        dataset   = SimpleDataset(gw_config)
        env       = SimpleEnvironment(dataset, gw_config)
        obs_keys  = None
        print("No dataset path provided — using SimpleGridworld with default config.")
        print(f"action_dim: {dataset.action_space.shape[0]}")
    else:
        obs_keys = list(cfg.dataset.obs_keys)
        dataset  = RobomimicDataset(hdf5_path=hdf5_path, obs_keys=obs_keys)
        print(f"Dataset loaded: {hdf5_path}")
        print(f"obs_keys: {obs_keys}  |  action_dim: {dataset.action_space.shape[0]}")
        env = RobomimicEnvironment(
            hdf5_path=hdf5_path,
            obs_keys=obs_keys,
            **{k: v for k, v in cfg.env.items()},
        )
    print("Environment created.")

    # --- LearningSystem ---
    load_from = cfg.learning_system.load_from
    if load_from is not None:
        system = LearningSystem.load(
            Path(load_from), dataset=dataset, env=env,
            device=cfg.learning_system.get("device", None),
        )
        print(f"LearningSystem loaded from {load_from} (step {system.training_step})")
        train_time = 0.0
    else:
        ls_config = dataclass_from_cfg(LearningSystemConfig, cfg.learning_system)
        system = LearningSystem(dataset=dataset, env=env, config=ls_config)
        print(f"LearningSystem ready on device: {system.device}")

        # --- Training ---
        print(f"\nTraining for {cfg.learning_system.num_steps} steps...")
        t0 = time.time()
        loss_history = system.train(
            num_steps=cfg.learning_system.num_steps,
            log_every=cfg.learning_system.log_every,
        )
        train_time = time.time() - t0
        print(f"Training complete in {train_time:.1f}s.")
        system.save(output_dir / "learning_system.pt")
        torch.save(loss_history, output_dir / "loss_history.pt")

    # --- Evaluation ---
    print(f"\nRunning {cfg.eval.num_rollouts} evaluation rollouts...")
    t0 = time.time()
    successes = 0
    for i in range(cfg.eval.num_rollouts):
        result = system.rollout(max_steps=cfg.eval.max_steps)
        successes += int(result['success'])
        print(f"  rollout {i+1}/{cfg.eval.num_rollouts} — "
              f"success={result['success']}  steps={len(result['actions'])}")
    eval_time = time.time() - t0

    success_rate = successes / cfg.eval.num_rollouts
    print(f"\nSuccess rate: {successes}/{cfg.eval.num_rollouts} = {success_rate:.2%}")

    # --- Perceiver & Graph ---
    print(f"\nBuilding perceiver (K={cfg.perceiver.K})...")
    t0 = time.time()
    perceiver = system.create_perceiver(K=cfg.perceiver.K, max_obs=cfg.perceiver.max_obs)
    print(f"\nBuilding graph...")
    graph, node_state_pool, edge_stats = system.create_graph(
        perceiver, env,
        num_rollouts=cfg.perceiver.num_rollouts,
        max_steps=cfg.perceiver.max_steps,
        success_threshold=cfg.perceiver.success_threshold,
    )
    perceiver_time = time.time() - t0
    print(f"Perceiver + graph built in {perceiver_time:.1f}s  ({len(graph)} edges).")
    save_pipeline_artifacts(output_dir, perceiver, graph, node_state_pool, edge_stats)

    # --- Save ---
    OmegaConf.save(cfg, output_dir / "config.yaml")
    with open(output_dir / "results.txt", "w") as f:
        f.write(f"dataset: {hdf5_path or 'simple_gridworld'}\n")
        f.write(f"obs_keys: {obs_keys}\n")
        f.write(f"num_steps: {cfg.learning_system.num_steps}\n")
        f.write(f"num_rollouts: {cfg.eval.num_rollouts}\n")
        f.write(f"successes: {successes}\n")
        f.write(f"success_rate: {success_rate:.4f}\n")
        f.write(f"train_time_s: {train_time:.1f}\n")
        f.write(f"eval_time_s: {eval_time:.1f}\n")
        f.write(f"perceiver_K: {cfg.perceiver.K}\n")
        f.write(f"graph_edges: {len(graph)}\n")
        f.write(f"perceiver_time_s: {perceiver_time:.1f}\n")
    print(f"Results saved to {output_dir}")

    # env.close()
    return success_rate


if __name__ == "__main__":
    main()
