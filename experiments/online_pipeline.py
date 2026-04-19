"""Online CRL pipeline: loads offline artifacts and fine-tunes with live environment interaction.

Requires a completed offline pipeline output directory (containing learning_system.pt,
config.yaml, and the perceiver/graph artifacts saved by pipeline.py).

Run with an explicit offline_dir override:
    python experiments/online_pipeline.py --config-name lift_ph_lowdim \\
        offline_dir=/scratch/gpfs/.../outputs/pipeline_job6100692

Override individual values:
    python experiments/online_pipeline.py --config-name lift_ph_lowdim \\
        offline_dir=... online_system.num_steps=5000 online_system.residualize=true
"""

import random
import sys
import time
from dataclasses import fields
from pathlib import Path

import torch
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.learning_system import LearningSystem
from src.online_system import OnlineSystem, OnlineCRLConfig
from src.robomimic_dataset import RobomimicDataset
from src.robomimic_env import RobomimicEnvironment
from src.simple_gridworld import GridworldConfig, SimpleDataset, SimpleEnvironment
from src.visualization import load_pipeline_artifacts


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
    print("Online CRL Pipeline")
    print("=" * 80)
    print(OmegaConf.to_yaml(cfg))
    print("=" * 80)

    output_dir  = Path(HydraConfig.get().runtime.output_dir)
    offline_dir = Path(cfg.offline_dir)

    # ── Reconstruct dataset & env from the offline config ──────────────────
    offline_cfg = OmegaConf.load(offline_dir / "config.yaml")
    hdf5_path   = offline_cfg.dataset.get("path", None)

    if hdf5_path is None:
        gw_config = GridworldConfig()
        dataset   = SimpleDataset(gw_config)
        env       = SimpleEnvironment(dataset, gw_config)
        print("Using SimpleGridworld dataset.")
    else:
        obs_keys = list(offline_cfg.dataset.obs_keys)
        dataset  = RobomimicDataset(hdf5_path=hdf5_path, obs_keys=obs_keys)
        env      = RobomimicEnvironment(
            hdf5_path=hdf5_path,
            obs_keys=obs_keys,
            **{k: v for k, v in offline_cfg.env.items()},
        )
        print(f"Dataset loaded: {hdf5_path}")
    print("Environment ready.")

    # ── Load LearningSystem and pipeline artifacts ─────────────────────────
    ls = LearningSystem.load(
        offline_dir / "learning_system.pt",
        dataset=dataset,
        env=env,
        device="cpu",   # artifacts are small; keep on CPU until OnlineSystem moves them
    )
    print(f"LearningSystem loaded (step {ls.training_step}).")

    perceiver, graph, node_state_pool, edge_stats = load_pipeline_artifacts(offline_dir, ls)
    print(f"Artifacts loaded: K={len(perceiver.node_obs)} nodes, {len(graph)} graph edges.")

    # ── Build OnlineSystem ─────────────────────────────────────────────────
    online_cfg = dataclass_from_cfg(OnlineCRLConfig, cfg.online_system)
    online_sys = OnlineSystem(
        ls=ls,
        perceiver=perceiver,
        graph=graph,
        node_state_pool=node_state_pool,
        env=env,
        config=online_cfg,
    )
    print(f"OnlineSystem ready on device: {online_sys.device}")
    print(f"  sampling_method={online_cfg.sampling_method}  "
          f"residualize={online_cfg.residualize}  "
          f"num_steps={online_cfg.num_steps}")

    # ── Eval-every setup ────────────────────────────────────────────────────
    from src.visualization import (
        plot_dataset_nodes, plot_online_loss_curves,
        plot_online_edge_status, plot_graph_networkx, plot_node_distances,
    )
    from PIL import Image, ImageDraw

    K                = online_sys.K
    num_rollouts     = cfg.online_eval.num_rollouts
    max_edge_steps   = cfg.online_eval.get("max_edge_steps",   online_cfg.max_steps)
    max_last_mile    = cfg.online_eval.get("max_last_mile_steps", 500)
    render           = cfg.online_eval.get("render", True)
    replay_mode      = cfg.online_eval.get("replay_mode", "live")
    rollout_method   = cfg.online_eval.get("rollout_method", "retries")
    max_retries      = cfg.online_eval.get("max_retries", 10)
    eval_every       = cfg.online_eval.get("eval_every", online_cfg.num_steps or 1)

    def _annotate_frames(frames_list, node_labels, goal_node=None):
        """Overlay node label and goal info on each frame."""
        annotated = []
        for i, frame in enumerate(frames_list):
            img = Image.fromarray(frame).convert("RGB")
            draw = ImageDraw.Draw(img)
            label = f"node {node_labels[i]}" if i < len(node_labels) else "?"
            if goal_node is not None:
                label += f"  goal={goal_node}"
            draw.text((4, 4), label, fill="white")
            # Draw black outline for readability
            draw.text((3, 3), label, fill="black")
            draw.text((5, 5), label, fill="black")
            draw.text((4, 4), label, fill="white")
            annotated.append(img)
        return annotated

    def _save_gif(frames_list, path, node_labels=None, goal_node=None):
        if frames_list:
            if node_labels:
                pil = _annotate_frames(frames_list, node_labels, goal_node)
            else:
                pil = [Image.fromarray(f) for f in frames_list]
            pil[0].save(path, save_all=True, append_images=pil[1:], duration=50, loop=0)
            print(f"    saved {path.name}")

    def run_full_evaluation(eval_dir: Path, loss_history, train_time_so_far: float):
        """Generate all visualizations + run all evaluations into eval_dir.

        Uses closure over outer variables (online_sys, ls, perceiver, graph, env,
        config, etc.) so it can be called at multiple training checkpoints.
        """
        eval_dir.mkdir(parents=True, exist_ok=True)
        rollouts_dir = eval_dir / "rollouts"
        rollouts_dir.mkdir(exist_ok=True)

        # ---- Visualizations ---------------------------------------------
        if loss_history:
            print(f"\nPlotting online loss curves ({len(loss_history)} steps)...")
            plot_online_loss_curves(
                loss_history,
                save_path=str(eval_dir / "online_loss_curves.png"),
            )

        online_graph_vis  = {k: (None, None, v[2]) for k, v in online_sys.graph.items()}
        offline_graph_vis = {k: (None, None, v[2]) for k, v in graph.items()}
        print("\nPlotting dataset nodes with online edges highlighted...")
        plot_dataset_nodes(
            ls, perceiver,
            graph=offline_graph_vis,
            online_graph=online_graph_vis,
            node_state_pool=node_state_pool,
            save_path=str(eval_dir / "online_dataset_nodes.png"),
        )

        print("\nPlotting online edge status heatmap...")
        plot_online_edge_status(
            K=online_sys.K,
            offline_graph=graph,
            online_graph=online_sys.graph,
            save_path=str(eval_dir / "online_edge_status.png"),
        )

        print("\nComputing fresh node distances...")
        fresh_dists = online_sys.compute_fresh_distances()
        plot_node_distances(
            fresh_dists,
            title='Fresh CMD distances (current d_net)',
            save_path=str(eval_dir / "online_node_distances.png"),
        )

        print("\nPlotting graph diagram...")
        plot_graph_networkx(
            K=online_sys.K,
            offline_graph=graph,
            online_graph=online_sys.graph,
            save_path=str(eval_dir / "online_graph.png"),
        )

        # ---- 1. Node-to-node ---------------------------------------------
        print(f"\n{'='*60}")
        print(f"Node-to-node evaluation ({num_rollouts} rollouts)")
        print(f"{'='*60}")
        t_eval = time.time()
        n2n_successes   = 0
        n2n_env_success = 0
        n2n_data        = []

        for i in range(num_rollouts):
            goal_node = random.randrange(K)
            if rollout_method == 'retries':
                result = online_sys.rollout_with_retries(
                    goal_node=goal_node, max_edge_steps=max_edge_steps,
                    max_retries=max_retries, render=render,
                )
            else:
                result = online_sys.rollout(
                    goal_node=goal_node, max_edge_steps=max_edge_steps,
                    render=render, replay_mode=replay_mode,
                )
            n2n_successes   += int(result['success'])
            n2n_env_success += int(result['env_success'])
            print(f"  n2n {i+1:3d}/{num_rollouts}  goal={goal_node}  "
                  f"success={result['success']}  env_success={result['env_success']}  "
                  f"steps={result['steps']}  path={result['path']}")

            n2n_data.append({
                'episode': i, 'goal_node': goal_node,
                'success': result['success'], 'env_success': result['env_success'],
                'steps': result['steps'], 'path': result['path'],
            })

            if render and result['frames']:
                label = 'ok' if result['success'] else 'fail'
                _save_gif(result['frames'],
                          rollouts_dir / f"n2n_{i:03d}_goal{goal_node}_{label}.gif",
                          node_labels=result.get('node_labels'),
                          goal_node=goal_node)

        n2n_time = time.time() - t_eval
        n2n_rate = n2n_successes / num_rollouts
        n2n_env_rate = n2n_env_success / num_rollouts
        print(f"\nNode-to-node: {n2n_successes}/{num_rollouts} = {n2n_rate:.2%}  "
              f"env_success={n2n_env_rate:.2%}")
        torch.save(n2n_data, eval_dir / "n2n_trajectories.pt")

        # ---- 2. State-to-state -------------------------------------------
        print(f"\n{'='*60}")
        print(f"State-to-state evaluation ({num_rollouts} rollouts, demo end goals)")
        print(f"{'='*60}")
        t_eval = time.time()
        s2s_successes       = 0
        s2s_graph_successes = 0
        s2s_env_success     = 0
        s2s_data            = []

        for i in range(num_rollouts):
            start_pair = random.choice(env._start_pairs)
            goal_pair  = random.choice(env._goal_pairs)
            goal_obs   = goal_pair.obs
            goal_state = goal_pair.state

            goal_node = online_sys.perceiver(goal_obs)
            result = online_sys.rollout_to_state(
                goal_obs=goal_obs, goal_state=goal_state,
                start_state=start_pair.state,
                max_edge_steps=max_edge_steps,
                max_last_mile_steps=max_last_mile,
                render=render, replay_mode=replay_mode,
                rollout_method=rollout_method, max_retries=max_retries,
            )
            s2s_successes       += int(result['success'])
            s2s_graph_successes += int(result['graph_success'])
            s2s_env_success     += int(result['env_success'])
            print(f"  s2s {i+1:3d}/{num_rollouts}  goal_node={goal_node}  "
                  f"graph={result['graph_success']}  success={result['success']}  "
                  f"env_success={result['env_success']}  "
                  f"steps={result['steps']}  path={result['path']}")

            s2s_data.append({
                'episode': i, 'goal_node': goal_node,
                'success': result['success'], 'graph_success': result['graph_success'],
                'env_success': result['env_success'],
                'steps': result['steps'], 'path': result['path'],
            })

            if render and result['frames']:
                label = 'ok' if result['success'] else ('graph_ok' if result['graph_success'] else 'fail')
                _save_gif(result['frames'],
                          rollouts_dir / f"s2s_{i:03d}_goal{goal_node}_{label}.gif",
                          node_labels=result.get('node_labels'),
                          goal_node=goal_node)

        s2s_time = time.time() - t_eval
        s2s_rate = s2s_successes / num_rollouts
        s2s_graph_rate = s2s_graph_successes / num_rollouts
        s2s_env_rate   = s2s_env_success / num_rollouts
        print(f"\nState-to-state: success={s2s_rate:.2%}  "
              f"graph_success={s2s_graph_rate:.2%}  env_success={s2s_env_rate:.2%}")
        torch.save(s2s_data, eval_dir / "s2s_trajectories.pt")

        # ---- Save results ------------------------------------------------
        with open(eval_dir / "online_results.txt", "w") as f:
            f.write(f"offline_dir: {offline_dir}\n")
            f.write(f"online_num_steps: {online_sys.training_step}\n")
            f.write(f"sampling_method: {online_cfg.sampling_method}\n")
            f.write(f"residualize: {online_cfg.residualize}\n")
            f.write(f"num_rollouts: {num_rollouts}\n")
            f.write(f"train_time_s: {train_time_so_far:.1f}\n")
            f.write(f"initial_graph_edges: {len(graph)}\n")
            f.write(f"final_graph_edges: {len(online_sys.graph)}\n")
            f.write(f"# Node-to-node\n")
            f.write(f"n2n_success_rate: {n2n_rate:.4f}\n")
            f.write(f"n2n_env_success_rate: {n2n_env_rate:.4f}\n")
            f.write(f"n2n_eval_time_s: {n2n_time:.1f}\n")
            f.write(f"# State-to-state (demo end goals, is_at_goal as success)\n")
            f.write(f"s2s_success_rate: {s2s_rate:.4f}\n")
            f.write(f"s2s_graph_success_rate: {s2s_graph_rate:.4f}\n")
            f.write(f"s2s_env_success_rate: {s2s_env_rate:.4f}\n")
            f.write(f"s2s_eval_time_s: {s2s_time:.1f}\n")
        print(f"Results saved to {eval_dir}")
        return s2s_env_rate

    # ── Resume or train ─────────────────────────────────────────────────────
    OmegaConf.save(cfg, output_dir / "online_config.yaml")
    resume = cfg.get("resume", False)
    ckpt_path = output_dir / "online_system.pt"

    if resume and ckpt_path.exists():
        print(f"\nResuming from {ckpt_path} — skipping training.")
        online_sys.load_checkpoint(ckpt_path)
        train_time = 0.0
        loss_history = []
        loss_history_path = output_dir / "online_loss_history.pt"
        if loss_history_path.exists():
            loss_history = torch.load(loss_history_path, map_location="cpu", weights_only=False)
        final_rate = run_full_evaluation(output_dir, loss_history, train_time)
        return final_rate

    # Fresh training — evaluate at step 0 then every `eval_every` training steps.
    total_steps = online_cfg.num_steps
    loss_history: list[dict] = []
    train_time = 0.0

    def _eval_here():
        step = online_sys.training_step
        eval_dir = output_dir / f"step_{step:06d}"
        eval_dir.mkdir(parents=True, exist_ok=True)
        online_sys.save(eval_dir / "online_system.pt")
        torch.save(loss_history, eval_dir / "online_loss_history.pt")
        # Also overwrite top-level checkpoint so resume=true picks up latest.
        online_sys.save(output_dir / "online_system.pt")
        torch.save(loss_history, output_dir / "online_loss_history.pt")
        run_full_evaluation(eval_dir, loss_history, train_time)

    print(f"\nInitial evaluation at step 0...")
    (output_dir / "step_000000").mkdir(parents=True, exist_ok=True)
    _eval_here()

    steps_done = 0
    while steps_done < total_steps:
        chunk = min(eval_every, total_steps - steps_done)
        print(f"\nOnline training chunk: {chunk} steps "
              f"({steps_done} → {steps_done + chunk} / {total_steps})...")
        t0 = time.time()
        chunk_history = online_sys.train(
            num_steps=chunk,
            log_every=online_cfg.log_every,
            checkpoint_path=output_dir / "online_system_chkpt.pt",
            checkpoint_every=100,
        )
        train_time += time.time() - t0
        loss_history.extend(chunk_history)
        steps_done += chunk
        edges_added = len(online_sys.graph) - len(graph)
        print(f"Chunk complete: training_step={online_sys.training_step}  "
              f"edges_added={edges_added:+d}  total_train_time={train_time:.1f}s")
        _eval_here()

    print(f"\nAll training + evaluations complete. Total time: {train_time:.1f}s")
    return 0.0


if __name__ == "__main__":
    main()
