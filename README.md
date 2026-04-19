# Contrastive TAMP: Learning Shortcuts from Demonstrations

Thesis project on learning TAMP-style graph structure and shortcut edges from
robotic-manipulation demonstrations.

Given a set of offline demonstrations, this codebase:

1. **Offline pipeline** — trains a Contrastive Metric Distance (CMD) critic
   and a goal-conditioned actor, then clusters the latent space into a discrete
   set of *nodes* and builds a directed *graph* of reliable edges (transitions
   between adjacent nodes the offline policy can actually execute).
2. **Online pipeline** — loads the offline artifacts, continues training in the
   environment, and tries to *promote* new edges into the graph (particularly
   shortcuts) as the actor improves. Periodically evaluates navigation success.

The environments currently supported are the five single/dual-arm
[robomimic](https://robomimic.github.io/) manipulation tasks (lift, can, square,
tool_hang, transport) plus a toy SimpleGridworld used for unit tests.

## Repository layout

```
src/
  learning_system.py    — Offline training (CMD critic + actor) + perceiver + graph
  online_system.py      — Online training (shortcut promotion, UCB sampling,
                          rollouts, checkpoint save/load, etc.)
  networks.py           — PyTorch modules: d_net, c_net, ContinuousActor,
                          ResidualContinuousActor, discrete actor
  robomimic_dataset.py  — Wraps a robomimic HDF5 dataset in a Minari-style API
  robomimic_env.py      — Wraps a robosuite env (reset_to, is_at_goal, etc.)
  simple_gridworld.py   — Toy env + dataset for unit tests
  visualization.py      — All plotting (node scatter, loss curves, PCA,
                          edge-status heatmap, node-distance heatmap,
                          networkx graph diagram)

experiments/
  configs/              — Hydra yaml configs, one per environment
  pipeline.py           — Offline pipeline entry point
  online_pipeline.py    — Online pipeline entry point (with periodic eval)
  visualize.py          — Post-hoc visualization for an offline run
  visualize_online.py   — Post-hoc visualization for an online run
  extract_stats.py      — Aggregate all online runs into one wide CSV
  plot_sweeps.py        — Line plots of metrics vs. training steps
  slurm/
    run_pipeline.slurm  — Submits the offline pipeline to SLURM
    online_worker.slurm — Generic SLURM worker for online runs
    submit_online.sh    — Convenience wrapper to submit online jobs
```

## Installation

The project uses the `slap_env` conda environment (see that env's requirements
for the exact stack — includes `torch`, `robomimic`, `robosuite`, `mujoco`,
`hydra-core`, `omegaconf`, `h5py`, `numpy`, `scikit-learn`, `matplotlib`,
`networkx`, `Pillow`, `gymnasium`).

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate slap_env
```

## Running

### 1. Prepare a robomimic dataset

Download the `demo_v15.hdf5` file for the task of interest, then convert it to
an observation-enriched low-dim dataset with robomimic's conversion script:

```bash
python path/to/robomimic/scripts/dataset_states_to_obs.py \
    --dataset /path/to/demo_v15.hdf5 \
    --output_name low_dim_v15.hdf5 \
    --types low_dim
```

Update the corresponding `experiments/configs/{env}_ph_lowdim.yaml` so
`dataset.path` points at the new file.

### 2. Offline pipeline

Trains the CMD critic + actor, builds the perceiver (clustering), and builds
the initial edge graph.

```bash
# Locally
python experiments/pipeline.py --config-name lift_ph_lowdim

# On SLURM
sbatch experiments/slurm/run_pipeline.slurm
```

Output dir contains `learning_system.pt`, `perceiver_data.pt`, `graph_data.pt`,
`node_state_pool.pt`, `edge_stats.pt`, `loss_history.pt`, `config.yaml`, and
`results.txt`. Run `python experiments/visualize.py <output_dir>` afterwards
to get PCA plots, node sample grids, loss curves, etc.

### 3. Online pipeline

Continues training from the offline artifacts while trying to add shortcut
edges to the graph. Evaluations (node-to-node and state-to-state) are run
every `online_eval.eval_every` training steps into `step_NNNNNN/`
subdirectories so you can track progress over a single run.

```bash
# Submit via the convenience wrapper
./experiments/slurm/submit_online.sh <env> <num_steps> [--residual] [extra hydra overrides]

# Examples
./experiments/slurm/submit_online.sh lift 2000
./experiments/slurm/submit_online.sh can 2000 --residual
./experiments/slurm/submit_online.sh square 5000 online_system.sampling_method=uniform

# Re-evaluate an existing run (skips training, runs all evals again)
./experiments/slurm/submit_online.sh --resume /scratch/.../outputs/can_online_2000steps
```

Output dir for a fresh run:

```
{env}_online_{N}steps[_residual]/
  online_config.yaml
  online_system.pt             — latest checkpoint (for resume)
  online_loss_history.pt       — cumulative loss history
  step_000000/                 — evaluation before any training
  step_000500/
  step_001000/
  ...
  step_NNNNNN/                 — evaluation after N training steps
    online_system.pt           — checkpoint at this step
    online_loss_history.pt
    online_dataset_nodes.png
    online_edge_status.png
    online_node_distances.png
    online_graph.png
    online_loss_curves.png
    online_results.txt
    n2n_trajectories.pt
    s2s_trajectories.pt
    rollouts/*.gif             — per-rollout GIFs, node labels overlaid
```

### 4. Aggregation and plotting

```bash
# Build a wide CSV with one row per (env, residual) and columns per checkpoint
python experiments/extract_stats.py

# Plot metrics across training steps (edit metric/combo at the top)
python experiments/plot_sweeps.py
```

`online_stats.csv` contains success rates (n2n, s2s graph-level, s2s is_at_goal,
env-success), average steps conditional on success (with standard errors),
edges added, and cumulative training time — all per checkpoint.

## Key configuration knobs

Each environment's `configs/{env}_ph_lowdim.yaml` has sections for:

- **`dataset` / `env`** — HDF5 path, obs keys, MuJoCo rendering options
- **`learning_system`** — offline CMD+actor architecture + losses
- **`perceiver`** — K (number of nodes), edge success threshold
- **`online_system`** — online CRTR buffer, edge-promotion rules
  (`edge_window_size`, `edge_success_threshold`), UCB sampling (`ucb_beta`),
  `residualize` flag
- **`online_eval`** — `num_rollouts`, `max_edge_steps`, `rollout_method`
  (`retries` or `dijkstra`), `max_retries`, `eval_every`

## Debugging helpers

- `experiments/test_rollouts.py` — loads a saved `learning_system.pt` and runs
  repeated rollouts with memory tracing, useful for reproducing OOM-style bugs
  without retraining.
- `check_obs_keys.py`, `check_action_bounds.py` — quick diagnostics on a
  robomimic HDF5 file.

## Notes on the approach

- **CMD critic** learns a (quasi-)metric over observations, used both as a
  distance heuristic for graph construction and as the reward signal for the
  actor (maximize estimated distance reduction toward the goal).
- **Perceiver** is built by k-medoids clustering in the CMD latent space.
- **Initial graph edges** are (source, target) node pairs where a rollout with
  the offline actor reaches `target` from `source` at least `success_threshold`
  fraction of the time.
- **Online edge promotion** watches a rolling window of online rollouts per
  node pair; once the success rate exceeds `edge_success_threshold`, a frozen
  snapshot of the current actor is stored as a new graph edge.
- **Shortcut rollouts** at eval time plan through the graph (Dijkstra on CMD
  distance), then either roll out the edge policies live or (default)
  retry each edge stochastically up to `max_retries` times before giving up.


Disclaimer: AI-assisted tools (Claude and ChatGPT) were used during this work
for coding assistance, debugging, and generating diagrams. All experimental design,
analysis, interpretation, and writing are our own.
