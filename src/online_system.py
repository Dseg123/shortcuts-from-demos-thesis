"""Online CRL system for fine-tuning encoders and actor from environment rollouts.

After an offline LearningSystem has been trained and a perceiver/graph have been
built, OnlineSystem continues training using trajectories collected directly from
the environment.  The perceiver and graph remain fixed; only the networks are
updated.

Training loop (per gradient step):
  1. Collect collect_per_step trajectories:
       a. Sample source node A and goal node B uniformly (A != B).
       b. Sample a start state from A's state pool.
       c. Roll out the actor towards B's medoid obs for at most max_steps,
          terminating early when the perceiver classifies the current obs as B.
       d. Add trajectory to the replay buffer; add each visited obs to its
          perceived node's state pool.
  2. Sample a CRTR batch from the replay buffer.
  3. One gradient step: contrastive critic loss + CRL actor loss (no BC term).
"""

from __future__ import annotations

import copy
import heapq
import itertools
import math
import random
from collections import deque
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from src.networks import ResidualContinuousActor


@dataclass
class OnlineCRLConfig:
    # CRTR sampling
    batch_size:            int   = 64
    repetition_factor:     int   = 4
    gamma:                 float = 0.99

    # Loss weights (no BC term in online mode)
    crl_weight:            float = 1.0
    entropy_weight:        float = 0.01

    # Optimizers
    critic_lr:             float = 3e-4
    actor_lr:              float = 1e-4
    grad_clip:             float = 1.0

    # Episode collection
    max_steps:             int   = 100   # Horizon per trajectory
    collect_per_step:      int   = 4     # Trajectories collected before each update

    # Replay buffer
    buffer_size:           int   = 10000

    # Gain / distance estimation
    dist_scale:            float = 1.0   # Scales latent distance → steps conversion
    auto_dist_scale:       bool  = True  # Auto-calibrate dist_scale from graph edges
    dist_quantile:         float = 1.0   # Quantile used when auto-calibrating
    gain_samples:          int   = 50    # States sampled per node for distance estimates

    # Online graph expansion
    edge_window_size:       int   = 10    # Rolling window length for success tracking
    edge_success_threshold: float = 0.8   # Fraction of successes required to add an edge

    # Node-pair sampling strategy for collect_trajectory
    sampling_method:        str   = "uniform"  # "uniform" | "ucb" | "stochastic_ucb"
    ucb_beta:               float = 1.0        # Exploration bonus weight for UCB methods
    gain_update_frequency:  int   = 10         # Update gain table every N gradient steps

    # Residual actor: condition the online actor on the current graph's base action.
    # Only applies to continuous action spaces.
    residualize:            bool  = False

    # Training loop
    num_steps:              int   = 10000
    log_every:              int   = 100

    # Device
    device:                str   = 'cuda'


class OnlineSystem:
    """Online CRL: continues training encoder/actor from live environment rollouts.

    Args:
        ls:              Trained LearningSystem — source of network weights and
                         obs pre-processing metadata.  Not mutated.
        perceiver:       Fixed perceiver from ls.create_perceiver(), with .node_obs.
        graph:           Fixed graph from ls.create_graph() — not used during
                         training but stored for downstream planning.
        node_state_pool: Per-node list of observations, as returned by
                         ls.create_graph().  Reused directly to avoid re-classifying
                         the dataset.
        env:             Environment with reset_to / step interface.
        config:          OnlineCRLConfig.
    """

    def __init__(
        self,
        ls,
        perceiver,
        graph: dict,
        node_state_pool: list[list[np.ndarray]],
        env,
        config: OnlineCRLConfig | None = None,
    ):
        if config is None:
            config = OnlineCRLConfig()
        self.config    = config
        self.env       = env
        self.perceiver = perceiver
        self.graph     = graph

        # Borrow obs-handling metadata from the offline system.
        self._is_image_obs      = ls._is_image_obs
        self._obs_key           = ls._obs_key
        self.obs_shape          = ls.obs_shape
        self.action_space_type  = ls.action_space_type
        self.num_actions_or_dim = ls.num_actions_or_dim

        self.device = torch.device(
            config.device if torch.cuda.is_available() else 'cpu'
        )

        # Save architecture params needed to reconstruct a residual actor.
        self._obs_enc_dim       = ls.config.obs_enc_dim
        self._actor_hidden_dims = ls.config.actor_hidden_dims

        # Deep-copy networks so the original LearningSystem is not mutated.
        self.d_net = copy.deepcopy(ls.d_net).to(self.device)
        self.c_net = copy.deepcopy(ls.c_net).to(self.device)

        # Capture offline actor snapshot before self.actor is potentially replaced.
        # Used below to seed frozen pis into initial graph edges that loaded without them.
        _offline_actor_snap = copy.deepcopy(ls.actor).to(self.device)
        _offline_actor_snap.eval()

        if config.residualize and ls.action_space_type == 'continuous':
            # Fresh residual actor — starts from scratch, learns corrections on top of
            # the graph's current policies.
            self.actor = ResidualContinuousActor(
                obs_shape   = ls.obs_shape,
                action_dim  = ls.num_actions_or_dim,
                obs_enc_dim = ls.config.obs_enc_dim,
                hidden_dims = ls.config.actor_hidden_dims,
            ).to(self.device)
        else:
            self.actor = copy.deepcopy(ls.actor).to(self.device)

        self.critic_optimizer = torch.optim.Adam(
            list(self.d_net.parameters()) + list(self.c_net.parameters()),
            lr=config.critic_lr,
        )
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=config.actor_lr,
        )

        self.K = len(perceiver.node_obs)

        # Per-node state pool: reuse the one built during create_graph, then
        # enrich it online as trajectories are collected.
        self.node_state_pool: list[list[np.ndarray]] = node_state_pool

        # Ensure every node has at least its medoid obs as a fallback.
        for k in range(self.K):
            if not self.node_state_pool[k]:
                self.node_state_pool[k].append(perceiver.node_obs[k].copy())

        # Rolling success/failure window per (source, target) node pair.
        # Used to decide when to promote an actor-conditioned policy into a graph edge.
        self._edge_attempts: dict[tuple[int, int], deque] = {}

        # UCB sampling state: how many times each pair has been sampled, and the
        # most recently computed gain for each pair.  Both are (K, K) arrays.
        self.node_pair_samples: np.ndarray = np.zeros((self.K, self.K))
        self.node_pair_gains:   np.ndarray = np.zeros((self.K, self.K))
        self.total_samples:     int        = 0

        # Replay buffer: deque of trajectories.
        # Each trajectory is a list of (obs, action) pairs, with one extra
        # terminal entry (final_obs, dummy_action) so CRTR can sample the
        # last obs as a future goal.
        self._buffer: deque[list[tuple[np.ndarray, np.ndarray]]] = deque(
            maxlen=config.buffer_size
        )

        # Shortest-path distance matrix initialised from the graph's edge costs.
        # Rows = source, cols = target.  inf = no known path.
        self.node_pair_graph_dists: np.ndarray = self._init_graph_dists()

        # Cache of action sequences for each directed edge, populated during rollouts.
        # Key: (source_node, target_node).  Replay on subsequent visits to avoid
        # re-running the actor when a sequence is known to work.
        self.edge_action_cache: dict[tuple[int, int], list[np.ndarray]] = {}

        # Freeze the offline actor into initial graph edges that loaded with pi=None.
        # Without this, rollout() falls through to self._actor_pi() (live actor) for
        # those edges, so online training would silently degrade them.
        self.graph = {
            (A, B): (self._make_frozen_pi(B, actor_override=_offline_actor_snap), avg, cost)
            if pi is None else (pi, avg, cost)
            for (A, B), (pi, avg, cost) in self.graph.items()
        }
        del _offline_actor_snap

        # first_edge_dict[(src, goal)] = next_hop: for each reachable (src, goal) pair,
        # the immediate next node on the shortest path.  Used to look up base actions.
        self.first_edge_dict: dict[tuple[int, int], int] = self._build_first_edge_dict()

        self.training_step = 0

    # -----------------------------------------------------------------------
    # Obs helpers (mirrors LearningSystem)
    # -----------------------------------------------------------------------

    def _preprocess(self, obs_np: np.ndarray) -> torch.Tensor:
        if self._is_image_obs:
            if obs_np.dtype == np.uint8:
                obs = obs_np.astype(np.float32) / 255.0
            else:
                _MAX = np.array([10.0, 5.0, 2.0], dtype=np.float32)
                obs = obs_np.astype(np.float32) / _MAX
            obs = np.transpose(obs, (0, 3, 1, 2))
            return torch.from_numpy(obs).to(self.device)
        return torch.from_numpy(obs_np.astype(np.float32)).to(self.device)

    def _zero_acts(self, B: int) -> torch.Tensor:
        if self.action_space_type == 'discrete':
            return torch.zeros(B, dtype=torch.long, device=self.device)
        return torch.zeros(B, self.num_actions_or_dim, dtype=torch.float32, device=self.device)

    # -----------------------------------------------------------------------
    # Geometric future sampling (same as LearningSystem)
    # -----------------------------------------------------------------------

    def _sample_geometric_future(self, t: int, T: int) -> int:
        gamma = self.config.gamma
        max_k = T - t - 1
        if max_k == 0:
            return t
        probs = [(1 - gamma) * (gamma ** k) for k in range(max_k)]
        total = sum(probs)
        if total > 0:
            probs = [p / total for p in probs]
            k = random.choices(range(max_k), weights=probs)[0]
        else:
            k = 0
        return t + k + 1

    # -----------------------------------------------------------------------
    # Trajectory collection
    # -----------------------------------------------------------------------

    def collect_trajectory(
        self,
        source_node: int | None = None,
        target_node: int | None = None,
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """Roll out the actor from a node-A state towards node B's medoid.

        Args:
            source_node: Start node (uniform random if None).
            target_node: Goal node  (uniform random if None, must differ from source).

        Returns:
            Trajectory as a list of (obs, action) pairs, plus a terminal entry
            (final_obs, dummy_action) for CRTR future-goal sampling.
        """
        if source_node is None or target_node is None:
            source_node, target_node = self._sample_node_pair()

        self.node_pair_samples[source_node, target_node] += 1
        self.total_samples += 1

        goal_obs    = self.perceiver.node_obs[target_node]     # for actor conditioning
        goal_state  = self.perceiver.node_states[target_node]  # for env.reset_to
        start_obs   = random.choice(self.node_state_pool[source_node]).copy()
        start_state = self.perceiver.obs_to_state.get(
            start_obs.tobytes(), self.env.obs_to_state(start_obs)
        )

        obs, _ = self.env.reset_to(start_state, goal_state)

        trajectory: list[tuple[np.ndarray, np.ndarray]] = []

        for _ in range(self.config.max_steps):
            obs_t  = self._preprocess(obs[None])
            goal_t = self._preprocess(goal_obs[None])
            with torch.no_grad():
                if self.config.residualize and self.action_space_type == 'continuous':
                    base_np = self._get_base_action(obs, target_node)
                    base_t  = torch.from_numpy(base_np[None]).float().to(self.device)
                    action  = self.actor.sample(obs_t, goal_t, base_t, deterministic=False)
                else:
                    action = self.actor.sample(obs_t, goal_t, deterministic=False)
            action_np = action.cpu().numpy()[0]

            trajectory.append((obs.copy(), action_np.copy()))

            obs, _, done, _, _ = self.env.step(action_np)

            # Enrich state pool with newly visited obs.
            node_id = self.perceiver(obs)
            self.node_state_pool[node_id].append(obs.copy())

            if node_id == target_node or done:
                break

        # Terminal entry: obs after the last action; action ignored by CRTR.
        dummy = trajectory[-1][1] if trajectory else np.zeros(
            self.num_actions_or_dim, dtype=np.float32
        )
        trajectory.append((obs.copy(), dummy))

        self._buffer.append(trajectory)

        # Track success and maybe promote edge into graph.
        reached = (self.perceiver(obs) == target_node)
        self._record_attempt(source_node, target_node, reached)

        return trajectory

    def _sample_node_pair(self) -> tuple[int, int]:
        """Sample a (source, target) node pair according to config.sampling_method.

        uniform:        Pick uniformly at random from all K*(K-1) pairs.
        ucb:            Pick the pair with highest gain + UCB bonus (greedy).
        stochastic_ucb: Sample proportionally to softmax(gain + UCB bonus).

        For UCB methods, pairs already in the graph have gain=0 and will only
        be selected if heavily underexplored, naturally focusing effort on
        candidate new edges.
        """
        method = self.config.sampling_method

        if method == "uniform":
            src = random.randrange(self.K)
            tgt = random.choice([n for n in range(self.K) if n != src])
            return src, tgt

        # Build (K, K) weight matrix of gain + UCB bonus.
        gains = self.node_pair_gains.copy()
        n = max(self.total_samples, 1)
        ucb = self.config.ucb_beta * np.sqrt(
            2.0 * np.log(n) / (self.node_pair_samples + 1e-8)
        )
        weights = gains + ucb
        # Mask self-loops.
        np.fill_diagonal(weights, -np.inf)

        if method == "ucb":
            flat = int(np.argmax(weights))
            return flat // self.K, flat % self.K

        if method == "stochastic_ucb":
            w = weights.flatten()
            w = w - w[np.isfinite(w)].max()   # shift for numerical stability
            exp_w = np.where(np.isfinite(weights.flatten()), np.exp(w), 0.0)
            probs = exp_w / exp_w.sum()
            flat = int(np.random.choice(self.K * self.K, p=probs))
            return flat // self.K, flat % self.K

        raise ValueError(f"Unknown sampling_method: {method!r}")

    def _update_gains(self) -> None:
        """Recompute exact_gain for all pairs not already in the graph."""
        for src in range(self.K):
            for tgt in range(self.K):
                if src == tgt or (src, tgt) in self.graph:
                    self.node_pair_gains[src, tgt] = 0.0
                else:
                    self.node_pair_gains[src, tgt] = self.exact_gain(src, tgt)

    def _record_attempt(self, source_node: int, target_node: int, success: bool) -> None:
        """Record a trajectory outcome and add edge to graph if success rate is high enough."""
        key = (source_node, target_node)
        if key not in self._edge_attempts:
            self._edge_attempts[key] = deque(maxlen=self.config.edge_window_size)
        self._edge_attempts[key].append(success)

        # Only promote once the window is full and the edge isn't already in the graph.
        window = self._edge_attempts[key]
        if (
            len(window) < self.config.edge_window_size
            or key in self.graph
        ):
            return

        rate = sum(window) / len(window)
        if rate >= self.config.edge_success_threshold:
            cost = self.estimate_node_distance(source_node, target_node)
            pi   = self._make_frozen_pi(target_node)  # snapshot captured inside
            self.graph[key] = (pi, None, cost)
            self._update_graph_distances(source_node, target_node)
            self.first_edge_dict = self._build_first_edge_dict()
            n_edges = len(self.graph)
            print(
                f"\n*** EDGE PROMOTED: {source_node} → {target_node}  "
                f"success_rate={rate:.0%}  cost={cost:.2f}  "
                f"graph_edges={n_edges}  step={self.training_step} ***\n"
            )

    # -----------------------------------------------------------------------
    # CRTR batch sampling
    # -----------------------------------------------------------------------

    def sample_crtr_batch(self) -> dict:
        """Sample a CRTR batch from the online replay buffer.

        Returns:
            dict with keys: states (N,...), actions (N,...), goals (N,...).
        """
        if len(self._buffer) == 0:
            raise RuntimeError("Replay buffer is empty — collect trajectories first.")

        trajs  = random.choices(list(self._buffer), k=self.config.batch_size)
        states, actions, goals = [], [], []

        for traj in trajs:
            T = len(traj) - 1   # real steps; traj[T] is the terminal entry
            if T < 1:
                continue
            for _ in range(self.config.repetition_factor):
                t       = random.randint(0, T - 1)
                t_prime = self._sample_geometric_future(t, T)
                states.append(traj[t][0])
                actions.append(traj[t][1])
                goals.append(traj[t_prime][0])

        return {
            'states':  np.stack(states),
            'actions': np.stack(actions),
            'goals':   np.stack(goals),
        }

    # -----------------------------------------------------------------------
    # Network update
    # -----------------------------------------------------------------------

    def update_networks(self, batch: dict) -> tuple[float, float, dict]:
        """One gradient step: alignment-uniformity critic + CRL actor (no BC term).

        Returns:
            (critic_loss, actor_loss, diagnostics_dict).
        """
        state_obs = self._preprocess(batch['states'])
        goal_obs  = self._preprocess(batch['goals'])

        if self.action_space_type == 'discrete':
            acts_t = torch.from_numpy(batch['actions'].astype(np.int64)).to(self.device)
        else:
            acts_t = torch.from_numpy(batch['actions'].astype(np.float32)).to(self.device)

        B = state_obs.shape[0]

        # === Critic update ===
        self.critic_optimizer.zero_grad()

        perm = torch.randperm(B, device=self.device)
        goal_acts_perm = acts_t[perm]

        phi_sym, phi_asym = self.d_net.encode(state_obs, acts_t)
        psi_sym, psi_asym = self.d_net.encode(goal_obs,  goal_acts_perm)

        phi_sym_e  = phi_sym.unsqueeze(1).expand(B, B, -1).reshape(B*B, -1)
        phi_asym_e = phi_asym.unsqueeze(1).expand(B, B, -1).reshape(B*B, -1)
        psi_sym_e  = psi_sym.unsqueeze(0).expand(B, B, -1).reshape(B*B, -1)
        psi_asym_e = psi_asym.unsqueeze(0).expand(B, B, -1).reshape(B*B, -1)

        goal_obs_e = goal_obs.unsqueeze(0).expand(B, B, *goal_obs.shape[1:]).reshape(B*B, *goal_obs.shape[1:])
        neg_dists  = self.d_net._dist_from_encodings(phi_sym_e, phi_asym_e, psi_sym_e, psi_asym_e)
        energy     = (self.c_net(goal_obs_e) + neg_dists).reshape(B, B)

        labels      = torch.arange(B, device=self.device)
        critic_loss = F.cross_entropy(energy, labels) + F.cross_entropy(energy.T, labels)

        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.d_net.parameters()) + list(self.c_net.parameters()),
            self.config.grad_clip,
        )
        self.critic_optimizer.step()

        # === Actor update ===
        self.actor_optimizer.zero_grad()

        # Compute base actions from the current graph (zeros if not residualize).
        base_np   = self._get_base_actions_batch(batch['states'], batch['goals'])
        base_acts = torch.from_numpy(base_np).to(self.device) if self.config.residualize else None

        if base_acts is not None:
            sampled_acts = self.actor.sample(state_obs, goal_obs, base_acts)
        else:
            sampled_acts = self.actor.sample(state_obs, goal_obs)

        zero_goal_acts = self._zero_acts(B)

        with torch.no_grad():
            psi_sym_a, psi_asym_a = self.d_net.encode(goal_obs, zero_goal_acts)
        phi_sym_a, phi_asym_a = self.d_net.encode(state_obs, sampled_acts)
        crl_loss = -self.d_net._dist_from_encodings(
            phi_sym_a, phi_asym_a, psi_sym_a.detach(), psi_asym_a.detach()
        ).mean()

        if base_acts is not None:
            ent_loss = -self.actor.entropy(state_obs, goal_obs, base_acts).mean()
        else:
            ent_loss = -self.actor.entropy(state_obs, goal_obs).mean()
        actor_loss = (
            self.config.crl_weight       * crl_loss
            + self.config.entropy_weight * ent_loss
        )

        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.config.grad_clip)
        self.actor_optimizer.step()

        self.training_step += 1
        return critic_loss.item(), actor_loss.item(), {
            'critic': critic_loss.item(),
            'crl':    crl_loss.item(),
            'ent':    ent_loss.item(),
        }

    # -----------------------------------------------------------------------
    # Persistence
    # -----------------------------------------------------------------------

    def save(self, path) -> None:
        """Save network weights and training state to a checkpoint file.

        The graph's pi closures cannot be serialised (they contain deepcopied
        actor snapshots), so the graph topology is saved as edge keys + costs
        only.  Reload with LearningSystem.load() + load_pipeline_artifacts()
        to restore the full graph, then reconstruct an OnlineSystem from it.
        """
        from dataclasses import asdict
        from pathlib import Path
        # Serialise per-edge frozen actor snapshots so they can be restored.
        graph_edges = {}
        for (src, tgt), (pi, avg_steps, cost) in self.graph.items():
            edge_data = {'cost': cost}
            if pi is not None and hasattr(pi, 'actor_state_dict'):
                edge_data['actor_state_dict'] = pi.actor_state_dict
                edge_data['is_residual']      = pi.is_residual
                edge_data['goal_node']        = pi.goal_node
            graph_edges[str((src, tgt))] = edge_data

        checkpoint = {
            'config':            asdict(self.config),
            'obs_shape':         self.obs_shape,
            'action_space_type': self.action_space_type,
            'num_actions_or_dim': self.num_actions_or_dim,
            'd_net':             self.d_net.state_dict(),
            'c_net':             self.c_net.state_dict(),
            'actor':             self.actor.state_dict(),
            'training_step':     self.training_step,
            'graph_edges':       graph_edges,
            # UCB state
            'node_pair_samples': self.node_pair_samples,
            'node_pair_gains':   self.node_pair_gains,
            'total_samples':     self.total_samples,
        }
        torch.save(checkpoint, Path(path))
        print(f"OnlineSystem saved to {path}")

    def load_checkpoint(self, path) -> None:
        """Restore network weights, graph edges (with frozen pis), and UCB state.

        Must be called on an already-constructed OnlineSystem (built from offline
        artifacts).  Overwrites d_net, c_net, actor, graph, and UCB arrays with
        the saved state.
        """
        from pathlib import Path
        ckpt = torch.load(Path(path), map_location=self.device, weights_only=False)

        self.d_net.load_state_dict(ckpt['d_net'])
        self.c_net.load_state_dict(ckpt['c_net'])
        self.actor.load_state_dict(ckpt['actor'])
        self.training_step = ckpt.get('training_step', 0)

        # Restore UCB state.
        if 'node_pair_samples' in ckpt:
            self.node_pair_samples = ckpt['node_pair_samples']
            self.node_pair_gains   = ckpt['node_pair_gains']
            self.total_samples     = ckpt['total_samples']

        # Rebuild graph edges with frozen pis from saved actor state_dicts.
        saved_edges = ckpt.get('graph_edges', {})
        for key_str, edge_data in saved_edges.items():
            edge_key = eval(key_str)  # "(src, tgt)" -> (src, tgt)
            if edge_key in self.graph:
                continue  # offline edge — already has a frozen pi from __init__

            cost = edge_data['cost'] if isinstance(edge_data, dict) else edge_data
            if not isinstance(edge_data, dict) or 'actor_state_dict' not in edge_data:
                # Legacy format or missing snapshot — fall back to live actor.
                self.graph[edge_key] = (None, None, cost)
                continue

            # Reconstruct the frozen actor snapshot.
            is_residual = edge_data.get('is_residual', False)
            goal_node   = edge_data['goal_node']
            if is_residual:
                actor_shell = ResidualContinuousActor(
                    obs_shape=self.obs_shape,
                    action_dim=self.num_actions_or_dim,
                    obs_enc_dim=self._obs_enc_dim,
                    hidden_dims=self._actor_hidden_dims,
                ).to(self.device)
            else:
                from src.networks import ContinuousActor
                actor_shell = ContinuousActor(
                    obs_shape=self.obs_shape,
                    action_dim=self.num_actions_or_dim,
                    obs_enc_dim=self._obs_enc_dim,
                    hidden_dims=self._actor_hidden_dims,
                ).to(self.device)
            actor_shell.load_state_dict(edge_data['actor_state_dict'])
            actor_shell.eval()

            # Build frozen pi using the restored actor snapshot.
            pi = self._make_frozen_pi(goal_node, actor_override=actor_shell)
            self.graph[edge_key] = (pi, None, cost)

        # Refresh derived structures.
        self.node_pair_graph_dists = self._init_graph_dists()
        self.first_edge_dict = self._build_first_edge_dict()
        self.edge_action_cache.clear()

        print(f"OnlineSystem checkpoint loaded from {path}  "
              f"(step {self.training_step}, {len(self.graph)} edges)")

    # -----------------------------------------------------------------------
    # Training loop
    # -----------------------------------------------------------------------

    def train(
        self,
        num_steps: int,
        log_every: int = 100,
        checkpoint_path=None,
        checkpoint_every: int = 100,
    ) -> list[dict]:
        """Collect trajectories and update networks for num_steps gradient steps.

        Each step collects config.collect_per_step trajectories before updating.

        Args:
            num_steps:        Number of gradient steps to run.
            log_every:        Log progress every this many steps.
            checkpoint_path:  If set, overwrite this file every checkpoint_every
                              steps so partial progress survives early termination.
            checkpoint_every: Checkpoint interval in steps.

        Returns:
            loss_history: list of dicts with keys step, critic, actor, crl, ent —
                          one entry per gradient step, for plotting loss curves.
        """
        loss_history: list[dict] = []

        for step in range(num_steps):
            for _ in range(self.config.collect_per_step):
                self.collect_trajectory()

            batch = self.sample_crtr_batch()
            critic_loss, actor_loss, diag = self.update_networks(batch)

            loss_history.append({
                'step':   self.training_step,
                'critic': critic_loss,
                'actor':  actor_loss,
                'crl':    diag['crl'],
                'ent':    diag['ent'],
            })

            freq = self.config.gain_update_frequency
            if self.config.sampling_method != "uniform" and (step + 1) % freq == 0:
                self._update_gains()

            if step % 1 == 0:
                pool_sizes = [len(p) for p in self.node_state_pool]
                print(
                    f"step {step:6d} | "
                    f"critic={critic_loss:.4f} | "
                    f"actor={actor_loss:.4f} "
                    f"(crl={diag['crl']:.3f} ent={diag['ent']:.3f}) | "
                    f"buf={len(self._buffer)} | pool={pool_sizes}"
                )

            if checkpoint_path is not None and (step + 1) % checkpoint_every == 0:
                self.save(checkpoint_path)

        return loss_history

    # -----------------------------------------------------------------------
    # Graph distance helpers
    # -----------------------------------------------------------------------

    def _init_graph_dists(self) -> np.ndarray:
        """Build initial (K, K) shortest-path distance matrix from graph edges.

        Direct edge costs come from create_graph's estimated CMD distances (model space).
        Floyd-Warshall fills in multi-hop paths.
        """
        d = np.full((self.K, self.K), np.inf)
        np.fill_diagonal(d, 0.0)
        for (A, B), (_, avg_steps, estimated_dist) in self.graph.items():
            d[A, B] = min(d[A, B], estimated_dist)
        # Floyd-Warshall
        for k in range(self.K):
            d = np.minimum(d, d[:, k:k+1] + d[k:k+1, :])
        return d

    def _update_graph_distances(self, source_node: int, target_node: int) -> None:
        """Relax all-pairs distances after adding edge source_node → target_node.

        Uses the current estimate_node_distance as the edge cost.
        """
        d = self.node_pair_graph_dists
        L = self.estimate_node_distance(source_node, target_node)
        via = d[:, source_node:source_node+1] + L + d[target_node:target_node+1, :]
        self.node_pair_graph_dists = np.minimum(d, via)

    def compute_fresh_distances(self) -> np.ndarray:
        """Recompute all-pairs shortest-path distances using the CURRENT d_net.

        Unlike node_pair_graph_dists (which accumulates mixed-vintage costs from
        different training steps), this evaluates estimate_node_distance with the
        current model for every graph edge, then runs Floyd-Warshall.

        Returns:
            (K, K) distance matrix.
        """
        d = np.full((self.K, self.K), np.inf)
        np.fill_diagonal(d, 0.0)
        for (A, B) in self.graph:
            d[A, B] = self.estimate_node_distance(A, B)
        for k in range(self.K):
            d = np.minimum(d, d[:, k:k+1] + d[k:k+1, :])
        return d

    def compute_fresh_path(self, start_node: int, goal_node: int) -> list[int] | None:
        """Compute shortest path using fresh distances from the current d_net.

        Returns:
            List of node IDs from start to goal, or None if unreachable.
        """
        d = self.compute_fresh_distances()
        if not np.isfinite(d[start_node, goal_node]):
            return None

        path = [start_node]
        current = start_node
        visited = {start_node}
        while current != goal_node:
            # Find the neighbor that minimises edge_cost + remaining distance.
            best_hop, best_cost = None, math.inf
            for (a, b) in self.graph:
                if a != current or b in visited:
                    continue
                via = self.estimate_node_distance(a, b) + d[b, goal_node]
                if via < best_cost:
                    best_cost, best_hop = via, b
            if best_hop is None:
                return None
            path.append(best_hop)
            visited.add(best_hop)
            current = best_hop
        return path

    # -----------------------------------------------------------------------
    # Latent distance and gain estimation
    # -----------------------------------------------------------------------

    def latent_dist(self, obs: np.ndarray, target_node: int) -> float:
        """CMD distance d_net((obs,0), (goal,0)) — how far obs is from target_node."""
        goal_obs = self.perceiver.node_obs[target_node]
        obs_t    = self._preprocess(obs[None])
        goal_t   = self._preprocess(goal_obs[None])
        with torch.no_grad():
            zero1 = self._zero_acts(1)
            sym_s, asym_s = self.d_net.encode(obs_t, zero1)
            sym_g, asym_g = self.d_net.encode(goal_t, zero1)
            dist = -self.d_net._dist_from_encodings(sym_s, asym_s, sym_g, asym_g)
        return dist.item()

    def estimate_node_distance(self, source_node: int, target_node: int) -> float:
        """Estimated trajectory length from source_node to target_node.

        CMD formula (Eysenbach et al., quasi-metric version):
            d(s, g) ≈ -(1 / log γ) * d_net(φ(s,a), ψ(g))

        d_net already returns a proper quasi-metric, so no d(g,g) correction needed.
        Averages over sampled states from the source node's pool.
        """
        n = self.config.gain_samples
        source_states = random.sample(
            self.node_state_pool[source_node],
            min(n, len(self.node_state_pool[source_node])),
        )
        d_sg = float(np.mean([self.latent_dist(s, target_node) for s in source_states]))
        return self.config.dist_scale * max(
            0.0, -(1.0 / np.log(self.config.gamma)) * d_sg
        )

    def _calibrate_dist_scale(self) -> None:
        """Auto-set dist_scale so estimated edge costs match empirical graph costs.

        For each edge already in the graph, computes ratio = estimate / empirical.
        Sets dist_scale = 1 / quantile(ratios, dist_quantile), capped at 1.0.
        """
        ratios = []
        for (A, B), (_, avg_steps, offline_dist) in self.graph.items():
            if offline_dist <= 0:
                continue
            self.config.dist_scale = 1.0  # temporarily reset for raw estimate
            estimated = self.estimate_node_distance(A, B)
            if estimated > 0:
                ratios.append(estimated / offline_dist)

        if not ratios:
            return

        ratio = float(np.quantile(ratios, self.config.dist_quantile))
        if ratio > 0 and np.isfinite(ratio):
            self.config.dist_scale = min(1.0, 1.0 / ratio)
        else:
            self.config.dist_scale = 1.0
        print(f"  Auto-calibrated dist_scale={self.config.dist_scale:.4f} "
              f"(quantile={self.config.dist_quantile}, n={len(ratios)} edges)")

    def exact_gain(self, source_node: int, target_node: int) -> float:
        """Gain from adding edge source_node → target_node to the graph.

        Gain = total reduction in shortest-path distances across all reachable
        (u, v) pairs that would benefit from routing through the new edge.
        """
        x, y = source_node, target_node
        d    = self.node_pair_graph_dists
        L    = self.estimate_node_distance(x, y)

        U = np.where(np.isfinite(d[:, x]))[0]   # nodes that can reach x
        V = np.where(np.isfinite(d[y, :]))[0]   # nodes reachable from y

        d_ux       = d[U, x][:, None]
        d_yv       = d[y, V][None, :]
        new_paths  = d_ux + L + d_yv
        old_paths  = np.minimum(d[np.ix_(U, V)], self.config.max_steps)
        improvement = np.maximum(0.0, old_paths - new_paths)
        return float(np.sum(improvement))

    # -----------------------------------------------------------------------
    # Graph expansion
    # -----------------------------------------------------------------------

    def select_best_edges(self, M: int) -> dict:
        """Greedily select the M highest-gain new edges and return an expanded graph.

        For each of M rounds:
          1. (Re-)calibrate dist_scale if auto_dist_scale is set.
          2. Compute exact_gain for all K*(K-1) candidate pairs not yet selected.
          3. Pick the pair with highest gain, create its segmented policy,
             add it to the result graph, and update node_pair_graph_dists.

        Args:
            M: Number of new edges to add.

        Returns:
            Expanded graph dict (original edges + up to M new ones).
        """
        # Work on a snapshot of distances; update it as edges are added so
        # each gain computation reflects previously selected edges.
        saved_dists = self.node_pair_graph_dists.copy()
        saved_scale = self.config.dist_scale

        new_graph   = dict(self.graph)
        added       = 0

        for round_i in range(M):
            if self.config.auto_dist_scale:
                self._calibrate_dist_scale()

            # Evaluate gain for all pairs not already in new_graph.
            best_gain, best_pair = -np.inf, None
            for A in range(self.K):
                for B in range(self.K):
                    if A == B or (A, B) in new_graph:
                        continue
                    g = self.exact_gain(A, B)
                    if g > best_gain:
                        best_gain, best_pair = g, (A, B)

            if best_pair is None or best_gain <= 0:
                print(f"  No beneficial edges remain after {added} additions.")
                break

            A, B = best_pair
            cost_AB = self.estimate_node_distance(A, B)
            pi      = self._make_frozen_pi(B)
            new_graph[(A, B)] = (pi, None, cost_AB)
            self._update_graph_distances(A, B)
            self.first_edge_dict = self._build_first_edge_dict()
            added += 1
            print(f"  Round {round_i+1}: added edge ({A} -> {B})  "
                  f"gain={best_gain:.2f}  cost={cost_AB:.1f}")

        print(f"select_best_edges: added {added} edges "
              f"(graph now has {len(new_graph)} edges).")

        # Restore dist_scale (calibration was exploratory).
        self.config.dist_scale = saved_scale
        # node_pair_graph_dists now reflects the newly added edges — keep it.
        return new_graph

    # -----------------------------------------------------------------------
    # Residual actor helpers
    # -----------------------------------------------------------------------

    def _build_first_edge_dict(self) -> dict[tuple[int, int], int]:
        """For each reachable (src, goal) pair, find the next-hop node on the shortest path.

        Uses node_pair_graph_dists (Floyd-Warshall distances) plus the edge costs in
        self.graph to find the neighbour of src that lies on the cheapest path to goal.
        """
        d = self.node_pair_graph_dists
        first: dict[tuple[int, int], int] = {}
        for src in range(self.K):
            for goal in range(self.K):
                if src == goal or not np.isfinite(d[src, goal]):
                    continue
                best_hop, best_cost = None, math.inf
                for (a, b), edge_val in self.graph.items():
                    if a != src:
                        continue
                    via = edge_val[2] + (d[b, goal] if np.isfinite(d[b, goal]) else math.inf)
                    if via < best_cost:
                        best_cost, best_hop = via, b
                if best_hop is not None:
                    first[(src, goal)] = best_hop
        return first

    def _make_frozen_pi(self, goal_node: int, actor_override=None):
        """Create a frozen pi for storing in graph edges.

        Deepcopies the current actor so future training never affects it. For
        residual actors, also captures snapshots of the current graph and
        first_edge_dict so the base action can be looked up dynamically based on
        the agent's CURRENT perceived node — not statically pinned to a single
        first-hop policy at promotion time. This is essential for shortcut edges:
        a shortcut A→C that incidentally passes through B should use B→C's pi as
        the base when the agent is in node B, not always A→B's pi.

        actor_override, if provided, is deepcopied instead of self.actor — used to
        freeze the offline actor into initial graph edges at OnlineSystem init time.

        The chain pi_X → base_pi → … terminates because each frozen snapshot only
        references edges that already existed at its creation time, and the graph
        only ever grows (existing entries are never mutated).
        """
        goal_obs    = self.perceiver.node_obs[goal_node].copy()
        actor_snap  = copy.deepcopy(actor_override if actor_override is not None else self.actor)
        actor_snap.eval()
        action_dim  = self.num_actions_or_dim
        device      = self.device
        is_residual = isinstance(actor_snap, ResidualContinuousActor)

        # Snapshot graph state for runtime base lookup. Shallow copies are safe:
        # graph entries are added but never mutated, and frozen pi closures are
        # immutable.
        if is_residual:
            graph_snap      = dict(self.graph)
            first_edge_snap = dict(self.first_edge_dict)
            perceiver_ref   = self.perceiver
        else:
            graph_snap = first_edge_snap = perceiver_ref = None

        def _prep(x: np.ndarray) -> torch.Tensor:
            return torch.from_numpy(x.astype(np.float32)).to(device)

        def pi(obs: np.ndarray, deterministic: bool = True) -> np.ndarray:
            # Look up the base action dynamically: which edge would the
            # frozen-at-promotion graph route through from the current node?
            if is_residual:
                base_act = np.zeros(action_dim, dtype=np.float32)
                current_node = perceiver_ref(obs)
                next_hop = first_edge_snap.get((current_node, goal_node))
                if next_hop is not None:
                    base_edge = graph_snap.get((current_node, next_hop))
                    if base_edge is not None and base_edge[0] is not None:
                        base_act = base_edge[0](obs)
            else:
                base_act = np.zeros(action_dim, dtype=np.float32)

            obs_t  = _prep(obs[None])
            goal_t = _prep(goal_obs[None])
            with torch.no_grad():
                if is_residual:
                    base_t = _prep(base_act[None])
                    action = actor_snap.sample(obs_t, goal_t, base_t, deterministic=deterministic)
                else:
                    action = actor_snap.sample(obs_t, goal_t, deterministic=deterministic)
            return action.cpu().numpy()[0]

        # Attach serialisable metadata so save() can persist frozen snapshots.
        pi.actor_state_dict = {k: v.cpu() for k, v in actor_snap.state_dict().items()}
        pi.is_residual      = is_residual
        pi.goal_node        = goal_node
        if is_residual:
            # Save the snapshot keys (the edges this pi depends on) so save/load
            # can reconstruct the dependency graph in the right order.
            pi.snapshot_edges = list(graph_snap.keys())

        return pi

    def _get_base_action(self, obs: np.ndarray, goal_node: int) -> np.ndarray:
        """Get the base action from the graph's current first-hop frozen pi."""
        if not self.config.residualize or self.action_space_type != 'continuous':
            return np.zeros(self.num_actions_or_dim, dtype=np.float32)
        current_node = self.perceiver(obs)
        next_hop = self.first_edge_dict.get((current_node, goal_node))
        if next_hop is None:
            return np.zeros(self.num_actions_or_dim, dtype=np.float32)
        edge_val = self.graph.get((current_node, next_hop))
        if edge_val is None or edge_val[0] is None:
            return np.zeros(self.num_actions_or_dim, dtype=np.float32)
        return edge_val[0](obs)   # frozen pi — terminates, no recursion into live actor

    def _get_base_actions_batch(
        self, states_np: np.ndarray, goals_np: np.ndarray
    ) -> np.ndarray:
        """Compute base actions for a batch of (state, goal) pairs."""
        N = len(states_np)
        base = np.zeros((N, self.num_actions_or_dim), dtype=np.float32)
        if not self.config.residualize or self.action_space_type != 'continuous':
            return base
        for i in range(N):
            base[i] = self._get_base_action(states_np[i], self.perceiver(goals_np[i]))
        return base

    # -----------------------------------------------------------------------
    # Rollout (evaluation)
    # -----------------------------------------------------------------------

    def _actor_pi(self, goal_node: int, deterministic: bool = True):
        """Live pi using the current actor; uses residual base action if residualize=True.

        Used in rollout/executing-Dijkstra only — NOT stored in graph edges.
        """
        goal_obs = self.perceiver.node_obs[goal_node].copy()

        def pi(obs: np.ndarray) -> np.ndarray:
            base_act = self._get_base_action(obs, goal_node)
            obs_t    = self._preprocess(obs[None])
            goal_t   = self._preprocess(goal_obs[None])
            with torch.no_grad():
                if self.config.residualize and self.action_space_type == 'continuous':
                    base_t = torch.from_numpy(base_act[None]).float().to(self.device)
                    action = self.actor.sample(obs_t, goal_t, base_t, deterministic=deterministic)
                else:
                    action = self.actor.sample(obs_t, goal_t, deterministic=deterministic)
            return action.cpu().numpy()[0]

        return pi

    def rollout(
        self,
        goal_node: int,
        start_state=None,
        max_edge_steps: int | None = None,
        render: bool = False,
        replay_mode: str = 'live',
    ) -> dict:
        """Navigate to goal_node using an executing Dijkstra.

        Mirrors the shortcut-learning approach: during planning each candidate
        edge is actually executed against the environment.  Edges that fail to
        bring the perceiver to the expected target node are pruned.  The
        cheapest verified-executable path is found, its action sequences are
        cached, and the path is then replayed from the start state to produce a
        clean trajectory.

        Subsequent calls with the same edge reuse the cached action sequences
        (no re-planning needed unless the cache is cleared).

        Args:
            goal_node:      Target node index.
            start_state:    State for env.reset_to().  Sampled from a random
                            node's pool if None.
            max_edge_steps: Step budget per edge attempt.  Defaults to
                            config.max_steps.
            render:         If True, collect rendered frames during the final
                            replay and return them.

        Returns:
            dict with keys:
                success (bool)  — path reached goal_node,
                steps   (int)   — env steps taken during replay,
                path    (list)  — node IDs visited during replay,
                frames  (list)  — rendered frames (empty when render=False).
        """
        if max_edge_steps is None:
            max_edge_steps = self.config.max_steps

        goal_state = self.perceiver.node_states[goal_node]

        # ---- Resolve start state ----
        if start_state is None:
            nodes = list(range(self.K))
            src_node = random.choice([n for n in nodes if n != goal_node])
            start_obs_arr = random.choice(self.node_state_pool[src_node])
            start_state = self.perceiver.obs_to_state.get(
                start_obs_arr.tobytes(), self.env.obs_to_state(start_obs_arr)
            )

        obs, _ = self.env.reset_to(start_state, goal_state)
        start_node = self.perceiver(obs)

        if start_node == goal_node:
            frames = [self.env.render()] if render else []
            return {'success': True, 'env_success': False, 'steps': 0,
                    'path': [start_node], 'frames': frames,
                    'node_labels': [start_node] if render else [],
                    'obs': obs}

        # ---- Executing Dijkstra ----
        # Heap element: (cost, counter, node_id, sim_state_at_node, obs_at_node, path_so_far)
        # sim_state_at_node is the EXACT MuJoCo state (via env.get_state()), not an
        # approximate obs_to_state lookup, so replay from start_state reproduces the
        # same trajectory that planning verified. path_so_far: list of (src, tgt,
        # action_seq) for each edge taken.
        _ctr = itertools.count()
        heap: list = [(0.0, next(_ctr), start_node, start_state, obs, [])]
        best_cost: dict[int, float] = {start_node: 0.0}

        best_path: list[tuple[int, int, list]] | None = None   # (src, tgt, actions) per edge
        planning_steps = 0

        print(f"[rollout] start_node={start_node}  goal_node={goal_node}  "
              f"max_edge_steps={max_edge_steps}")

        while heap:
            cost, _, node, state_at_node, obs_at_node, path_so_far = heapq.heappop(heap)

            if node == goal_node:
                best_path = path_so_far
                print(f"[rollout]   GOAL REACHED at cost={cost:.0f}  "
                      f"planning_steps={planning_steps}")
                break

            if cost > best_cost.get(node, math.inf):
                continue  # stale entry

            visited_nodes = {p[0] for p in path_so_far} | {node}
            print(f"[rollout]  expanding node={node} cost={cost:.0f} "
                  f"visited={sorted(visited_nodes)}  planning_steps={planning_steps}")

            for (src, tgt), edge_val in self.graph.items():
                if src != node or tgt in visited_nodes:
                    continue

                # Reset env to the EXACT sim state at this node, then try the edge.
                obs_here, _ = self.env.reset_to(state_at_node, goal_state)

                pi = (
                    edge_val[0]
                    if edge_val[0] is not None
                    else self._actor_pi(tgt)
                )
                edge_actions: list[np.ndarray] = []
                reached = False
                final_node = None

                for _ in range(max_edge_steps):
                    action = pi(obs_here)
                    edge_actions.append(action.copy())
                    obs_here, _, done, _, _ = self.env.step(action)
                    planning_steps += 1
                    final_node = self.perceiver(obs_here)
                    if final_node == tgt:
                        reached = True
                        break
                    if done:
                        break

                if not reached:
                    print(f"[rollout]    edge {src}->{tgt}: FAIL after "
                          f"{len(edge_actions)} steps (ended at node {final_node})")
                    continue  # edge failed — prune

                # Capture the EXACT post-edge sim state so this node's outgoing
                # edges can be replayed from here identically later.
                state_after = self.env.get_state()

                edge_cost = len(edge_actions)
                new_cost  = cost + edge_cost
                if new_cost < best_cost.get(tgt, math.inf):
                    best_cost[tgt] = new_cost
                    print(f"[rollout]    edge {src}->{tgt}: OK in {edge_cost} steps  "
                          f"(new_cost={new_cost:.0f})")
                    heapq.heappush(heap, (
                        new_cost, next(_ctr), tgt, state_after, obs_here,
                        path_so_far + [(src, tgt, edge_actions)],
                    ))
                else:
                    print(f"[rollout]    edge {src}->{tgt}: OK but not better "
                          f"({new_cost:.0f} >= {best_cost[tgt]:.0f})")

        if best_path is None:
            print(f"[rollout] FAILED — no path found "
                  f"(planning_steps={planning_steps}, best_costs={best_cost})")
            return {'success': False, 'env_success': False, 'steps': 0,
                    'path': [start_node], 'frames': [], 'node_labels': [],
                    'obs': None}

        print(f"[rollout] path found: "
              f"{[(s,t) for s,t,_ in best_path]}  "
              f"total cost={sum(len(a) for _,_,a in best_path)}")

        # Cache edge action sequences from the winning path.
        for src, tgt, actions in best_path:
            self.edge_action_cache[(src, tgt)] = actions

        # ---- Replay best path from start ────────────────────────────────
        obs, _ = self.env.reset_to(start_state, goal_state)
        frames: list = []
        node_labels: list[int] = []
        if render:
            frames.append(self.env.render())
            node_labels.append(self.perceiver(obs))

        replay_steps = 0
        env_success = False
        visited: list[int] = [start_node]

        if replay_mode == 'cached':
            # Replay cached action sequences exactly as recorded during planning.
            print(f"[rollout] replaying from start_state (cached actions)...")
            for src, tgt, actions in best_path:
                for action in actions:
                    obs, _, done, _, _ = self.env.step(action)
                    replay_steps += 1
                    if done:
                        env_success = True
                    if render:
                        frames.append(self.env.render())
                        node_labels.append(self.perceiver(obs))
                    if done:
                        break
                actual_node = self.perceiver(obs)
                match = "OK" if actual_node == tgt else f"MISMATCH(landed={actual_node})"
                print(f"[rollout]   replay edge {src}->{tgt}: {match}  "
                      f"steps={len(actions)}")
                visited.append(tgt)
        else:
            # Run edge policies live from the continuous trajectory state.
            # Avoids mismatch from reset_to not perfectly restoring sim state.
            print(f"[rollout] replaying from start_state (live policies)...")
            for src, tgt, planned_actions in best_path:
                edge_val = self.graph.get((src, tgt))
                pi = (
                    edge_val[0]
                    if edge_val is not None and edge_val[0] is not None
                    else self._actor_pi(tgt)
                )
                edge_steps = 0
                for _ in range(max_edge_steps):
                    action = pi(obs)
                    obs, _, done, _, _ = self.env.step(action)
                    replay_steps += 1
                    edge_steps += 1
                    if done:
                        env_success = True
                    if render:
                        frames.append(self.env.render())
                        node_labels.append(self.perceiver(obs))
                    if self.perceiver(obs) == tgt:
                        break
                    if done:
                        break
                actual_node = self.perceiver(obs)
                match = "OK" if actual_node == tgt else f"MISMATCH(landed={actual_node})"
                print(f"[rollout]   replay edge {src}->{tgt}: {match}  "
                      f"steps={edge_steps} (planned={len(planned_actions)})")
                visited.append(tgt)

        print(f"[rollout] replay done: visited={visited}  "
              f"env_success={env_success}  total_steps={replay_steps}")

        return {
            'success': visited[-1] == goal_node,
            'env_success': env_success,
            'steps': replay_steps,
            'path': visited,
            'frames': frames,
            'node_labels': node_labels,
            'obs': obs,
        }

    def rollout_with_retries(
        self,
        goal_node: int,
        start_state=None,
        max_edge_steps: int | None = None,
        max_retries: int = 10,
        render: bool = False,
    ) -> dict:
        """Navigate to goal_node using graph-planned path with per-edge retries.

        1. Compute shortest path via first_edge_dict (pure graph, no env).
        2. For each edge in the path, try executing it up to max_retries times
           with stochastic action sampling. On first success, cache the actions
           and advance. On failure after all retries, abort.

        This is more robust than executing Dijkstra for bottleneck edges: each
        edge gets multiple stochastic attempts rather than a single deterministic
        one.

        Returns:
            dict with keys: success, env_success, steps, path, frames,
            node_labels, obs.
        """
        if max_edge_steps is None:
            max_edge_steps = self.config.max_steps

        goal_state_env = self.perceiver.node_states[goal_node]

        # ---- Resolve start state ----
        if start_state is None:
            nodes = list(range(self.K))
            src_node = random.choice([n for n in nodes if n != goal_node])
            start_obs_arr = random.choice(self.node_state_pool[src_node])
            start_state = self.perceiver.obs_to_state.get(
                start_obs_arr.tobytes(), self.env.obs_to_state(start_obs_arr)
            )

        obs, _ = self.env.reset_to(start_state, goal_state_env)
        start_node = self.perceiver(obs)

        if start_node == goal_node:
            frames = [self.env.render()] if render else []
            return {'success': True, 'env_success': False, 'steps': 0,
                    'path': [start_node], 'frames': frames,
                    'node_labels': [start_node] if render else [],
                    'obs': obs}

        # ---- Compute path using fresh distances from current d_net ----
        path_nodes = self.compute_fresh_path(start_node, goal_node)
        if path_nodes is None:
            print(f"[rollout_retries] no graph path from {start_node} to {goal_node}")
            return {'success': False, 'env_success': False, 'steps': 0,
                    'path': [start_node], 'frames': [], 'node_labels': [],
                    'obs': obs}

        planned_edges = list(zip(path_nodes[:-1], path_nodes[1:]))
        print(f"[rollout_retries] start={start_node} goal={goal_node}  "
              f"planned path: {path_nodes}  edges: {planned_edges}")

        # ---- Execute each edge with retries ----
        obs, _ = self.env.reset_to(start_state, goal_state_env)
        frames: list = []
        node_labels: list[int] = []
        if render:
            frames.append(self.env.render())
            node_labels.append(self.perceiver(obs))

        total_steps = 0
        env_success = False
        visited: list[int] = [start_node]

        for src, tgt in planned_edges:
            # Use the frozen graph pi (the validated snapshot) with stochastic
            # sampling so each retry produces a different trajectory.
            edge_val = self.graph.get((src, tgt))
            frozen_pi = edge_val[0] if edge_val is not None and edge_val[0] is not None else None

            # Save state before attempting so we can retry from here.
            state_before = self.env.get_state()
            succeeded = False

            for attempt in range(max_retries):
                # Reset to the saved state for each retry.
                if attempt > 0:
                    obs, _ = self.env.reset_to(state_before, goal_state_env)

                edge_frames: list = []
                edge_labels: list[int] = []
                edge_steps = 0
                reached = False

                for _ in range(max_edge_steps):
                    if frozen_pi is not None:
                        action = frozen_pi(obs, deterministic=False)
                    else:
                        action = self._actor_pi(tgt, deterministic=False)(obs)
                    obs, _, done, _, _ = self.env.step(action)
                    edge_steps += 1
                    if done:
                        env_success = True
                    if render:
                        edge_frames.append(self.env.render())
                        edge_labels.append(self.perceiver(obs))
                    if self.perceiver(obs) == tgt:
                        reached = True
                        break
                    if done:
                        break

                if reached:
                    print(f"[rollout_retries]   edge {src}->{tgt}: OK on attempt "
                          f"{attempt+1}/{max_retries}  steps={edge_steps}")
                    total_steps += edge_steps
                    frames.extend(edge_frames)
                    node_labels.extend(edge_labels)
                    visited.append(tgt)
                    succeeded = True
                    break
                else:
                    actual = self.perceiver(obs)
                    print(f"[rollout_retries]   edge {src}->{tgt}: attempt "
                          f"{attempt+1}/{max_retries} FAIL (landed={actual})")

            if not succeeded:
                print(f"[rollout_retries]   edge {src}->{tgt}: FAILED all "
                      f"{max_retries} retries")
                break

        success = visited[-1] == goal_node
        print(f"[rollout_retries] done: visited={visited}  success={success}  "
              f"env_success={env_success}  total_steps={total_steps}")

        return {
            'success': success,
            'env_success': env_success,
            'steps': total_steps,
            'path': visited,
            'frames': frames,
            'node_labels': node_labels,
            'obs': obs,
        }

    def rollout_to_state(
        self,
        goal_obs: np.ndarray,
        goal_state: np.ndarray | None = None,
        start_state=None,
        max_edge_steps: int | None = None,
        max_last_mile_steps: int = 200,
        render: bool = False,
        replay_mode: str = 'live',
        rollout_method: str = 'retries',
        max_retries: int = 10,
    ) -> dict:
        """Navigate to a specific goal observation using the graph then a last-mile actor.

        Phase 1 (graph): identifies the goal node from goal_obs and runs the
        executing-Dijkstra rollout() to reach that node.
        Phase 2 (last-mile): from the state at the end of phase 1, runs the live
        actor conditioned on the exact goal_obs until is_at_goal is satisfied or
        max_last_mile_steps is exhausted.

        Not reaching the goal node in phase 1 is counted as failure (last-mile is
        not attempted).

        Args:
            goal_obs:            Observation of the desired goal state (e.g. from a
                                 demonstration end state).
            goal_state:          Corresponding MuJoCo state vector.  If None,
                                 env.obs_to_state() is used to approximate it.
            start_state:         Start MuJoCo state.  Sampled randomly if None.
            max_edge_steps:      Step budget per graph edge.
            max_last_mile_steps: Step budget for the last-mile phase.
            render:              Collect rendered frames if True.

        Returns:
            dict with keys:
                success          (bool)  — is_at_goal satisfied at end,
                graph_success    (bool)  — reached goal_node in graph phase,
                steps            (int)   — total env steps (graph + last-mile),
                path             (list)  — node IDs from graph phase,
                frames           (list)  — rendered frames (empty if render=False).
        """
        if max_edge_steps is None:
            max_edge_steps = self.config.max_steps

        # Resolve goal state for is_at_goal.
        if goal_state is None:
            goal_state = self.env.obs_to_state(goal_obs)

        goal_node = self.perceiver(goal_obs)

        # ---- Phase 1: graph navigation ----
        if rollout_method == 'retries':
            graph_result = self.rollout_with_retries(
                goal_node=goal_node,
                start_state=start_state,
                max_edge_steps=max_edge_steps,
                max_retries=max_retries,
                render=render,
            )
        else:
            graph_result = self.rollout(
                goal_node=goal_node,
                start_state=start_state,
                max_edge_steps=max_edge_steps,
                render=render,
                replay_mode=replay_mode,
            )

        if not graph_result['success']:
            return {
                'success':       False,
                'env_success':   graph_result['env_success'],
                'graph_success': False,
                'steps':         graph_result['steps'],
                'path':          graph_result['path'],
                'frames':        graph_result['frames'],
                'node_labels':   graph_result['node_labels'],
            }

        # ---- Phase 2: last-mile ----
        # Point is_at_goal at the exact goal state rather than the node representative.
        self.env._goal_state = goal_state

        obs    = graph_result['obs']
        frames = graph_result['frames']
        node_labels = graph_result['node_labels']
        env_success = graph_result['env_success']
        last_mile_steps = 0
        success = self.env.is_at_goal(obs, goal_obs)

        goal_t = self._preprocess(goal_obs[None])

        while not success and last_mile_steps < max_last_mile_steps:
            obs_t = self._preprocess(obs[None])
            with torch.no_grad():
                if self.config.residualize and self.action_space_type == 'continuous':
                    base_act = self._get_base_action(obs, goal_node)
                    base_t   = torch.from_numpy(base_act[None]).float().to(self.device)
                    action   = self.actor.sample(obs_t, goal_t, base_t, deterministic=True)
                else:
                    action = self.actor.sample(obs_t, goal_t, deterministic=True)
            action = action.cpu().numpy()[0]

            obs, _, done, _, _ = self.env.step(action)
            last_mile_steps += 1
            if done:
                env_success = True
            if render:
                frames.append(self.env.render())
                node_labels.append(self.perceiver(obs))
            if done:
                break
            success = self.env.is_at_goal(obs, goal_obs)

        return {
            'success':       success,
            'env_success':   env_success,
            'graph_success': True,
            'steps':         graph_result['steps'] + last_mile_steps,
            'path':          graph_result['path'],
            'frames':        frames,
            'node_labels':   node_labels,
        }
