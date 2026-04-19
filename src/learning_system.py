"""Offline contrastive RL learning system.

LearningSystem owns a d_net (MRN), c_net (CostNet), and actor. It alternates between:
  - sample_crtr_batch: draws episodes from the Minari dataset and applies
    CRTR (Contrastive Random Trajectory Repetition) sampling to form
    (state, action, future_state) triples.
  - update_networks: computes encoder and actor losses and steps the optimisers.

Supports both image observations (MiniGrid) and flat observations (Pusher),
and both discrete and continuous action spaces — controlled by config.
"""

from __future__ import annotations

import dataclasses
import math
import random
from dataclasses import dataclass, field
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F

from src.networks import ContinuousActor, CostNet, DiscreteActor, MRN
from src.robomimic_env import RobomimicEnvironment
from src.robomimic_dataset import RobomimicDataset
from src.simple_gridworld import SimpleDataset, SimpleEnvironment


@dataclass
class LearningSystemConfig:
    # MRN architecture (sym_dim doubles as the embedding dim for visualization)
    sym_dim: int = 64
    asym_dim: int = 16
    obs_enc_dim: int = 256
    action_embed_dim: int = 16
    hidden_dims: list[int] = field(default_factory=lambda: [256, 256])
    actor_hidden_dims: list[int] = field(default_factory=lambda: [256, 256])

    # Optimiser
    critic_lr: float = 3e-4
    actor_lr: float = 1e-4
    grad_clip: float = 1.0

    # CRTR sampling
    batch_size: int = 64           # episodes sampled per update
    repetition_factor: int = 4     # repeats per episode → effective batch = batch_size * repetition_factor
    gamma: float = 0.99            # geometric discount for future timestep sampling

    # Actor loss weights
    crl_weight: float = 1
    bc_weight: float = 0.1
    entropy_weight: float = 0.01

    # Device
    device: str = "cuda"


# ---------------------------------------------------------------------------
# Observation pre-processing helpers
# ---------------------------------------------------------------------------

# MiniGrid symbolic channel maximums: object_type ≤ 10, color ≤ 5, state ≤ 2
_MINIGRID_CHANNEL_MAX = np.array([10.0, 5.0, 2.0], dtype=np.float32)


def _preprocess_image_obs(obs_np: np.ndarray, device: torch.device) -> torch.Tensor:
    """(B, H, W, 3) uint8/int → (B, 3, H, W) float in [0, 1].

    uint8 arrays (robomimic RGB images) are divided by 255.
    Integer arrays with small values (MiniGrid symbolic) use per-channel maxima.
    """
    if obs_np.dtype == np.uint8:
        obs = obs_np.astype(np.float32) / 255.0
    else:
        obs = obs_np.astype(np.float32) / _MINIGRID_CHANNEL_MAX  # broadcast over HW
    obs = np.transpose(obs, (0, 3, 1, 2))                        # HWC → CHW
    return torch.from_numpy(obs).to(device)


def _preprocess_flat_obs(obs_np: np.ndarray, device: torch.device) -> torch.Tensor:
    """(B, D) float → FloatTensor."""
    return torch.from_numpy(obs_np.astype(np.float32)).to(device)


# ---------------------------------------------------------------------------
# LearningSystem
# ---------------------------------------------------------------------------


class LearningSystem:
    """Offline contrastive RL system for goal-conditioned policy learning.

    Args:
        dataset: A loaded Minari dataset.
        env:     Optional Environment wrapper (used for rollout / evaluation).
        config:  Hyperparameter configuration.
    """

    def __init__(
        self,
        dataset: RobomimicDataset | SimpleDataset,
        env: RobomimicEnvironment | SimpleEnvironment | None = None,
        config: LearningSystemConfig | None = None,
    ):
        if config is None:
            config = LearningSystemConfig()
        self.config = config
        self.dataset = dataset
        self.device = torch.device(
            config.device if torch.cuda.is_available() else "cpu"
        )

        # --- Infer obs shape and action space from dataset ---
        sample_ep = next(dataset.iterate_episodes())

        raw_obs = sample_ep.observations
        if isinstance(raw_obs, dict):
            # e.g. MiniGrid: {'image': (T+1, H, W, 3), 'direction': ..., ...}
            self._obs_key = 'image'
            first_obs = raw_obs['image'][0]
        else:
            # e.g. Pusher: (T+1, D) flat array
            self._obs_key = None
            first_obs = raw_obs[0]

        self.obs_shape: tuple[int, ...] = first_obs.shape  # e.g. (H, W, 3) or (D,)
        self._is_image_obs = len(self.obs_shape) == 3

        action_space = dataset.action_space
        if isinstance(action_space, gym.spaces.Discrete):
            self.action_space_type = 'discrete'
            self.num_actions_or_dim = int(action_space.n)
        elif isinstance(action_space, gym.spaces.Box):
            self.action_space_type = 'continuous'
            self.num_actions_or_dim = int(np.prod(action_space.shape))
        else:
            raise ValueError(f"Unsupported action space: {type(action_space)}")

        print(f"obs_shape:         {self.obs_shape}")
        print(f"action_space_type: {self.action_space_type}")
        print(f"num_actions/dim:   {self.num_actions_or_dim}")
        print(f"device:            {self.device}")

        # --- Build networks ---
        self.d_net = MRN(
            obs_shape=self.obs_shape,
            action_space_type=self.action_space_type,
            num_actions_or_dim=self.num_actions_or_dim,
            sym_dim=config.sym_dim,
            asym_dim=config.asym_dim,
            obs_enc_dim=config.obs_enc_dim,
            action_embed_dim=config.action_embed_dim,
            hidden_dims=config.hidden_dims,
        ).to(self.device)

        self.c_net = CostNet(
            obs_shape=self.obs_shape,
            obs_enc_dim=config.obs_enc_dim,
            hidden_dims=config.hidden_dims,
        ).to(self.device)

        if self.action_space_type == 'discrete':
            self.actor = DiscreteActor(
                obs_shape=self.obs_shape,
                num_actions=self.num_actions_or_dim,
                obs_enc_dim=config.obs_enc_dim,
                hidden_dims=config.actor_hidden_dims,
            ).to(self.device)
        else:
            self.actor = ContinuousActor(
                obs_shape=self.obs_shape,
                action_dim=self.num_actions_or_dim,
                obs_enc_dim=config.obs_enc_dim,
                hidden_dims=config.actor_hidden_dims,
            ).to(self.device)

        # --- Optimisers ---
        self.critic_optimizer = torch.optim.Adam(
            list(self.d_net.parameters()) + list(self.c_net.parameters()),
            lr=config.critic_lr,
        )
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=config.actor_lr
        )

        self.training_step = 0
        self.env = env

    # -----------------------------------------------------------------------
    # Observation helpers
    # -----------------------------------------------------------------------

    def _extract_obs(self, raw_obs) -> np.ndarray:
        """Extract the relevant array from an episode's observation field."""
        if self._obs_key is not None:
            return raw_obs[self._obs_key]
        return raw_obs

    def _preprocess(self, obs_np: np.ndarray) -> torch.Tensor:
        """Normalise and move a batch of observations to device."""
        if self._is_image_obs:
            return _preprocess_image_obs(obs_np, self.device)
        return _preprocess_flat_obs(obs_np, self.device)

    # -----------------------------------------------------------------------
    # Inference
    # -----------------------------------------------------------------------

    def get_action(self, obs: np.ndarray, goal_obs: np.ndarray):
        """Return a deterministic action given a single current and goal observation.

        Args:
            obs:      Single observation array, e.g. (H, W, 3) or (D,).
            goal_obs: Single goal observation array, same shape as obs.

        Returns:
            int for discrete actions, np.ndarray for continuous.
        """
        state_t = self._preprocess(obs[None])       # add batch dim → (1, ...)
        goal_t  = self._preprocess(goal_obs[None])

        with torch.no_grad():
            if self.action_space_type == 'discrete':
                logits = self.actor(state_t, goal_t)   # (1, num_actions)
                action = int(logits.argmax(dim=-1).item())
            else:
                action = self.actor.sample(state_t, goal_t, deterministic=True)
                action = action.squeeze(0).cpu().numpy()

        return action

    # -----------------------------------------------------------------------
    # CRTR batch sampling
    # -----------------------------------------------------------------------

    def _sample_geometric_future(self, t: int, T: int) -> int:
        """Sample t' > t geometrically; t' ∈ [t+1, T] (obs indices)."""
        max_k = T - t
        if max_k == 0:
            return t
        probs = [(1 - self.config.gamma) * (self.config.gamma ** k) for k in range(max_k)]
        total = sum(probs)
        if total > 0:
            probs = [p / total for p in probs]
            k = random.choices(range(max_k), weights=probs)[0]
        else:
            k = 0
        return t + k + 1

    def sample_crtr_batch(self) -> dict:
        """Sample a CRTR batch from the offline dataset.

        Returns a dict with:
            states:  (N, ...) — obs at time t
            actions: (N, ...) — action at time t
            goals:   (N, ...) — obs at time t' > t (future state as goal)

        where N = batch_size * repetition_factor.
        """
        episodes = self.dataset.sample_episodes(n_episodes=self.config.batch_size)
        states, actions, goals = [], [], []

        for ep in episodes:
            obs = self._extract_obs(ep.observations)  # (T+1, ...)
            acts = ep.actions                          # (T, ...) or (T,)
            T = len(acts)

            for _ in range(self.config.repetition_factor):
                t = random.randint(0, T - 1)
                t_prime = self._sample_geometric_future(t, T)
                states.append(obs[t])
                actions.append(acts[t])
                goals.append(obs[t_prime])

        return {
            'states':  np.stack(states),
            'actions': np.stack(actions),
            'goals':   np.stack(goals),
        }

    # -----------------------------------------------------------------------
    # Network updates
    # -----------------------------------------------------------------------

    def _zero_acts(self, B: int) -> torch.Tensor:
        """Return a batch of zero actions (used as goal-side input to d_net)."""
        if self.action_space_type == 'discrete':
            return torch.zeros(B, dtype=torch.long, device=self.device)
        return torch.zeros(B, self.num_actions_or_dim, dtype=torch.float32, device=self.device)

    def update_networks(self, batch: dict) -> tuple[float, float, dict]:
        """One gradient step on the critic (d_net + c_net) and actor.

        Returns:
            (critic_loss, actor_loss, diagnostics) as (float, float, dict).
        """
        state_obs  = self._preprocess(batch['states'])
        goal_obs   = self._preprocess(batch['goals'])
        demo_acts  = batch['actions']

        if self.action_space_type == 'discrete':
            demo_acts_t = torch.from_numpy(demo_acts.astype(np.int64)).to(self.device)
        else:
            demo_acts_t = torch.from_numpy(demo_acts.astype(np.float32)).to(self.device)

        B = state_obs.shape[0]

        # === Critic update (CMD InfoNCE) ===
        self.critic_optimizer.zero_grad()

        # Source: real (state, action) pairs.
        # Goal: real goal obs but with randomly permuted actions — network learns to
        #       ignore goal-side actions, making the metric action-invariant at goals.
        perm = torch.randperm(B, device=self.device)
        goal_acts_perm = demo_acts_t[perm]

        phi_sym, phi_asym = self.d_net.encode(state_obs, demo_acts_t)       # (B, sym/asym)
        psi_sym, psi_asym = self.d_net.encode(goal_obs,  goal_acts_perm)    # (B, sym/asym)

        # Expand to B×B, compute energy[i,j] = c(goal_j) + d_net(sa_i, goal_j)
        phi_sym_e  = phi_sym.unsqueeze(1).expand(B, B, -1).reshape(B*B, -1)
        phi_asym_e = phi_asym.unsqueeze(1).expand(B, B, -1).reshape(B*B, -1)
        psi_sym_e  = psi_sym.unsqueeze(0).expand(B, B, -1).reshape(B*B, -1)
        psi_asym_e = psi_asym.unsqueeze(0).expand(B, B, -1).reshape(B*B, -1)

        goal_obs_e = goal_obs.unsqueeze(0).expand(B, B, *goal_obs.shape[1:]).reshape(B*B, *goal_obs.shape[1:])
        c_vals     = self.c_net(goal_obs_e)                                  # (B*B, 1)

        neg_dists = self.d_net._dist_from_encodings(phi_sym_e, phi_asym_e, psi_sym_e, psi_asym_e)
        energy    = (c_vals + neg_dists).reshape(B, B)

        labels = torch.arange(B, device=self.device)
        critic_loss = F.cross_entropy(energy, labels) + F.cross_entropy(energy.T, labels)

        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.d_net.parameters()) + list(self.c_net.parameters()),
            self.config.grad_clip,
        )
        self.critic_optimizer.step()

        # === Actor update ===
        self.actor_optimizer.zero_grad()

        sampled_acts  = self.actor.sample(state_obs, goal_obs)
        zero_goal_acts = self._zero_acts(B)

        # CMD actor: maximize d_net((state, sampled_act), (goal, 0)).
        # d_net / c_net are detached; only actor receives gradients.
        with torch.no_grad():
            psi_sym_a, psi_asym_a = self.d_net.encode(goal_obs, zero_goal_acts)
        phi_sym_a, phi_asym_a = self.d_net.encode(state_obs, sampled_acts)
        crl_loss = -self.d_net._dist_from_encodings(
            phi_sym_a, phi_asym_a, psi_sym_a.detach(), psi_asym_a.detach()
        ).mean()

        bc_loss  = -self.actor.get_log_prob(state_obs, goal_obs, demo_acts_t).mean()
        ent_loss = -self.actor.entropy(state_obs, goal_obs).mean()

        actor_loss = (
            self.config.crl_weight    * crl_loss
            + self.config.bc_weight   * bc_loss
            + self.config.entropy_weight * ent_loss
        )

        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.config.grad_clip)
        self.actor_optimizer.step()

        self.training_step += 1
        return critic_loss.item(), actor_loss.item(), {
            'critic': critic_loss.item(),
            'crl':    crl_loss.item(),
            'bc':     bc_loss.item(),
            'ent':    ent_loss.item(),
        }

    # -----------------------------------------------------------------------
    # Training loop
    # -----------------------------------------------------------------------

    def train(
        self,
        num_steps: int,
        log_every: int = 100,
        patience: int | None = None,
        min_delta: float = 1e-4,
    ) -> list[dict]:
        """Run num_steps update steps, printing progress every log_every steps.

        Args:
            num_steps:  Maximum number of gradient steps.
            log_every:  Print interval.
            patience:   Stop early if critic loss has not improved by more than
                        min_delta within this many steps.  None disables early stopping.
            min_delta:  Minimum improvement to reset the patience counter.

        Returns:
            loss_history: list of dicts with keys step, critic, actor, crl, bc, ent,
                          recorded every log_every steps.
        """
        best_loss = float('inf')
        steps_without_improvement = 0
        loss_history: list[dict] = []

        for step in range(num_steps):
            batch = self.sample_crtr_batch()
            critic_loss, actor_loss, diag = self.update_networks(batch)

            if step % log_every == 0:
                print(
                    f"step {step:6d} | "
                    f"critic={critic_loss:.4f} | "
                    f"actor={actor_loss:.4f} "
                    f"(crl={diag['crl']:.3f} bc={diag['bc']:.3f} ent={diag['ent']:.3f})"
                )
                loss_history.append({
                    'step':   step,
                    'critic': critic_loss,
                    'actor':  actor_loss,
                    'crl':    diag['crl'],
                    'bc':     diag['bc'],
                    'ent':    diag['ent'],
                })

            if patience is not None:
                if critic_loss < best_loss - min_delta:
                    best_loss = critic_loss
                    steps_without_improvement = 0
                else:
                    steps_without_improvement += 1
                    if steps_without_improvement >= patience:
                        print(f"Early stopping at step {step} "
                              f"(no improvement in critic loss for {patience} steps).")
                        break

        return loss_history

    # -----------------------------------------------------------------------
    # Evaluation
    # -----------------------------------------------------------------------

    def rollout(self, max_steps: int = 100) -> dict:
        """Roll out the policy from a randomly-reset start state towards a goal.

        Args:
            max_steps: Maximum number of steps before the rollout terminates.

        Returns:
            dict with keys:
                frames:  list of RGB arrays, one per step
                obs:     list of normalised obs arrays, one per step (includes start)
                actions: list of actions taken
                success: bool — whether is_at_goal was satisfied before max_steps
        """
        if self.env is None:
            raise RuntimeError(
                "LearningSystem was created without an Environment. "
                "Pass env=Environment(dataset) at initialisation to use rollout()."
            )

        curr_obs, goal_obs = self.env.reset()
        frames, obs_list, actions_list, success = [], [curr_obs], [], False

        print("Starting rollout...")
        for _ in range(max_steps):
            frames.append(self.env.render())
            action = self.get_action(curr_obs, goal_obs)
            print(f"Action taken: {action}")
            curr_obs, _, terminated, truncated, _ = self.env.step(action)
            actions_list.append(action)
            obs_list.append(curr_obs)
            if self.env.is_at_goal(curr_obs, goal_obs):
                print("Goal reached!")
                success = True
                break
            if terminated or truncated:
                print("Episode ended before reaching goal.")
                break

        return {
            'frames':  frames,
            'obs':     obs_list,
            'actions': actions_list,
            'success': success,
            'goal_obs': goal_obs,
        }

    # -----------------------------------------------------------------------
    # TAMP structure construction
    # -----------------------------------------------------------------------

    def _encode_dataset(
        self,
        batch_size: int = 512,
        max_obs: int | None = None,
    ) -> tuple[np.ndarray, list]:
        """Encode dataset observations with d_net (zero action) → sym embeddings.

        Used for visualization / PCA only — create_perceiver does not call this.

        Args:
            batch_size: Encoding batch size.
            max_obs:    If set, randomly subsample before encoding.

        Returns:
            sym_embs: (N, sym_dim) — d_net.encode(obs, 0)[0]
            obs_list: list of N raw observations aligned with embs.
        """
        if self.dataset is None:
            raise RuntimeError(
                "dataset is required; pass dataset= when calling LearningSystem.load()."
            )
        obs_list: list[np.ndarray] = []
        for ep in self.dataset.iterate_episodes():
            obs_list.extend(self._extract_obs(ep.observations))

        if max_obs is not None and len(obs_list) > max_obs:
            total = len(obs_list)
            idx = np.random.choice(total, max_obs, replace=False)
            obs_list = [obs_list[i] for i in idx]
            print(f"  Subsampled {max_obs} observations from {total} total.")

        all_sym = []
        for i in range(0, len(obs_list), batch_size):
            batch = np.stack(obs_list[i : i + batch_size])
            obs_t = self._preprocess(batch)
            act_t = self._zero_acts(len(batch))
            with torch.no_grad():
                sym, _ = self.d_net.encode(obs_t, act_t)
            all_sym.append(sym.cpu().numpy())

        return np.concatenate(all_sym, axis=0), obs_list

    def _encode_obs_batch(
        self,
        obs_list: list[np.ndarray],
        batch_size: int = 512,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Encode a list of observations with zero actions → (sym_embs, asym_embs)."""
        all_sym, all_asym = [], []
        for i in range(0, len(obs_list), batch_size):
            batch = np.stack(obs_list[i : i + batch_size])
            obs_t = self._preprocess(batch)
            act_t = self._zero_acts(len(batch))
            with torch.no_grad():
                sym, asym = self.d_net.encode(obs_t, act_t)
            all_sym.append(sym.cpu().numpy())
            all_asym.append(asym.cpu().numpy())
        return np.concatenate(all_sym, axis=0), np.concatenate(all_asym, axis=0)

    def _sym_dist_matrix(
        self,
        sym_a:  np.ndarray,
        asym_a: np.ndarray,
        sym_b:  np.ndarray,
        asym_b: np.ndarray,
    ) -> np.ndarray:
        """Symmetric CMD distance matrix D[i,j] = ½(d(a_i→b_j) + d(b_j→a_i)).

        Args:
            sym_a, asym_a:  (M, *) pre-encoded query embeddings.
            sym_b, asym_b:  (N, *) pre-encoded candidate embeddings.

        Returns:
            (M, N) symmetric distance matrix.
        """
        M, N = len(sym_a), len(sym_b)

        def _t(x):
            return torch.from_numpy(x.astype(np.float32))  # CPU — avoids large GPU allocs

        sym_a_t, asym_a_t = _t(sym_a), _t(asym_a)
        sym_b_t, asym_b_t = _t(sym_b), _t(asym_b)

        sym_a_e  = sym_a_t.unsqueeze(1).expand(M, N, -1).reshape(M*N, -1)
        asym_a_e = asym_a_t.unsqueeze(1).expand(M, N, -1).reshape(M*N, -1)
        sym_b_e  = sym_b_t.unsqueeze(0).expand(M, N, -1).reshape(M*N, -1)
        asym_b_e = asym_b_t.unsqueeze(0).expand(M, N, -1).reshape(M*N, -1)

        with torch.no_grad():
            d_fwd = -self.d_net._dist_from_encodings(sym_a_e, asym_a_e, sym_b_e, asym_b_e)
            d_bwd = -self.d_net._dist_from_encodings(sym_b_e, asym_b_e, sym_a_e, asym_a_e)

        return ((d_fwd + d_bwd) / 2.0).reshape(M, N).cpu().numpy()

    def _kmedoids(
        self,
        sym_embs:  np.ndarray,
        asym_embs: np.ndarray,
        K: int,
        max_iter: int = 100,
    ) -> np.ndarray:
        """K-medoids using D(x,z) = ½(d_net(x→z) + d_net(z→x)).

        Returns:
            medioid_idx: (K,) int indices into sym_embs / asym_embs.
        """
        N = len(sym_embs)
        assert K <= N, f"K={K} must be <= number of observations {N}"

        rng = np.random.default_rng()
        medioid_idx = rng.choice(N, K, replace=False)

        for iteration in range(max_iter):
            dists = self._sym_dist_matrix(
                sym_embs, asym_embs, sym_embs[medioid_idx], asym_embs[medioid_idx]
            )  # (N, K)
            assignments = dists.argmin(axis=1)

            new_medioid_idx = medioid_idx.copy()
            for k in range(K):
                cluster = np.where(assignments == k)[0]
                if len(cluster) == 0:
                    continue
                costs = self._sym_dist_matrix(
                    sym_embs[cluster], asym_embs[cluster],
                    sym_embs[cluster], asym_embs[cluster],
                ).sum(axis=0)
                new_medioid_idx[k] = cluster[costs.argmin()]

            if np.array_equal(np.sort(new_medioid_idx), np.sort(medioid_idx)):
                print(f"  K-medoids converged after {iteration + 1} iterations.")
                break
            medioid_idx = new_medioid_idx
        else:
            print(f"  K-medoids reached max_iter={max_iter}.")

        return medioid_idx

    def create_perceiver(self, K: int, max_obs: int | None = None):
        """Build Per: obs → node_id via K-medoids on CMD symmetric distances.

        Does not call _encode_dataset — encodes obs inline with zero actions.

        Args:
            K:       Number of nodes.
            max_obs: If set, subsample at most this many observations before clustering.

        Returns:
            perceiver: Callable[[np.ndarray], int] → node id in [0, K-1].
                       Attributes: node_embeddings (K, sym_dim), node_obs (K, ...),
                       node_states (K, ...), obs_to_state.
        """
        if self.dataset is None:
            raise RuntimeError("dataset is required for create_perceiver.")

        # --- Collect obs and states ---
        print(f"Collecting dataset observations (K={K})...")
        obs_list: list[np.ndarray] = []
        states_list: list[np.ndarray] = []
        for ep in self.dataset.iterate_episodes():
            ep_obs = self._extract_obs(ep.observations)  # (T+1, ...)
            for t, ob in enumerate(ep_obs[:-1]):
                obs_list.append(ob)
                states_list.append(ep.states[t])
            obs_list.append(ep_obs[-1])
            states_list.append(ep.states[-1])  # terminal approx

        if max_obs is not None and len(obs_list) > max_obs:
            total = len(obs_list)
            idx = np.random.choice(total, max_obs, replace=False)
            obs_list    = [obs_list[i]    for i in idx]
            states_list = [states_list[i] for i in idx]
            print(f"  Subsampled {max_obs} from {total} observations.")

        # --- Encode with zero actions ---
        print(f"  Encoding {len(obs_list)} observations...")
        sym_embs, asym_embs = self._encode_obs_batch(obs_list)

        # --- K-medoids ---
        print("  Running K-medoids...")
        medioid_idx    = self._kmedoids(sym_embs, asym_embs, K)
        node_sym       = sym_embs[medioid_idx].copy()                         # (K, sym_dim)
        node_asym      = asym_embs[medioid_idx].copy()                        # (K, asym_dim)
        node_obs       = np.stack([obs_list[i]    for i in medioid_idx])      # (K, ...)
        node_states    = np.stack([states_list[i] for i in medioid_idx])      # (K, ...)
        print(f"  Done. Medioid indices: {medioid_idx.tolist()}")

        # --- obs→state map (full dataset, unsubsampled) ---
        print("  Building obs→state mapping...")
        obs_to_state: dict[bytes, np.ndarray] = {}
        for ep in self.dataset.iterate_episodes():
            ep_obs = self._extract_obs(ep.observations)
            for t, ob in enumerate(ep_obs[:-1]):
                obs_to_state[ob.tobytes()] = ep.states[t]
            obs_to_state[ep_obs[-1].tobytes()] = ep.states[-1]
        print(f"  obs→state map has {len(obs_to_state)} entries.")

        # --- Perceiver closure ---
        def perceiver(obs: np.ndarray) -> int:
            obs_t = self._preprocess(obs[None])
            act_t = self._zero_acts(1)
            with torch.no_grad():
                sym_q, asym_q = self.d_net.encode(obs_t, act_t)
            # print("Shapes:", sym_q.shape, asym_q.shape, node_sym.shape, node_asym.shape)
            dists = self._sym_dist_matrix(
                sym_q.cpu().numpy(), asym_q.cpu().numpy(), node_sym, node_asym
            )[0]
            return int(dists.argmin())

        perceiver.node_embeddings = node_sym     # (K, sym_dim) — used for PCA / visualization
        perceiver.node_asym       = node_asym   # (K, asym_dim) — needed to reconstruct perceiver
        perceiver.node_obs        = node_obs
        perceiver.node_states     = node_states
        perceiver.obs_to_state    = obs_to_state
        return perceiver

    def _estimate_edge_distance(
        self,
        source_obs: np.ndarray,
        goal_obs: np.ndarray,
    ) -> float:
        """CMD-formula distance between two obs (medoid→medoid), dist_scale=1.

        Uses zero actions on both sides — distance is in model space.
        Formula: max(0, (1 / log γ) * d_net(src→goal))
        Since log γ < 0 and d_net returns negative values, the product is positive.
        """
        src_t  = self._preprocess(source_obs[None])
        goal_t = self._preprocess(goal_obs[None])
        zero1  = self._zero_acts(1)
        with torch.no_grad():
            phi_sym, phi_asym = self.d_net.encode(src_t, zero1)
            psi_sym, psi_asym = self.d_net.encode(goal_t, zero1)
            neg_d = self.d_net._dist_from_encodings(phi_sym, phi_asym, psi_sym, psi_asym)
        return max(0.0, (1.0 / math.log(self.config.gamma)) * float(neg_d.item()))

    def create_graph(
        self,
        perceiver,
        env,
        num_rollouts: int = 10,
        max_steps: int = 100,
        success_threshold: float = 0.5,
    ) -> dict:
        """Build a graph of reliable inter-node edges.

        For each ordered pair (A, B):
          - Creates segmented policy pi_AB(x) = actor(x, medoid_obs_B)
          - Samples num_rollouts starting states from dataset observations in node A
          - Rolls out pi_AB; marks success when perceiver assigns obs to node B
          - Adds edge if success_rate >= success_threshold

        Args:
            perceiver:         Callable from create_perceiver, with .node_obs attribute.
            env:               Environment with reset_to / step interface.
            num_rollouts:      Rollouts per edge to estimate success rate.
            max_steps:         Rollout horizon before declaring failure.
            success_threshold: Minimum success rate to include an edge.

        Returns:
            graph:           {(A, B): (pi_AB, avg_steps, estimated_dist)} where avg_steps is
                             the mean rollout length and estimated_dist is the CMD model-space
                             distance (dist_scale=1).
            node_state_pool: list of K lists, each containing the dataset obs classified
                             into that node.  Pass directly to OnlineSystem to avoid
                             re-classifying the dataset.
            edge_stats:      {(A, B): success_rate} for every tested edge (including failed
                             ones).  Useful for diagnosing why edges were dropped.
        """
        K = len(perceiver.node_obs)

        # --- Classify all dataset observations into nodes ---
        print("Classifying dataset observations into nodes...")
        node_state_pool: list[list[np.ndarray]] = [[] for _ in range(K)]
        for ep in self.dataset.iterate_episodes():
            for obs in self._extract_obs(ep.observations):
                node_state_pool[perceiver(obs)].append(obs)
        for k in range(K):
            print(f"  Node {k}: {len(node_state_pool[k])} states")

        graph: dict = {}
        edge_stats: dict[tuple[int, int], float] = {}  # success rate for every tested edge

        for A in range(K):
            if not node_state_pool[A]:
                print(f"  Node {A} has no dataset states — skipping all edges from it.")
                continue

            for B in range(K):
                if A == B:
                    continue

                print(f"Testing edge ({A} -> {B})...")

                goal_obs_B   = perceiver.node_obs[B]     # medoid obs — used to condition actor
                goal_state_B = perceiver.node_states[B]  # medoid state — used for env.reset_to
                
                # Segmented policy: goal is always the medoid of node B
                def _make_pi(goal_obs_b: np.ndarray):
                    def pi(obs: np.ndarray) -> np.ndarray:
                        obs_t  = self._preprocess(obs[None])
                        goal_t = self._preprocess(goal_obs_b[None])
                        with torch.no_grad():
                            action = self.actor.sample(obs_t, goal_t, deterministic=True)
                        return action.cpu().numpy()[0]
                    return pi

                pi_AB = _make_pi(goal_obs_B)

                # Sample starting states from node A (with replacement if pool is small)
                pool = node_state_pool[A]
                starts = [pool[i % len(pool)] for i in
                          np.random.choice(len(pool), num_rollouts, replace=True)]

                successes: list[float] = []
                step_counts: list[int] = []

                for start_obs in starts:
                    start_state = perceiver.obs_to_state.get(
                        start_obs.tobytes(), env.obs_to_state(start_obs)
                    )
                    obs, _ = env.reset_to(start_state, goal_state_B)
                    success = False
                    for step in range(max_steps):
                        action = pi_AB(obs)
                        obs, _, done, _, _ = env.step(action)
                        if perceiver(obs) == B:
                            success = True
                            step_counts.append(step + 1)
                            break
                    if not success:
                        step_counts.append(max_steps)
                    successes.append(float(success))

                success_rate = float(np.mean(successes))
                avg_steps    = float(np.mean(step_counts))

                edge_stats[(A, B)] = success_rate

                if success_rate >= success_threshold:
                    estimated_dist = self._estimate_edge_distance(perceiver.node_obs[A], goal_obs_B)
                    graph[(A, B)] = (pi_AB, avg_steps, estimated_dist)
                    print(f"  Edge ({A} -> {B}): success={success_rate:.2f}, avg_steps={avg_steps:.1f}, "
                          f"est_dist={estimated_dist:.2f}  [added]")
                else:
                    print(f"  Edge ({A} -> {B}): success={success_rate:.2f}  [skipped]")

        print(f"Graph complete: {len(graph)} reliable edges out of {K*(K-1)} tested.")
        return graph, node_state_pool, edge_stats

    # -----------------------------------------------------------------------
    # Persistence
    # -----------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Save networks and metadata to a checkpoint file.

        Args:
            path: File path (typically *.pt or *.ckpt).
        """
        checkpoint = {
            'config':              dataclasses.asdict(self.config),
            'obs_shape':           self.obs_shape,
            'action_space_type':   self.action_space_type,
            'num_actions_or_dim':  self.num_actions_or_dim,
            '_obs_key':            self._obs_key,
            '_is_image_obs':       self._is_image_obs,
            'training_step':       self.training_step,
            'd_net':               self.d_net.state_dict(),
            'c_net':               self.c_net.state_dict(),
            'actor':               self.actor.state_dict(),
        }
        torch.save(checkpoint, path)
        print(f"Saved LearningSystem to {path} (step {self.training_step})")

    @classmethod
    def load(
        cls,
        path: str | Path,
        dataset: RobomimicDataset | SimpleDataset | None = None,
        env: RobomimicEnvironment | SimpleEnvironment | None = None,
        device: str | None = None,
    ) -> LearningSystem:
        """Load a LearningSystem from a checkpoint file.

        The dataset is not required for inference; pass it if you want to
        continue training after loading.

        Args:
            path:    Path to a checkpoint saved by :meth:`save`.
            dataset: Optional dataset for continued training.
            env:     Optional Environment for rollouts.
            device:  Override the device (e.g. 'cpu' for visualization on a
                     busy GPU node). Defaults to the device stored in config.

        Returns:
            A fully initialised LearningSystem with weights restored.
        """
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        config = LearningSystemConfig(**checkpoint['config'])

        # Bypass __init__ so we don't need a live dataset.
        obj: LearningSystem = cls.__new__(cls)
        obj.config            = config
        obj.dataset           = dataset
        obj.env               = env
        if device is not None:
            obj.device = torch.device(device)
        else:
            obj.device = torch.device(
                config.device if torch.cuda.is_available() else 'cpu'
            )
        obj.obs_shape          = tuple(checkpoint['obs_shape'])
        obj.action_space_type  = checkpoint['action_space_type']
        obj.num_actions_or_dim = checkpoint['num_actions_or_dim']
        obj._obs_key           = checkpoint['_obs_key']
        obj._is_image_obs      = checkpoint['_is_image_obs']
        obj.training_step      = checkpoint['training_step']

        # Rebuild networks and load weights.
        obj.d_net = MRN(
            obs_shape=obj.obs_shape,
            action_space_type=obj.action_space_type,
            num_actions_or_dim=obj.num_actions_or_dim,
            sym_dim=config.sym_dim,
            asym_dim=config.asym_dim,
            obs_enc_dim=config.obs_enc_dim,
            action_embed_dim=config.action_embed_dim,
            hidden_dims=config.hidden_dims,
        ).to(obj.device)
        obj.d_net.load_state_dict(checkpoint['d_net'])

        obj.c_net = CostNet(
            obs_shape=obj.obs_shape,
            obs_enc_dim=config.obs_enc_dim,
            hidden_dims=config.hidden_dims,
        ).to(obj.device)
        obj.c_net.load_state_dict(checkpoint['c_net'])

        if obj.action_space_type == 'discrete':
            obj.actor = DiscreteActor(
                obs_shape=obj.obs_shape,
                num_actions=obj.num_actions_or_dim,
                obs_enc_dim=config.obs_enc_dim,
                hidden_dims=config.actor_hidden_dims,
            ).to(obj.device)
        else:
            obj.actor = ContinuousActor(
                obs_shape=obj.obs_shape,
                action_dim=obj.num_actions_or_dim,
                obs_enc_dim=config.obs_enc_dim,
                hidden_dims=config.actor_hidden_dims,
            ).to(obj.device)
        obj.actor.load_state_dict(checkpoint['actor'])

        # Fresh optimisers (optimizer state is not saved).
        obj.critic_optimizer = torch.optim.Adam(
            list(obj.d_net.parameters()) + list(obj.c_net.parameters()),
            lr=config.critic_lr,
        )
        obj.actor_optimizer = torch.optim.Adam(
            obj.actor.parameters(), lr=config.actor_lr,
        )

        print(f"Loaded LearningSystem from {path} (step {obj.training_step})")
        return obj
