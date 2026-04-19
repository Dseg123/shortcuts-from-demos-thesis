"""Neural network modules for offline contrastive RL.

Supports both image observations (MiniGrid) and flat observations (Pusher),
and both discrete and continuous action spaces. The three main classes
(StateActionEncoder, GoalEncoder, DiscreteActor / ContinuousActor) share the
same external call interface regardless of obs/action type.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Internal building blocks
# ---------------------------------------------------------------------------

class _ImageObsEncoder(nn.Module):
    """CNN for symbolic MiniGrid images (H x W x 3 → out_dim).

    Input is expected to be (B, 3, H, W) float in [0, 1].
    Two MaxPool layers so the spatial dimension shrinks before the linear head.
    """

    def __init__(self, img_h: int, img_w: int, out_dim: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
        )
        with torch.no_grad():
            flat_dim = self.conv(torch.zeros(1, 3, img_h, img_w)).flatten(1).shape[1]
        self.fc = nn.Linear(flat_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.conv(x).flatten(1))


class _FlatObsEncoder(nn.Module):
    """MLP for flat (1-D) observation vectors (obs_dim → out_dim)."""

    def __init__(self, obs_dim: int, out_dim: int, hidden_dims: list[int] | None = None):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 256]
        layers: list[nn.Module] = []
        prev = obs_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def _make_obs_encoder(
    obs_shape: tuple[int, ...],
    out_dim: int,
    hidden_dims: list[int] | None = None,
) -> nn.Module:
    """Factory: returns CNN encoder for 3-D obs, MLP encoder for 1-D obs."""
    if len(obs_shape) == 3:
        h, w, _ = obs_shape
        return _ImageObsEncoder(h, w, out_dim)
    elif len(obs_shape) == 1:
        return _FlatObsEncoder(obs_shape[0], out_dim, hidden_dims)
    else:
        raise ValueError(f"Unsupported obs_shape {obs_shape}")


class _DiscreteActionEncoder(nn.Module):
    def __init__(self, num_actions: int, embed_dim: int):
        super().__init__()
        self.embed = nn.Embedding(num_actions, embed_dim)

    def forward(self, actions: torch.Tensor) -> torch.Tensor:
        return self.embed(actions)


class _ContinuousActionEncoder(nn.Module):
    def __init__(self, action_dim: int, embed_dim: int):
        super().__init__()
        self.linear = nn.Linear(action_dim, embed_dim)

    def forward(self, actions: torch.Tensor) -> torch.Tensor:
        return self.linear(actions)


# ---------------------------------------------------------------------------
# CMD distance / cost networks
# ---------------------------------------------------------------------------


class MRN(nn.Module):
    """Metric Residual Network for CMD.

    Takes two (obs, action) pairs and computes a quasi-metric distance via
    shared symmetric + asymmetric encoders.  Because both sides go through the
    same weights, d(s, s) = 0 by construction.

    Returns *negative* distance (energy convention): larger ↔ closer.

    Usage:
        # Single forward pass (small batches)
        neg_dist = d_net(obs1, act1, obs2, act2)

        # Efficient B×B matrix (encode once, combine)
        sym1, asym1 = d_net.encode(obs1, act1)
        sym2, asym2 = d_net.encode(obs2, act2)
        neg_dist_matrix = d_net._dist_from_encodings(
            sym1_exp, asym1_exp, sym2_exp, asym2_exp
        )
    """

    def __init__(
        self,
        obs_shape: tuple[int, ...],
        action_space_type: str,
        num_actions_or_dim: int,
        sym_dim: int = 64,
        asym_dim: int = 16,
        obs_enc_dim: int = 256,
        action_embed_dim: int = 16,
        hidden_dims: list[int] | None = None,
    ):
        super().__init__()

        def _make_act_enc() -> nn.Module:
            if action_space_type == 'discrete':
                return _DiscreteActionEncoder(num_actions_or_dim, action_embed_dim)
            return _ContinuousActionEncoder(num_actions_or_dim, action_embed_dim)

        fused_dim = obs_enc_dim + action_embed_dim

        # Symmetric branch
        self.sym_obs_enc = _make_obs_encoder(obs_shape, obs_enc_dim, hidden_dims)
        self.sym_act_enc = _make_act_enc()
        self.sym_head    = nn.Linear(fused_dim, sym_dim)

        # Asymmetric branch
        self.asym_obs_enc = _make_obs_encoder(obs_shape, obs_enc_dim, hidden_dims)
        self.asym_act_enc = _make_act_enc()
        self.asym_head    = nn.Linear(fused_dim, asym_dim)

    # ------------------------------------------------------------------

    def _sym(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        return self.sym_head(
            torch.cat([self.sym_obs_enc(obs), self.sym_act_enc(act)], dim=-1)
        )

    def _asym(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        return self.asym_head(
            torch.cat([self.asym_obs_enc(obs), self.asym_act_enc(act)], dim=-1)
        )

    def encode(
        self, obs: torch.Tensor, act: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (sym, asym) embeddings for an (obs, act) pair.

        Useful for pre-computing B encodings before assembling a B×B matrix,
        and for extracting sym embeddings for visualization / PCA.
        """
        return self._sym(obs, act), self._asym(obs, act)

    def _dist_from_encodings(
        self,
        sym1:  torch.Tensor,
        asym1: torch.Tensor,
        sym2:  torch.Tensor,
        asym2: torch.Tensor,
    ) -> torch.Tensor:
        """Negative quasi-metric from pre-computed (sym, asym) pairs.

        Args:
            sym1, asym1: (B, sym_dim / asym_dim) for source.
            sym2, asym2: (B, sym_dim / asym_dim) for goal.

        Returns:
            (B, 1) negative distances.
        """
        dist_sym  = (sym1 - sym2).pow(2).mean(-1, keepdim=True)
        res       = F.relu(asym1 - asym2)
        dist_asym = (F.softmax(res, dim=-1) * res).sum(-1, keepdim=True)
        return -(dist_sym + dist_asym)

    def forward(
        self,
        obs1: torch.Tensor,
        act1: torch.Tensor,
        obs2: torch.Tensor,
        act2: torch.Tensor,
    ) -> torch.Tensor:
        """Compute negative quasi-metric d((obs1,act1), (obs2,act2)).

        For goal inputs use a zero action tensor so the goal side is
        action-invariant.  d(s, s) = 0 when obs1==obs2 and act1==act2.

        Returns:
            (B, 1) negative distances.
        """
        sym1, asym1 = self.encode(obs1, act1)
        sym2, asym2 = self.encode(obs2, act2)
        return self._dist_from_encodings(sym1, asym1, sym2, asym2)


class CostNet(nn.Module):
    """Maps a raw goal observation to a scalar baseline cost c(g).

    Used as the energy offset in InfoNCE: f(s,g) = c(g) − d(sa, g).
    Has its own internal obs encoder so it operates directly on raw obs.

    Args:
        obs_shape:   Shape of a single observation.
        obs_enc_dim: Internal encoding dimension.
        hidden_dims: Hidden layer sizes for the obs encoder.
    """

    def __init__(
        self,
        obs_shape: tuple[int, ...],
        obs_enc_dim: int = 256,
        hidden_dims: list[int] | None = None,
    ):
        super().__init__()
        self.obs_enc = _make_obs_encoder(obs_shape, obs_enc_dim, hidden_dims)
        self.head    = nn.Linear(obs_enc_dim, 1)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Returns (B, 1) scalar costs."""
        return self.head(self.obs_enc(obs))


class DiscreteActor(nn.Module):
    """Goal-conditioned actor for discrete action spaces.

    Call: actor.sample(state_obs, goal_obs) → (B,) long actions
    """

    def __init__(
        self,
        obs_shape: tuple[int, ...],
        num_actions: int,
        obs_enc_dim: int = 256,
        hidden_dims: list[int] | None = None,
    ):
        super().__init__()
        self.state_enc = _make_obs_encoder(obs_shape, obs_enc_dim, hidden_dims)
        self.goal_enc = _make_obs_encoder(obs_shape, obs_enc_dim, hidden_dims)

        if hidden_dims is None:
            hidden_dims = [256, 256]
        layers: list[nn.Module] = []
        prev = 2 * obs_enc_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers.append(nn.Linear(prev, num_actions))
        self.network = nn.Sequential(*layers)

    def forward(self, state_obs: torch.Tensor, goal_obs: torch.Tensor) -> torch.Tensor:
        """Returns logits (B, num_actions)."""
        return self.network(torch.cat([self.state_enc(state_obs), self.goal_enc(goal_obs)], dim=-1))

    def sample_with_log_prob(
        self, state_obs: torch.Tensor, goal_obs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Single forward pass returning (actions, log_probs).

        log_probs are differentiable w.r.t. network params (REINFORCE-compatible).
        """
        dist = torch.distributions.Categorical(logits=self.forward(state_obs, goal_obs))
        actions = dist.sample()
        return actions, dist.log_prob(actions)

    def sample(self, state_obs: torch.Tensor, goal_obs: torch.Tensor) -> torch.Tensor:
        return torch.distributions.Categorical(logits=self.forward(state_obs, goal_obs)).sample()

    def get_log_prob(
        self, state_obs: torch.Tensor, goal_obs: torch.Tensor, actions: torch.Tensor
    ) -> torch.Tensor:
        return torch.distributions.Categorical(
            logits=self.forward(state_obs, goal_obs)
        ).log_prob(actions)

    def entropy(self, state_obs: torch.Tensor, goal_obs: torch.Tensor) -> torch.Tensor:
        return torch.distributions.Categorical(logits=self.forward(state_obs, goal_obs)).entropy()


class ContinuousActor(nn.Module):
    """Goal-conditioned actor for continuous action spaces (squashed Gaussian).

    Call: actor.sample(state_obs, goal_obs) → (actions, log_probs)
    """

    def __init__(
        self,
        obs_shape: tuple[int, ...],
        action_dim: int,
        obs_enc_dim: int = 256,
        hidden_dims: list[int] | None = None,
        log_std_min: float = -20.0,
        log_std_max: float = 2.0,
    ):
        super().__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.state_enc = _make_obs_encoder(obs_shape, obs_enc_dim, hidden_dims)
        self.goal_enc = _make_obs_encoder(obs_shape, obs_enc_dim, hidden_dims)

        if hidden_dims is None:
            hidden_dims = [256, 256]
        layers: list[nn.Module] = []
        prev = 2 * obs_enc_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        self.trunk = nn.Sequential(*layers)
        self.mean_head = nn.Linear(prev, action_dim)
        self.log_std_head = nn.Linear(prev, action_dim)

    def forward(
        self, state_obs: torch.Tensor, goal_obs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (mean, log_std) before squashing."""
        features = self.trunk(
            torch.cat([self.state_enc(state_obs), self.goal_enc(goal_obs)], dim=-1)
        )
        mean = self.mean_head(features)
        log_std = torch.clamp(self.log_std_head(features), self.log_std_min, self.log_std_max)
        return mean, log_std

    def sample_with_log_prob(
        self, state_obs: torch.Tensor, goal_obs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Reparameterised sample + log_prob (both differentiable)."""
        mean, log_std = self.forward(state_obs, goal_obs)
        std = log_std.exp()
        x_t = torch.distributions.Normal(mean, std).rsample()
        action = torch.tanh(x_t)
        log_prob = torch.distributions.Normal(mean, std).log_prob(x_t)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        return action, log_prob.sum(dim=-1)

    def sample(
        self, state_obs: torch.Tensor, goal_obs: torch.Tensor, deterministic: bool = False
    ) -> torch.Tensor:
        mean, log_std = self.forward(state_obs, goal_obs)
        if deterministic:
            return torch.tanh(mean)
        std = log_std.exp()
        x_t = torch.distributions.Normal(mean, std).rsample()
        
        return torch.tanh(x_t)

    def get_log_prob(
        self, state_obs: torch.Tensor, goal_obs: torch.Tensor, actions: torch.Tensor
    ) -> torch.Tensor:
        """Log-prob of given (pre-squash) actions. Approximation via tanh inverse."""
        mean, log_std = self.forward(state_obs, goal_obs)
        std = log_std.exp()
        # Invert tanh: x_t = atanh(action)
        x_t = torch.atanh(actions.clamp(-1 + 1e-6, 1 - 1e-6))
        log_prob = torch.distributions.Normal(mean, std).log_prob(x_t)
        log_prob -= torch.log(1 - actions.pow(2) + 1e-6)
        return log_prob.sum(dim=-1)

    def entropy(self, state_obs: torch.Tensor, goal_obs: torch.Tensor) -> torch.Tensor:
        """Differential entropy of the pre-squash Gaussian (approximate)."""
        _, log_std = self.forward(state_obs, goal_obs)
        return torch.distributions.Normal(torch.zeros_like(log_std), log_std.exp()).entropy().sum(dim=-1)


class ResidualContinuousActor(nn.Module):
    """Goal-conditioned actor that additionally conditions on a base action.

    Architecture:
        branch_sg  : (obs, goal) → obs_enc via two obs encoders + trunk
        branch_base: base_action → obs_enc_dim // 2 via a small MLP
        combine    : concat(sg, base) → last_hidden → mean / log_std

    The base_action is the action the current graph would suggest; the actor
    learns a correction on top of it.  When base_actions=None, zeros are used
    (same behaviour as a standard ContinuousActor).

    Call: actor.sample(state_obs, goal_obs, base_actions=None) → actions
    """

    def __init__(
        self,
        obs_shape: tuple[int, ...],
        action_dim: int,
        obs_enc_dim: int = 128,
        hidden_dims: list[int] | None = None,
        log_std_min: float = -20.0,
        log_std_max: float = 2.0,
    ):
        super().__init__()
        self.action_dim  = action_dim
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.state_enc = _make_obs_encoder(obs_shape, obs_enc_dim, hidden_dims)
        self.goal_enc  = _make_obs_encoder(obs_shape, obs_enc_dim, hidden_dims)

        if hidden_dims is None:
            hidden_dims = [128, 128]

        base_enc_dim = obs_enc_dim // 2
        self.base_enc = nn.Sequential(
            nn.Linear(action_dim, base_enc_dim),
            nn.ReLU(),
        )

        layers: list[nn.Module] = []
        prev = 2 * obs_enc_dim + base_enc_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        self.trunk = nn.Sequential(*layers)
        self.mean_head    = nn.Linear(prev, action_dim)
        self.log_std_head = nn.Linear(prev, action_dim)

    def _features(
        self,
        state_obs: torch.Tensor,
        goal_obs: torch.Tensor,
        base_actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sg  = torch.cat([self.state_enc(state_obs), self.goal_enc(goal_obs)], dim=-1)
        ba  = self.base_enc(base_actions)
        out = self.trunk(torch.cat([sg, ba], dim=-1))
        mean    = self.mean_head(out)
        log_std = torch.clamp(self.log_std_head(out), self.log_std_min, self.log_std_max)
        return mean, log_std

    def _resolve_base(self, base_actions: torch.Tensor | None, B: int, device) -> torch.Tensor:
        if base_actions is None:
            return torch.zeros(B, self.action_dim, dtype=torch.float32, device=device)
        return base_actions

    @staticmethod
    def _clip(action: torch.Tensor) -> torch.Tensor:
        """Clip to the robosuite action range [-1, 1]."""
        return torch.clamp(action, -1.0, 1.0)

    def sample(
        self,
        state_obs: torch.Tensor,
        goal_obs: torch.Tensor,
        base_actions: torch.Tensor | None = None,
        deterministic: bool = False,
    ) -> torch.Tensor:
        base = self._resolve_base(base_actions, state_obs.shape[0], state_obs.device)
        mean, log_std = self._features(state_obs, goal_obs, base)
        if deterministic:
            residual = torch.tanh(mean)
        else:
            x_t = torch.distributions.Normal(mean, log_std.exp()).rsample()
            residual = torch.tanh(x_t)
        return self._clip(base + residual)

    def sample_with_log_prob(
        self,
        state_obs: torch.Tensor,
        goal_obs: torch.Tensor,
        base_actions: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        base = self._resolve_base(base_actions, state_obs.shape[0], state_obs.device)
        mean, log_std = self._features(state_obs, goal_obs, base)
        std = log_std.exp()
        x_t      = torch.distributions.Normal(mean, std).rsample()
        residual = torch.tanh(x_t)
        action   = self._clip(base + residual)
        log_prob = torch.distributions.Normal(mean, std).log_prob(x_t)
        log_prob -= torch.log(1 - residual.pow(2) + 1e-6)
        return action, log_prob.sum(dim=-1)

    def get_log_prob(
        self,
        state_obs: torch.Tensor,
        goal_obs: torch.Tensor,
        actions: torch.Tensor,
        base_actions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        base = self._resolve_base(base_actions, state_obs.shape[0], state_obs.device)
        mean, log_std = self._features(state_obs, goal_obs, base)
        std = log_std.exp()
        # Recover the residual from the total action. (Clipping at inference
        # can make this an approximation; it's exact whenever base + residual
        # was already inside [-1, 1].)
        residual = (actions - base).clamp(-1 + 1e-6, 1 - 1e-6)
        x_t = torch.atanh(residual)
        log_prob = torch.distributions.Normal(mean, std).log_prob(x_t)
        log_prob -= torch.log(1 - residual.pow(2) + 1e-6)
        return log_prob.sum(dim=-1)

    def entropy(
        self,
        state_obs: torch.Tensor,
        goal_obs: torch.Tensor,
        base_actions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        base = self._resolve_base(base_actions, state_obs.shape[0], state_obs.device)
        _, log_std = self._features(state_obs, goal_obs, base)
        return torch.distributions.Normal(torch.zeros_like(log_std), log_std.exp()).entropy().sum(dim=-1)
