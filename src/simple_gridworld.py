"""Simple 2D continuous gridworld dataset and environment.

Observations are (x, y) positions in [0, 1]^2.
Actions are 2D movement vectors in [-1, 1]^2, applied as:
    pos += step_size * action  (then clipped to [0, 1]^2)

Demonstrations follow Manhattan paths: move horizontally to goal_x, then
vertically to goal_y.  This produces suboptimal-but-structured trajectories
whose L-shaped geometry should naturally partition the state space into a grid
of clusters when K-medoids is applied (e.g. K=9 → roughly 3x3).

Both SimpleDataset and SimpleEnvironment match the public interface of
RobomimicDataset and RobomimicEnvironment, so LearningSystem works without
any modification.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import gymnasium as gym
import numpy as np


# ---------------------------------------------------------------------------
# Episode dataclass
# ---------------------------------------------------------------------------

@dataclass
class SimpleEpisode:
    observations: np.ndarray  # (T+1, 2) float32 — positions at t = 0 .. T
    actions:      np.ndarray  # (T, 2)   float32 — actions  at t = 0 .. T-1
    states:       np.ndarray  # (T, 2)   float32 — same as observations[:-1]


# ---------------------------------------------------------------------------
# Shared config
# ---------------------------------------------------------------------------

@dataclass
class GridworldConfig:
    n_episodes:  int   = 500    # number of demo episodes to generate
    step_size:   float = 0.05   # fraction of [0,1] moved per unit action
    noise_std:   float = 0.01   # Gaussian noise added to actions during generation
    eps:         float = 0.05   # goal-reaching tolerance (L2 in obs space)
    render_size: int   = 128    # side length of render() output in pixels
    seed:        int | None = None


# ---------------------------------------------------------------------------
# Demo generation
# ---------------------------------------------------------------------------

def _generate_episode(
    start:     np.ndarray,
    goal:      np.ndarray,
    step_size: float,
    noise_std: float,
    rng:       np.random.Generator,
) -> SimpleEpisode:
    """Manhattan-path episode: move horizontally to goal_x, then vertically."""
    pos = start.astype(np.float32).copy()
    obs_list = [pos.copy()]
    act_list = []

    for axis in (0, 1):  # 0 = x (horizontal first), 1 = y (vertical second)
        while abs(pos[axis] - goal[axis]) > step_size * 0.5:
            action = np.zeros(2, dtype=np.float32)
            action[axis] = 0.95 * float(np.sign(goal[axis] - pos[axis]))
            if noise_std > 0:
                action += rng.normal(0, noise_std, 2).astype(np.float32)
                action = np.clip(action, -1.0, 1.0)
            act_list.append(action.copy())
            pos = np.clip(pos + step_size * action, 0.0, 1.0).astype(np.float32)
            obs_list.append(pos.copy())

    # Edge case: start == goal — add a single no-op step so T >= 1.
    if len(act_list) == 0:
        act_list.append(np.zeros(2, dtype=np.float32))
        obs_list.append(pos.copy())

    observations = np.stack(obs_list).astype(np.float32)  # (T+1, 2)
    actions      = np.stack(act_list).astype(np.float32)  # (T, 2)
    states       = observations[:-1].copy()               # (T, 2)
    return SimpleEpisode(observations=observations, actions=actions, states=states)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class SimpleDataset:
    """Procedurally-generated gridworld dataset with the RobomimicDataset interface.

    All episodes are generated at construction time and stored in memory.

    Args:
        config: GridworldConfig controlling generation parameters.
    """

    def __init__(self, config: GridworldConfig | None = None):
        if config is None:
            config = GridworldConfig()
        self.config = config

        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )

        rng    = np.random.default_rng(config.seed)
        starts = rng.uniform(0.0, 1.0, (config.n_episodes, 2)).astype(np.float32)
        goals  = rng.uniform(0.0, 1.0, (config.n_episodes, 2)).astype(np.float32)

        self._episodes: list[SimpleEpisode] = [
            _generate_episode(
                starts[i], goals[i], config.step_size, config.noise_std, rng
            )
            for i in range(config.n_episodes)
        ]

    def iterate_episodes(self):
        """Yield every episode."""
        yield from self._episodes

    def sample_episodes(self, n_episodes: int) -> list[SimpleEpisode]:
        """Sample n_episodes uniformly at random (without replacement)."""
        return random.sample(self._episodes, min(n_episodes, len(self._episodes)))


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class SimpleEnvironment:
    """Interactive gridworld environment with the RobomimicEnvironment interface.

    Args:
        dataset: A SimpleDataset (used to build the goal pool).
        config:  GridworldConfig — defaults to the dataset's own config.
    """

    def __init__(self, dataset: SimpleDataset, config: GridworldConfig | None = None):
        if config is None:
            config = dataset.config
        self.config      = config
        self._step_size  = config.step_size
        self._eps        = config.eps
        self._render_size = config.render_size

        # Goal pool: final observation of each demo episode.
        self._goal_obs: list[np.ndarray] = [
            ep.observations[-1].copy() for ep in dataset.iterate_episodes()
        ]

        self._pos:      np.ndarray = np.array([0.5, 0.5], dtype=np.float32)
        self._goal_pos: np.ndarray = np.array([0.5, 0.5], dtype=np.float32)

    # -----------------------------------------------------------------------
    # State / observation conversion  (state == obs == position here)
    # -----------------------------------------------------------------------

    def state_to_obs(self, state: np.ndarray) -> np.ndarray:
        return state.astype(np.float32).copy()

    def obs_to_state(self, obs: np.ndarray) -> np.ndarray:
        return obs.astype(np.float32).copy()

    # -----------------------------------------------------------------------
    # LearningSystem interface
    # -----------------------------------------------------------------------

    def reset(self) -> tuple[np.ndarray, np.ndarray]:
        """Random start in [0,1]^2; goal sampled from the dataset goal pool."""
        self._pos = np.random.uniform(0.0, 1.0, 2).astype(np.float32)
        goal_obs = random.choice(self._goal_obs).copy()
        self._goal_pos = goal_obs.copy()
        return self._pos.copy(), goal_obs

    def reset_to(
        self, start_state: np.ndarray, goal_state: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        self._pos      = start_state.astype(np.float32).copy()
        self._goal_pos = goal_state.astype(np.float32).copy()
        return self._pos.copy(), self._goal_pos.copy()

    def step(self, action) -> tuple[np.ndarray, float, bool, bool, dict]:
        action    = np.asarray(action, dtype=np.float32)
        print(f"Received action: {action}")
        self._pos = np.clip(self._pos + self._step_size * action, 0.0, 1.0)
        done      = bool(np.linalg.norm(self._pos - self._goal_pos) < self._eps)
        return self._pos.copy(), 0.0, done, False, {}

    def render(self) -> np.ndarray:
        """RGB image: white background, red dot for goal, blue dot for agent."""
        size = self._render_size
        img  = np.full((size, size, 3), 255, dtype=np.uint8)

        def draw_dot(pos: np.ndarray, color: list[int], radius: int = 5) -> None:
            # (0,0) is bottom-left; flip y so it renders naturally.
            cx = int(np.clip(pos[0] * (size - 1), 0, size - 1))
            cy = int(np.clip((1.0 - pos[1]) * (size - 1), 0, size - 1))
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    if dx * dx + dy * dy <= radius * radius:
                        px = int(np.clip(cx + dx, 0, size - 1))
                        py = int(np.clip(cy + dy, 0, size - 1))
                        img[py, px] = color

        draw_dot(self._goal_pos, color=[220, 50,  50])   # red   — goal
        draw_dot(self._pos,      color=[50,  100, 220])  # blue  — agent
        return img

    def is_at_goal(self, curr_obs: np.ndarray, goal_obs: np.ndarray) -> bool:
        return bool(np.linalg.norm(curr_obs - goal_obs) < self._eps)
