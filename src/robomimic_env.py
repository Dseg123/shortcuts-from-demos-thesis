"""Robomimic environment wrapper with the same interface as env_wrapper.Environment.

The environment is reconstructed from metadata stored in the HDF5 file, so it
exactly matches the settings used to collect the demos.

At construction time the wrapper preloads two pools:
  - _goal_pairs   : (state, obs) at the final timestep of each demo.
  - _lookup_pairs : (state, obs) at every timestep, randomly capped at
                    max_lookup_pairs for memory efficiency with image datasets.

reset() calls the environment's own reset(), then samples a random goal.
reset_to(start_state, goal_state) gives explicit control over both.
state_to_obs(state) converts a state vector to its observation without
  disturbing the current simulation.
obs_to_state(obs) finds the closest dataset state by observation L2 distance.
is_at_goal checks L2 distance between live and goal MuJoCo state vectors.

Usage:
    from robomimic_env import RobomimicEnvironment
    from robomimic_dataset import LOW_DIM_KEYS

    env = RobomimicEnvironment(hdf5_path, obs_keys=LOW_DIM_KEYS)
    env = RobomimicEnvironment(hdf5_path, obs_keys=['agentview_image'], eps=0.5)
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass

import h5py
import numpy as np

import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.obs_utils as ObsUtils


@dataclass
class _StatePair:
    state: np.ndarray  # (state_dim,) MuJoCo state vector
    obs:   np.ndarray  # extracted observation, same format as RobomimicDataset


class RobomimicEnvironment:
    """Wraps a robomimic/robosuite environment with a LearningSystem-compatible API.

    Args:
        hdf5_path:        Path to a robomimic HDF5 dataset (low_dim or image).
        obs_keys:         Same obs_keys as the paired RobomimicDataset.
        eps:              L2 tolerance on the MuJoCo state vector for is_at_goal.
                          The state is 32-dimensional for Lift, so eps ~0.1–1.0
                          is a reasonable starting range.
        max_lookup_pairs: Maximum number of (state, obs) pairs to preload for
                          obs_to_state search.  If the dataset has more, a random
                          sample of this size is kept.  Mainly a concern for image
                          datasets where each obs is 84×84×3 uint8.
        camera_name:      Camera used for render() calls.
        render_height:    Height of render() frames (pixels).
        render_width:     Width of render() frames (pixels).
    """

    def __init__(
        self,
        hdf5_path: str,
        obs_keys: list[str],
        eps: float = 0.5,
        max_lookup_pairs: int = 5000,
        camera_name: str = 'agentview',
        render_height: int = 256,
        render_width: int = 256,
    ):
        self._obs_keys = obs_keys
        self._eps = eps
        self._camera_name = camera_name
        self._render_height = render_height
        self._render_width = render_width
        self._is_image = len(obs_keys) == 1 and obs_keys[0].endswith('_image')
        self._goal_state: np.ndarray | None = None  # set by reset() / reset_to()

        with h5py.File(hdf5_path, 'r') as f:
            env_meta = json.loads(f['data'].attrs['env_args'])
            self._start_pairs, self._goal_pairs, self._lookup_pairs = self._preload(f, max_lookup_pairs)

        ObsUtils.initialize_obs_utils_with_obs_specs({
            'obs': {
                'low_dim': [] if self._is_image else obs_keys,
                'rgb':     obs_keys if self._is_image else [],
            }
        })

        self._env = EnvUtils.create_env_from_metadata(
            env_meta=env_meta,
            render=False,
            render_offscreen=True,
            use_image_obs=self._is_image,
        )

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    def _make_obs(self, demo, t: int, use_next: bool = False) -> np.ndarray:
        """Extract and format a single observation from an open demo group."""
        prefix = 'next_obs' if use_next else 'obs'
        arrays = [demo[f'{prefix}/{k}'][t] for k in self._obs_keys]
        if self._is_image:
            return arrays[0].copy()
        return np.concatenate([a.astype(np.float32) for a in arrays])

    def _preload(
        self, f: h5py.File, max_lookup_pairs: int
    ) -> tuple[list[_StatePair], list[_StatePair]]:
        """Build goal pool (final timestep per demo) and lookup pool (all timesteps).

        The lookup pool is randomly downsampled to max_lookup_pairs if needed.
        """
        start_pairs:  list[_StatePair] = []
        goal_pairs:   list[_StatePair] = []
        all_pairs:    list[_StatePair] = []

        for key in sorted(f['data'].keys()):
            demo = f[f'data/{key}']
            states = demo['states'][:]  # (T, state_dim)
            T = len(states)

            for t in range(T):
                all_pairs.append(_StatePair(
                    state=states[t].copy(),
                    obs=self._make_obs(demo, t),
                ))

            start_pairs.append(_StatePair(
                state=states[0].copy(),
                obs=self._make_obs(demo, 0),
            ))

            goal_pairs.append(_StatePair(
                state=states[-1].copy(),
                obs=self._make_obs(demo, -1, use_next=True),
            ))

        if len(all_pairs) > max_lookup_pairs:
            lookup_pairs = random.sample(all_pairs, max_lookup_pairs)
        else:
            lookup_pairs = all_pairs

        return start_pairs, goal_pairs, lookup_pairs

    def _extract_obs(self, obs_dict: dict) -> np.ndarray:
        if self._is_image:
            img = np.asarray(obs_dict[self._obs_keys[0]])
            if img.ndim == 3 and img.shape[0] in (1, 3, 4):
                img = np.transpose(img, (1, 2, 0))  # CHW → HWC
            return img.astype(np.uint8)
        return np.concatenate(
            [np.asarray(obs_dict[k], dtype=np.float32) for k in self._obs_keys]
        )

    # -----------------------------------------------------------------------
    # State / observation conversion utilities
    # -----------------------------------------------------------------------

    def state_to_obs(self, state: np.ndarray) -> np.ndarray:
        """Return the observation corresponding to a MuJoCo state vector.

        Saves and restores the current simulation state so this can be called
        safely during a rollout without disrupting the episode.
        """
        saved_state = self._env.get_state()['states']
        obs = self._extract_obs(self._env.reset_to({'states': state}))
        self._env.reset_to({'states': saved_state})
        return obs

    def obs_to_state(self, obs: np.ndarray) -> np.ndarray:
        """Find the dataset state whose observation is closest to obs (by L2).

        Searches _lookup_pairs, which contains up to max_lookup_pairs (state, obs)
        pairs drawn uniformly from the full dataset.

        For image obs, both query and candidates are normalised to [0, 1] before
        computing distance so the scale is consistent.
        """
        if self._is_image:
            query = obs.astype(np.float32) / 255.0
            candidates = np.stack([p.obs for p in self._lookup_pairs]).astype(np.float32) / 255.0
        else:
            query = obs.astype(np.float32)
            candidates = np.stack([p.obs for p in self._lookup_pairs]).astype(np.float32)

        # (N, D) - (D,) broadcasts correctly; norm over flattened obs dim
        dists = np.linalg.norm(
            candidates.reshape(len(candidates), -1) - query.flatten(), axis=1
        )
        return self._lookup_pairs[int(np.argmin(dists))].state.copy()

    # -----------------------------------------------------------------------
    # LearningSystem interface
    # -----------------------------------------------------------------------

    def reset(self) -> tuple[np.ndarray, np.ndarray]:
        """Reset using the environment's own reset; sample a random goal from the dataset.

        Returns:
            (start_obs, goal_obs)
        """
        start_obs = self._extract_obs(self._env.reset())
        goal = random.choice(self._goal_pairs)
        self._goal_state = goal.state
        return start_obs, goal.obs.copy()

    def reset_to(
        self, start_state: np.ndarray, goal_state: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Reset to specific start and goal MuJoCo state vectors.

        Briefly resets to goal_state to extract goal_obs, then resets to
        start_state so the simulation is ready to roll out.

        Returns:
            (start_obs, goal_obs)
        """
        # print("Goal state:", goal_state)
        goal_obs  = self._extract_obs(self._env.reset_to({'states': goal_state}))
        start_obs = self._extract_obs(self._env.reset_to({'states': start_state}))
        self._goal_state = goal_state
        return start_obs, goal_obs

    def step(self, action) -> tuple[np.ndarray, float, bool, bool, dict]:
        obs_dict, reward, done, info = self._env.step(action)
        return self._extract_obs(obs_dict), reward, done, False, info

    def get_state(self) -> np.ndarray:
        """Return the exact live MuJoCo state vector (copy)."""
        return self._env.get_state()['states'].copy()

    def render(self) -> np.ndarray:
        return self._env.render(
            mode='rgb_array',
            height=self._render_height,
            width=self._render_width,
            camera_name=self._camera_name,
        )

    def is_at_goal(self, curr_obs: np.ndarray, goal_obs: np.ndarray) -> bool:
        """Return True when the live MuJoCo state is within eps of the goal state."""
        if self._goal_state is None:
            return bool(self._env.is_success()['task'])
        curr_state = self._env.get_state()['states']
        return bool(np.linalg.norm(curr_state - self._goal_state) < self._eps)

    # def close(self) -> None:
    #     self._env.close()
