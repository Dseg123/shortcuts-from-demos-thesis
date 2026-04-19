"""Thin wrapper around a robomimic HDF5 dataset that mimics the Minari episode API.

Usage:
    # Low-dim (concatenates the specified flat keys)
    dataset = RobomimicDataset(
        '/path/to/low_dim_v15.hdf5',
        obs_keys=['robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos', 'object'],
    )

    # Image
    dataset = RobomimicDataset(
        '/path/to/image_v15.hdf5',
        obs_keys=['agentview_image'],
    )

Both return episode objects with .observations (T+1, ...), .actions (T, 7),
and .states (T, state_dim), matching the convention expected by LearningSystem.
The .states field holds raw MuJoCo simulator states (not observations) and can
be passed to RobomimicEnvironment.reset_to() to restore an exact simulator snapshot.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import gymnasium as gym
import h5py
import numpy as np


# Standard low-dim keys used in robomimic lift/ph benchmarks.
LOW_DIM_KEYS = ['robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos', 'object']


@dataclass
class RobomimicEpisode:
    observations: np.ndarray  # (T+1, ...) — obs[0..T-1] + final next_obs
    actions: np.ndarray       # (T, action_dim) float32
    states: np.ndarray        # (T, state_dim) — raw MuJoCo states for simulator reset


class RobomimicDataset:
    """Wraps a robomimic HDF5 file with a Minari-compatible episode API.

    Args:
        hdf5_path: Path to the .hdf5 file produced by dataset_states_to_obs.
        obs_keys:  Which obs group keys to use.
                   - For a single image key (e.g. ['agentview_image']), episodes
                     return (T+1, H, W, C) uint8 arrays.
                   - For multiple flat keys (e.g. LOW_DIM_KEYS), episodes return
                     (T+1, D) float32 arrays concatenated in key order.
    """

    def __init__(self, hdf5_path: str, obs_keys: list[str]):
        self._path = hdf5_path
        self._obs_keys = obs_keys

        with h5py.File(hdf5_path, 'r') as f:
            self._demo_keys = sorted(f['data'].keys())
            action_dim = f[f'data/{self._demo_keys[0]}/actions'].shape[1]
            # Preload all episodes into memory so training doesn't repeatedly
            # open/close the HDF5 file (which leaks memory via the HDF5 C library).
            self._episodes = [self._load_episode(f, k) for k in self._demo_keys]

        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(action_dim,), dtype=np.float32
        )
        total_steps = sum(len(e.actions) for e in self._episodes)
        print(f"Preloaded {len(self._episodes)} episodes ({total_steps} steps) into memory.")

    def _load_episode(self, f: h5py.File, demo_key: str) -> RobomimicEpisode:
        demo = f[f'data/{demo_key}']
        actions = demo['actions'][:].astype(np.float32)  # (T, action_dim)

        obs_arrays = [demo[f'obs/{k}'][:] for k in self._obs_keys]
        next_arrays = [demo[f'next_obs/{k}'][-1:] for k in self._obs_keys]

        is_image = len(obs_arrays) == 1 and obs_arrays[0].ndim == 4
        if is_image:
            # (T, H, W, C) uint8 — keep dtype for preprocessing detection
            obs_seq = np.concatenate([obs_arrays[0], next_arrays[0]], axis=0)
        else:
            # Concatenate flat keys → (T, D) float32
            obs_t = np.concatenate([a.astype(np.float32) for a in obs_arrays], axis=-1)
            final = np.concatenate([a.astype(np.float32) for a in next_arrays], axis=-1)
            obs_seq = np.concatenate([obs_t, final], axis=0)  # (T+1, D)

        states = demo['states'][:]  # (T, state_dim) — aligns with obs[0..T-1]

        return RobomimicEpisode(observations=obs_seq, actions=actions, states=states)

    def iterate_episodes(self):
        """Yield every episode in the dataset."""
        yield from self._episodes

    def sample_episodes(self, n_episodes: int) -> list[RobomimicEpisode]:
        """Sample n_episodes episodes uniformly at random (without replacement)."""
        return random.sample(self._episodes, min(n_episodes, len(self._episodes)))
