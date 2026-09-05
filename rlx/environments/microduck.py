"""RLX adapter for microduck_local's shared-model vector environments."""

from __future__ import annotations

import importlib
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import gymnasium as gym
import numpy as np

if TYPE_CHECKING:
    import mlx.core as mx

MICRODUCK_OBSERVATION_DIM = 61
MICRODUCK_ACTION_DIM = 14


class _RunningMeanStd:
    def __init__(self, shape: tuple[int, ...] | tuple[()]):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = 1e-4

    def update(self, values: np.ndarray) -> None:
        values = np.asarray(values, dtype=np.float64)
        batch_mean = np.mean(values, axis=0)
        batch_var = np.var(values, axis=0)
        batch_count = values.shape[0]
        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        self.mean += delta * batch_count / total_count
        first_moment = self.var * self.count
        second_moment = batch_var * batch_count
        correction = np.square(delta) * self.count * batch_count / total_count
        self.var = (first_moment + second_moment + correction) / total_count
        self.count = total_count


class MicroDuckVecEnv:
    """Expose an old-style, autoresetting MicroDuck vec env through RLX's API.

    ``microduck_local.vec_env.make_vec_env`` returns observations, rewards,
    one ``done`` vector, and one info dict per environment. This adapter keeps
    that vec env's autoreset behavior and reconstructs Gymnasium's terminated
    and truncated vectors from ``TimeLimit.truncated``. RLX RNG keys and state
    are accepted for API compatibility; randomness remains owned by the seeded
    MicroDuck workers.
    """

    def __init__(
        self,
        envs: Any,
        *,
        normalize_observations: bool = False,
        normalize_rewards: bool = False,
        gamma: float = 0.99,
        epsilon: float = 1e-8,
        clip: float = 10.0,
    ) -> None:
        self.envs = envs
        self.num_envs = int(envs.num_envs)
        self._observation_space = envs.observation_space
        self._action_space = envs.action_space
        self._validate_spaces()

        self.normalize_observations = normalize_observations
        self.normalize_rewards = normalize_rewards
        self.gamma = gamma
        self.epsilon = epsilon
        self.clip = clip
        self.discounted_returns = np.zeros(self.num_envs, dtype=np.float64)
        self.episode_returns = np.zeros(self.num_envs, dtype=np.float64)
        self.episode_lengths = np.zeros(self.num_envs, dtype=np.int64)
        self.observation_rms = _RunningMeanStd((MICRODUCK_OBSERVATION_DIM,))
        self.return_rms = _RunningMeanStd(())
        self.closed = False

    def _validate_spaces(self) -> None:
        if not isinstance(self._observation_space, gym.spaces.Box):
            raise TypeError("MicroDuck observations must use a gymnasium Box space")
        if self._observation_space.shape != (MICRODUCK_OBSERVATION_DIM,):
            raise ValueError(
                "MicroDuck observation space must have shape "
                f"({MICRODUCK_OBSERVATION_DIM},), got "
                f"{self._observation_space.shape}"
            )
        if not isinstance(self._action_space, gym.spaces.Box):
            raise TypeError("MicroDuck actions must use a gymnasium Box space")
        if self._action_space.shape != (MICRODUCK_ACTION_DIM,):
            raise ValueError(
                "MicroDuck action space must have shape "
                f"({MICRODUCK_ACTION_DIM},), got {self._action_space.shape}"
            )

    def _validate_observation(self, observation: Any) -> np.ndarray:
        array = np.asarray(observation, dtype=np.float32)
        expected = (self.num_envs, MICRODUCK_OBSERVATION_DIM)
        if array.shape != expected:
            raise ValueError(
                f"MicroDuck observation batch must have shape {expected}, got {array.shape}"
            )
        return array

    def _normalize_observation(self, observation: np.ndarray) -> np.ndarray:
        if not self.normalize_observations:
            return observation
        self.observation_rms.update(observation)
        normalized = (observation - self.observation_rms.mean) / np.sqrt(
            self.observation_rms.var + self.epsilon
        )
        return np.clip(normalized, -self.clip, self.clip).astype(np.float32)

    @property
    def observation_space(self) -> gym.Space:
        return self._observation_space

    @property
    def action_space(self) -> gym.Space:
        return self._action_space

    def reset(self, key: mx.array | None = None) -> tuple[mx.array, dict, dict]:
        del key
        result = self.envs.reset()
        reset_info: Any = {}
        if isinstance(result, tuple) and len(result) == 2:
            observation, reset_info = result
        else:
            observation = result
        observation = self._validate_observation(observation)
        self.discounted_returns.fill(0.0)
        self.episode_returns.fill(0.0)
        self.episode_lengths.fill(0)
        info = reset_info if isinstance(reset_info, dict) else {"infos": reset_info}
        mx = _import_mlx()
        return mx.array(self._normalize_observation(observation)), {}, info

    def step(
        self,
        key: mx.array | None,
        state: dict,
        action: mx.array,
    ) -> tuple[mx.array, dict, mx.array, mx.array, mx.array, dict]:
        del key, state
        actions = np.asarray(action, dtype=np.float32)
        expected_actions = (self.num_envs, MICRODUCK_ACTION_DIM)
        if actions.shape != expected_actions:
            raise ValueError(
                f"MicroDuck action batch must have shape {expected_actions}, got "
                f"{actions.shape}"
            )

        observation, reward, done, infos = self.envs.step(actions)
        observation = self._validate_observation(observation)
        reward = np.asarray(reward, dtype=np.float64)
        done = np.asarray(done, dtype=bool)
        if reward.shape != (self.num_envs,) or done.shape != (self.num_envs,):
            raise ValueError(
                "MicroDuck rewards and done flags must have shape "
                f"({self.num_envs},), got {reward.shape} and {done.shape}"
            )
        if not isinstance(infos, Sequence) or len(infos) != self.num_envs:
            raise ValueError(
                f"MicroDuck infos must contain {self.num_envs} dictionaries"
            )
        if not all(isinstance(item, dict) for item in infos):
            raise TypeError("each MicroDuck info entry must be a dictionary")

        truncated = np.array(
            [
                is_done and bool(item.get("TimeLimit.truncated", False))
                for is_done, item in zip(done, infos)
            ],
            dtype=bool,
        )
        terminated = np.logical_and(done, np.logical_not(truncated))

        self.episode_returns += reward
        self.episode_lengths += 1
        reported_returns = self.episode_returns.copy()
        reported_lengths = self.episode_lengths.copy()
        for index, (is_done, item) in enumerate(zip(done, infos)):
            episode = item.get("episode")
            if is_done and isinstance(episode, dict):
                reported_returns[index] = episode.get("r", reported_returns[index])
                reported_lengths[index] = episode.get("l", reported_lengths[index])

        info = {
            "episode": {"r": reported_returns, "l": reported_lengths},
            "_episode": done.copy(),
            "infos": list(infos),
        }
        self.episode_returns[done] = 0.0
        self.episode_lengths[done] = 0

        normalized_reward = reward
        if self.normalize_rewards:
            self.discounted_returns = self.discounted_returns * self.gamma + reward
            self.return_rms.update(self.discounted_returns)
            normalized_reward = reward / np.sqrt(self.return_rms.var + self.epsilon)
            normalized_reward = np.clip(normalized_reward, -self.clip, self.clip)
            self.discounted_returns[done] = 0.0

        observation = self._normalize_observation(observation)
        mx = _import_mlx()
        return (
            mx.array(observation),
            {},
            mx.array(normalized_reward.astype(np.float32)),
            mx.array(terminated),
            mx.array(truncated),
            info,
        )

    def close(self) -> None:
        if not self.closed:
            self.envs.close()
            self.closed = True

    def __enter__(self) -> MicroDuckVecEnv:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


def _import_mlx() -> Any:
    """Load MLX only after the MicroDuck worker processes have been created."""
    return importlib.import_module("mlx.core")


def _import_microduck_modules() -> tuple[Any, Any]:
    try:
        vec_env = importlib.import_module("microduck_local.vec_env")
        walk_env = importlib.import_module("microduck_local.walk_env")
    except (ImportError, ModuleNotFoundError) as exc:
        missing = getattr(exc, "name", None) or "a required dependency"
        raise ImportError(
            "MicroDuck support requires microduck_local and its dependencies. "
            "Install the sibling microduck_local checkout in this environment "
            "(for example, `uv pip install -e ../microduck_local`) and ensure "
            "the microduck_rl assets are available. "
            f"Import failed for {missing!r}."
        ) from exc
    return vec_env, walk_env


def make_microduck_env(
    num_envs: int = 16,
    *,
    backend: str | None = None,
    seed: int = 0,
    actuator: str = "xml",
    domain_rand: bool = True,
    obs_noise: bool = True,
    action_delay: bool = True,
    random_yaw: bool = True,
    max_episode_s: float = 20.0,
    normalize_observations: bool = False,
    normalize_rewards: bool = False,
    gamma: float = 0.99,
    epsilon: float = 1e-8,
    clip: float = 10.0,
) -> MicroDuckVecEnv:
    """Create a shared-model MicroDuck vector environment for RLX.

    Imports are deliberately local to this function. Normal RLX imports do not
    require microduck_local, and no torch-using SB3 module is imported before
    the default fork backend creates its workers.
    """
    if num_envs < 1:
        raise ValueError(f"num_envs must be positive, got {num_envs}")

    vec_env_module, walk_env_module = _import_microduck_modules()
    env_class = walk_env_module.MicroduckWalkEnv

    def make_env(rank: int):
        def initialize():
            return env_class(
                seed=seed + rank,
                actuator_force=actuator,
                domain_rand=domain_rand,
                obs_noise=obs_noise,
                action_delay=action_delay,
                random_yaw=random_yaw,
                max_episode_s=max_episode_s,
            )

        return initialize

    envs = vec_env_module.make_vec_env(
        [make_env(rank) for rank in range(num_envs)], backend=backend
    )
    try:
        return MicroDuckVecEnv(
            envs,
            normalize_observations=normalize_observations,
            normalize_rewards=normalize_rewards,
            gamma=gamma,
            epsilon=epsilon,
            clip=clip,
        )
    except Exception:
        envs.close()
        raise
