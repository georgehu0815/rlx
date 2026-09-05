import importlib

import gymnasium as gym
import numpy as np
import pytest

from rlx.environments.microduck import MicroDuckVecEnv, make_microduck_env


class FakeVecEnv:
    def __init__(self, observation_dim=61, action_dim=14, num_envs=3):
        self.num_envs = num_envs
        self.observation_space = gym.spaces.Box(
            -np.inf, np.inf, (observation_dim,), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            -4.0, 4.0, (action_dim,), dtype=np.float32
        )
        self.closed = False
        self.actions = None
        self._step = 0

    def reset(self):
        return np.arange(self.num_envs * 61, dtype=np.float64).reshape(
            self.num_envs, 61
        )

    def step(self, actions):
        self.actions = actions
        self._step += 1
        observation = np.full((self.num_envs, 61), self._step, dtype=np.float64)
        reward = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        done = np.array([True, True, False])
        infos = [
            {"episode": {"r": 11.0, "l": 7}, "terminal_observation": "first"},
            {"TimeLimit.truncated": True, "terminal_observation": "second"},
            {},
        ]
        return observation, reward, done, infos

    def close(self):
        self.closed = True


def test_reset_ignores_rng_key_and_returns_float32_mlx_batch():
    env = MicroDuckVecEnv(FakeVecEnv())
    mx = importlib.import_module("mlx.core")
    observation, state, info = env.reset(mx.random.key(123))

    assert observation.shape == (3, 61)
    assert observation.dtype == mx.float32
    assert state == {}
    assert info == {}


def test_step_converts_actions_and_splits_termination_from_truncation():
    inner = FakeVecEnv()
    env = MicroDuckVecEnv(inner)
    mx = importlib.import_module("mlx.core")
    env.reset(None)
    observation, state, reward, terminated, truncated, info = env.step(
        mx.random.key(1), {"ignored": mx.array(1)}, mx.zeros((3, 14))
    )

    assert observation.shape == (3, 61)
    assert observation.dtype == mx.float32
    assert reward.dtype == mx.float32
    assert state == {}
    np.testing.assert_array_equal(np.asarray(terminated), [True, False, False])
    np.testing.assert_array_equal(np.asarray(truncated), [False, True, False])
    assert inner.actions.dtype == np.float32
    assert inner.actions.shape == (3, 14)
    assert info["infos"][1]["terminal_observation"] == "second"


def test_autoreset_episode_statistics_match_logger_shape_and_passthrough():
    env = MicroDuckVecEnv(FakeVecEnv())
    mx = importlib.import_module("mlx.core")
    env.reset(None)
    *_, info = env.step(None, {}, mx.zeros((3, 14)))

    np.testing.assert_array_equal(info["_episode"], [True, True, False])
    np.testing.assert_allclose(info["episode"]["r"], [11.0, 2.0, 3.0])
    np.testing.assert_array_equal(info["episode"]["l"], [7, 1, 1])

    *_, second_info = env.step(None, {}, mx.zeros((3, 14)))
    np.testing.assert_allclose(second_info["episode"]["r"], [11.0, 2.0, 6.0])
    np.testing.assert_array_equal(second_info["episode"]["l"], [7, 1, 2])


def test_observation_and_reward_normalization_are_finite_and_clipped():
    env = MicroDuckVecEnv(
        FakeVecEnv(),
        normalize_observations=True,
        normalize_rewards=True,
        clip=1.0,
    )
    mx = importlib.import_module("mlx.core")
    observation, _, _ = env.reset(None)
    _, _, reward, *_ = env.step(None, {}, mx.zeros((3, 14)))

    assert np.isfinite(np.asarray(observation)).all()
    assert np.isfinite(np.asarray(reward)).all()
    assert np.max(np.abs(np.asarray(observation))) <= 1.0
    assert np.max(np.abs(np.asarray(reward))) <= 1.0
    assert env.observation_rms.count > env.num_envs
    assert env.return_rms.count > env.num_envs


def test_close_is_idempotent_and_context_manager_closes():
    inner = FakeVecEnv()
    with MicroDuckVecEnv(inner) as env:
        assert env is not None
    assert inner.closed
    env.close()


@pytest.mark.parametrize(
    ("observation_dim", "action_dim", "message"),
    [(60, 14, "observation space"), (61, 13, "action space")],
)
def test_dimension_validation(observation_dim, action_dim, message):
    with pytest.raises(ValueError, match=message):
        MicroDuckVecEnv(FakeVecEnv(observation_dim, action_dim))


def test_missing_microduck_dependency_has_actionable_error(monkeypatch):
    real_import_module = importlib.import_module

    def missing(name, package=None):
        if name.startswith("microduck_local"):
            error = ModuleNotFoundError("No module named 'microduck_local'")
            error.name = "microduck_local"
            raise error
        return real_import_module(name, package)

    monkeypatch.setattr(importlib, "import_module", missing)
    with pytest.raises(ImportError, match="uv pip install -e ../microduck_local"):
        make_microduck_env(num_envs=1)
