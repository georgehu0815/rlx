"""RLX environment exports, loaded on first access."""

from importlib import import_module
from typing import Any

_EXPORTS = {
    "CartPole": ("rlx.environments.classic_control", "CartPole"),
    "Pendulum": ("rlx.environments.classic_control", "Pendulum"),
    "Acrobot": ("rlx.environments.classic_control", "Acrobot"),
    "MountainCar": ("rlx.environments.classic_control", "MountainCar"),
    "Environment": ("rlx.environments.environment", "Environment"),
    "EnvState": ("rlx.environments.environment", "EnvState"),
    "EnvPool": ("rlx.environments.envpool", "EnvPool"),
    "MicroDuckVecEnv": ("rlx.environments.microduck", "MicroDuckVecEnv"),
    "make_microduck_env": ("rlx.environments.microduck", "make_microduck_env"),
    "RecordEpisodeStatistics": (
        "rlx.environments.wrappers",
        "RecordEpisodeStatistics",
    ),
    "Vectorize": ("rlx.environments.wrappers", "Vectorize"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    try:
        module_name, attribute_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(module_name), attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
