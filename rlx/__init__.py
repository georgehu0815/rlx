"""Top-level RLX exports, loaded on first access."""

from importlib import import_module
from typing import Any

_EXPORTS = {
    "DQN": ("rlx.algorithms", "DQN"),
    "DQNConfig": ("rlx.algorithms", "DQNConfig"),
    "REINFORCE": ("rlx.algorithms", "REINFORCE"),
    "REINFORCEConfig": ("rlx.algorithms", "REINFORCEConfig"),
    "A2C": ("rlx.algorithms", "A2C"),
    "A2CConfig": ("rlx.algorithms", "A2CConfig"),
    "PPO": ("rlx.algorithms", "PPO"),
    "PPOConfig": ("rlx.algorithms", "PPOConfig"),
    "SAC": ("rlx.algorithms", "SAC"),
    "SACConfig": ("rlx.algorithms", "SACConfig"),
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
