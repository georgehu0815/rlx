"""Train PPO on MicroDuck without importing thread runtimes before fork."""

import argparse
import random
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Sequence

from rlx.environments.microduck import make_microduck_env


@dataclass
class PPOArgs:
    num_envs: int = 16
    num_steps: int = 24
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_minibatches: int = 4
    update_epochs: int = 5
    clip_coefficient: float = 0.2
    entropy_coefficient: float = 0.01


@dataclass
class Args:
    experiment_name: str = "ppo_microduck"
    seed: int = 1
    total_timesteps: int = 1_000_000
    evaluate_steps: int = 10_000
    learning_rate: float = 1e-3
    backend: str = "fork"
    actuator: str = "xml"
    domain_rand: bool = True
    obs_noise: bool = True
    action_delay: bool = True
    random_yaw: bool = True
    max_episode_s: float = 20.0
    normalize_rewards: bool = False
    track: bool = False
    wandb_project_name: str = "rlx"
    wandb_entity: str = ""
    ppo: PPOArgs = field(default_factory=PPOArgs)


def parse_args(argv: Sequence[str] | None = None) -> Args:
    defaults = Args()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-name", default=defaults.experiment_name)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--total-timesteps", type=int, default=defaults.total_timesteps)
    parser.add_argument("--evaluate-steps", type=int, default=defaults.evaluate_steps)
    parser.add_argument("--learning-rate", type=float, default=defaults.learning_rate)
    parser.add_argument("--backend", default=defaults.backend)
    parser.add_argument("--actuator", default=defaults.actuator)
    parser.add_argument(
        "--domain-rand", action=argparse.BooleanOptionalAction, default=defaults.domain_rand
    )
    parser.add_argument(
        "--obs-noise", action=argparse.BooleanOptionalAction, default=defaults.obs_noise
    )
    parser.add_argument(
        "--action-delay", action=argparse.BooleanOptionalAction, default=defaults.action_delay
    )
    parser.add_argument(
        "--random-yaw", action=argparse.BooleanOptionalAction, default=defaults.random_yaw
    )
    parser.add_argument("--max-episode-s", type=float, default=defaults.max_episode_s)
    parser.add_argument(
        "--normalize-rewards",
        action=argparse.BooleanOptionalAction,
        default=defaults.normalize_rewards,
    )
    parser.add_argument(
        "--track", action=argparse.BooleanOptionalAction, default=defaults.track
    )
    parser.add_argument("--wandb-project-name", default=defaults.wandb_project_name)
    parser.add_argument("--wandb-entity", default=defaults.wandb_entity)
    for name, value in vars(defaults.ppo).items():
        parser.add_argument(
            f"--ppo.{name.replace('_', '-')}",
            dest=f"ppo_{name}",
            type=type(value),
            default=value,
        )
    parsed = vars(parser.parse_args(argv))
    ppo = PPOArgs(
        **{
            name: parsed.pop(f"ppo_{name}")
            for name in vars(defaults.ppo)
        }
    )
    return Args(**parsed, ppo=ppo)


def make_actor_critic(env: Any, mx: Any, nn: Any, gaussian: Any) -> Any:
    class MicroDuckActorCritic(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            observation_dim = env.observation_space.shape[0]
            action_dim = env.action_space.shape[0]
            self.actor_mean = nn.Sequential(
                nn.Linear(observation_dim, 512),
                nn.ELU(),
                nn.Linear(512, 256),
                nn.ELU(),
                nn.Linear(256, 128),
                nn.ELU(),
                nn.Linear(128, action_dim),
            )
            self.actor_log_std = mx.zeros((action_dim,))
            self.critic = nn.Sequential(
                nn.Linear(observation_dim, 512),
                nn.ELU(),
                nn.Linear(512, 256),
                nn.ELU(),
                nn.Linear(256, 128),
                nn.ELU(),
                nn.Linear(128, 1),
            )

        def __call__(self, observation: Any) -> Any:
            mean = self.actor_mean(observation)
            log_std = mx.broadcast_to(self.actor_log_std, mean.shape)
            return gaussian(mean, log_std), self.critic(observation)

    return MicroDuckActorCritic()


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    random.seed(args.seed)

    env = None
    writer = None
    wandb_run = None
    try:
        # This must happen before imports that initialize MLX, torch, TensorBoard,
        # or wandb: the default backend forks after compiling one shared model.
        env = make_microduck_env(
            num_envs=args.ppo.num_envs,
            backend=args.backend,
            seed=args.seed,
            actuator=args.actuator,
            domain_rand=args.domain_rand,
            obs_noise=args.obs_noise,
            action_delay=args.action_delay,
            random_yaw=args.random_yaw,
            max_episode_s=args.max_episode_s,
            normalize_observations=True,
            normalize_rewards=args.normalize_rewards,
            gamma=args.ppo.gamma,
        )

        import gymnasium as gym
        import mlx.core as mx
        import mlx.nn as nn
        import mlx.optimizers as optim
        import numpy as np
        from torch.utils.tensorboard import SummaryWriter

        from rlx.algorithms.ppo import PPO, PPOConfig
        from rlx.buffers.rollout_buffer import RolloutBuffer
        from rlx.utils.distributions import Gaussian
        from rlx.utils.logger import Logger

        assert isinstance(env.action_space, gym.spaces.Box), (
            "only continuous action space is supported"
        )
        np.random.seed(args.seed)
        mx.random.seed(args.seed)
        config = PPOConfig(**vars(args.ppo))
        run_name = (
            f"MicroDuck__{args.experiment_name}__{args.seed}__{int(time.time())}"
        )
        if args.track:
            import wandb

            wandb_run = wandb.init(
                project=args.wandb_project_name,
                entity=args.wandb_entity,
                sync_tensorboard=True,
                config=asdict(args),
                name=run_name,
                save_code=True,
            )
        writer = SummaryWriter(f"runs/{run_name}")
        writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s"
            % "\n".join(f"|{key}|{value}|" for key, value in vars(config).items()),
        )

        network = make_actor_critic(env, mx, nn, Gaussian)
        mx.eval(network.parameters())
        optimizer = optim.Adam(learning_rate=args.learning_rate)
        buffer = RolloutBuffer(
            config.num_steps,
            env.observation_space,
            env.action_space,
            gamma=config.gamma,
            num_envs=config.num_envs,
        )
        algorithm = PPO(
            config=config,
            env=env,
            network=network,
            optimizer=optimizer,
            buffer=buffer,
            key=mx.random.key(args.seed),
        )
        logger = Logger()
        algorithm.train(args.total_timesteps, callback=logger)
        if args.evaluate_steps > 0:
            algorithm.evaluate(args.evaluate_steps, callback=logger)
    finally:
        if env is not None:
            env.close()
        if writer is not None:
            writer.close()
        if wandb_run is not None:
            wandb_run.finish()


if __name__ == "__main__":
    main()
