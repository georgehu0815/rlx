import subprocess
import sys
import textwrap
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def run_fresh_python(script: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        cwd=REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )


def test_microduck_factory_reaches_vector_builder_before_thread_runtimes():
    result = run_fresh_python(
        """
        import sys
        from rlx.environments import make_microduck_env
        import rlx.environments.microduck as microduck

        forbidden = ("mlx", "torch", "tensorboard", "wandb")

        def loaded_roots():
            return sorted(
                root
                for root in forbidden
                if any(name == root or name.startswith(root + ".") for name in sys.modules)
            )

        assert loaded_roots() == []

        class FakeVecEnv:
            num_envs = 1
            observation_space = microduck.gym.spaces.Box(
                -1.0, 1.0, (microduck.MICRODUCK_OBSERVATION_DIM,)
            )
            action_space = microduck.gym.spaces.Box(
                -1.0, 1.0, (microduck.MICRODUCK_ACTION_DIM,)
            )

            def close(self):
                pass

        class VecEnvModule:
            @staticmethod
            def make_vec_env(env_fns, backend=None):
                assert backend == "fork"
                assert len(env_fns) == 1
                assert loaded_roots() == []
                return FakeVecEnv()

        class WalkEnvModule:
            class MicroduckWalkEnv:
                pass

        microduck._import_microduck_modules = lambda: (VecEnvModule, WalkEnvModule)
        env = make_microduck_env(num_envs=1, backend="fork")
        assert loaded_roots() == []
        env.close()
        """
    )
    assert result.returncode == 0, result.stderr


def test_example_calls_factory_before_thread_runtime_imports():
    result = run_fresh_python(
        """
        import importlib.util
        import sys
        from pathlib import Path

        forbidden = ("mlx", "torch", "tensorboard", "wandb")

        def loaded_roots():
            return sorted(
                root
                for root in forbidden
                if any(name == root or name.startswith(root + ".") for name in sys.modules)
            )

        path = Path("examples/ppo_microduck.py")
        spec = importlib.util.spec_from_file_location("ppo_microduck", path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        assert loaded_roots() == []

        class FactoryReached(Exception):
            pass

        def fake_factory(**kwargs):
            assert kwargs["backend"] == "fork"
            assert loaded_roots() == []
            raise FactoryReached

        module.make_microduck_env = fake_factory
        try:
            module.main(["--ppo.num-envs", "1"])
        except FactoryReached:
            pass
        else:
            raise AssertionError("example did not call the MicroDuck factory")
        """
    )
    assert result.returncode == 0, result.stderr


def test_lazy_packages_preserve_public_exports():
    result = run_fresh_python(
        """
        import sys
        import rlx
        import rlx.environments as environments

        assert rlx.__all__ == [
            "DQN", "DQNConfig", "REINFORCE", "REINFORCEConfig",
            "A2C", "A2CConfig", "PPO", "PPOConfig", "SAC", "SACConfig",
        ]
        assert "CartPole" in environments.__all__
        assert "make_microduck_env" in environments.__all__
        assert not any(name == "mlx" or name.startswith("mlx.") for name in sys.modules)

        from rlx.environments import CartPole, make_microduck_env
        assert CartPole.__name__ == "CartPole"
        assert callable(make_microduck_env)

        from rlx import PPO
        assert PPO.__name__ == "PPO"
        """
    )
    assert result.returncode == 0, result.stderr
