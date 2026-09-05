import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def test_microduck_shared_model_smoke():
    if os.environ.get("RUN_MICRODUCK_INTEGRATION") != "1":
        pytest.skip("set RUN_MICRODUCK_INTEGRATION=1 to run the MuJoCo smoke test")

    script = textwrap.dedent(
        """
        import json
        import sys

        from rlx.environments.microduck import make_microduck_env

        assert not any(name == "mlx" or name.startswith("mlx.") for name in sys.modules)
        with make_microduck_env(
            num_envs=1,
            backend="fork",
            seed=7,
            domain_rand=False,
            obs_noise=False,
            action_delay=False,
            random_yaw=False,
        ) as env:
            assert not any(
                name == "mlx" or name.startswith("mlx.") for name in sys.modules
            )
            import mlx.core as mx
            import numpy as np

            observation, state, _ = env.reset(mx.random.key(7))
            result = env.step(None, state, mx.zeros((1, 14)))
            payload = {
                "observation_shape": list(observation.shape),
                "step_shape": list(result[0].shape),
                "finite": bool(np.isfinite(np.asarray(result[0])).all()),
            }
        print(json.dumps(payload))
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    if result.returncode != 0 and "No module named 'microduck_local'" in result.stderr:
        pytest.skip("microduck_local is not installed in the test environment")
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout.strip().splitlines()[-1])
    assert payload == {
        "observation_shape": [1, 61],
        "step_shape": [1, 61],
        "finite": True,
    }
