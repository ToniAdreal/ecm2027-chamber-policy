"""Adapter tests: real starter BasePolicy + ChamberReplayPolicy + mock env.

The starter repo's ``policy/base.py`` imports torch and mediapy, which do
not exist on this CPU-only host. This test injects minimal stand-ins for
those two modules purely so the *real* starter BasePolicy file and the
adapter's own rollout logic can execute; the mock environment mimics the
RemoteEnv step/reset contract from the starter eval entrypoint
(``reset() -> (obs, extras)``, ``step(action) -> (obs, reward, terminated,
truncated, extras)``) with perfect EE tracking. Nothing here exercises the
simulator itself; it verifies client-side policy logic only.
"""

import importlib.util
import sys
import types
import unittest
from pathlib import Path

import numpy as np

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
STARTER_ROOT = PACKAGE_ROOT.parent / "ecm2027-sim-track"
sys.path.insert(0, str(PACKAGE_ROOT))


class _StubTensor:
    def __init__(self, arr):
        self._arr = np.asarray(arr)

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self._arr


def _install_module_stubs():
    torch_stub = types.ModuleType("torch")
    torch_stub.Tensor = _StubTensor
    torch_stub.float32 = np.float32
    torch_stub.is_tensor = lambda value: isinstance(value, _StubTensor)
    torch_stub.as_tensor = lambda arr, dtype=None: _StubTensor(
        np.asarray(arr, dtype=dtype or np.float32)
    )
    sys.modules.setdefault("torch", torch_stub)
    sys.modules.setdefault("mediapy", types.ModuleType("mediapy"))


def _load_real_base_policy():
    _install_module_stubs()
    policy_pkg = types.ModuleType("policy")
    policy_pkg.__path__ = []
    sys.modules["policy"] = policy_pkg
    spec = importlib.util.spec_from_file_location(
        "policy.base", STARTER_ROOT / "policy" / "base.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["policy.base"] = module
    spec.loader.exec_module(module)
    policy_pkg.base = module
    return module


_load_real_base_policy()

from ecm_policy.adapter import ChamberReplayPolicy  # noqa: E402
from ecm_policy.trajectory import load_ee_poses_npz  # noqa: E402

DEMO = load_ee_poses_npz(PACKAGE_ROOT / "data" / "demo_7_ee_poses.npz")


class MockEnv:
    """RemoteEnv-shaped stand-in with perfect end-effector tracking."""

    def __init__(self, terminate_at: int, with_success_extras: bool):
        self.terminate_at = terminate_at
        self.with_success_extras = with_success_extras
        self.ee = DEMO[0].copy()
        self.t = 0
        self.actions_seen = []

    def _obs(self):
        return {
            "policy": {
                "eef_pos": self.ee[None, :, 0:3].copy(),
                "eef_quat": self.ee[None, :, 3:7].copy(),
            },
            "embodiment_general_obs": {},
        }

    def reset(self):
        self.t = 0
        self.ee = DEMO[0].copy()
        return self._obs(), {}

    def step(self, action):
        arr = action.numpy() if hasattr(action, "numpy") else np.asarray(action)
        self.actions_seen.append(arr)
        assert arr.shape == (1, 20), f"unexpected action shape {arr.shape}"
        assert np.all(np.isfinite(arr)), "non-finite action"
        self.ee = arr[0, 0:14].reshape(2, 7).copy()
        self.t += 1
        terminated = np.array([self.t >= self.terminate_at])
        extras = {}
        if terminated[0] and self.with_success_extras:
            extras = {"success": np.array([True])}
        return self._obs(), 0.0, terminated, np.array([False]), extras


def make_policy(**overrides):
    usr_args = {"instruction": "Service the chamber", "env_cfg": {"num_envs": 1}}
    usr_args.update(overrides)
    return ChamberReplayPolicy(usr_args), usr_args


@unittest.skipUnless((STARTER_ROOT / "policy" / "base.py").exists(), "starter repo missing")
class AdapterRolloutTests(unittest.TestCase):
    def test_success_from_extras(self):
        policy, usr_args = make_policy()
        env = MockEnv(terminate_at=30, with_success_extras=True)
        obs, _ = env.reset()
        policy.reset_model()
        result = policy.eval(env, obs, usr_args, None)
        self.assertEqual(result.tolist(), [True])
        self.assertEqual(len(env.actions_seen), 30)
        summary = policy.format_eval_results([result], usr_args, 1)
        self.assertEqual(summary["test_count"], 1)
        self.assertEqual(summary["success_count"], 1)

    def test_success_falls_back_to_terminated(self):
        policy, usr_args = make_policy()
        env = MockEnv(terminate_at=5, with_success_extras=False)
        obs, _ = env.reset()
        policy.reset_model()
        result = policy.eval(env, obs, usr_args, None)
        self.assertEqual(result.tolist(), [True])
        self.assertEqual(len(env.actions_seen), 5)

    def test_step_budget_caps_unfinished_rollout(self):
        policy, usr_args = make_policy(policy_max_steps=10)
        env = MockEnv(terminate_at=10 ** 9, with_success_extras=False)
        obs, _ = env.reset()
        policy.reset_model()
        result = policy.eval(env, obs, usr_args, None)
        self.assertEqual(result.tolist(), [False])
        self.assertEqual(len(env.actions_seen), 10)

    def test_first_command_matches_demo_lookahead(self):
        policy, usr_args = make_policy()
        env = MockEnv(terminate_at=1, with_success_extras=True)
        obs, _ = env.reset()
        policy.reset_model()
        policy.eval(env, obs, usr_args, None)
        first = env.actions_seen[0][0]
        np.testing.assert_allclose(first[0:7], DEMO[1, 0], atol=1e-6)
        np.testing.assert_allclose(first[7:14], DEMO[1, 1], atol=1e-6)


if __name__ == "__main__":
    unittest.main()
