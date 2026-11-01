"""Unit tests for ReplayFollowCore (pure CPU, no simulator)."""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ecm_policy.core import ReplayFollowCore  # noqa: E402
from ecm_policy.trajectory import load_ee_poses_npz  # noqa: E402

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
DEMO_NPZ = PACKAGE_ROOT / "data" / "demo_7_ee_poses.npz"


def synthetic_trajectory(steps: int = 10) -> np.ndarray:
    traj = np.zeros((steps, 2, 7), dtype=np.float64)
    for t in range(steps):
        traj[t, 0] = [0.1 * t, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0]
        traj[t, 1] = [0.1 * t, 0.5, 1.0, 1.0, 0.0, 0.0, 0.0]
    return traj


class CoreWithRealDemoTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.demo = load_ee_poses_npz(DEMO_NPZ)

    def test_action_layout_and_static_channels(self):
        core = ReplayFollowCore(self.demo)
        core.reset()
        action = core.predict()
        self.assertEqual(action.shape, (20,))
        self.assertEqual(action.dtype, np.float32)
        self.assertTrue(np.all(np.isfinite(action)))
        # Grippers hold the configured (open) command; wheels stay at zero.
        self.assertEqual(float(action[14]), 1.0)
        self.assertEqual(float(action[15]), 1.0)
        self.assertTrue(np.all(action[16:20] == 0.0))

    def test_first_action_targets_demo_at_lookahead(self):
        core = ReplayFollowCore(self.demo, lookahead=1)
        core.reset()
        action = core.predict()
        np.testing.assert_allclose(action[0:7], self.demo[1, 0], atol=1e-6)
        np.testing.assert_allclose(action[7:14], self.demo[1, 1], atol=1e-6)

    def test_phase_tracking_anchors_to_observed_pose(self):
        core = ReplayFollowCore(self.demo)
        core.reset()
        query = self.demo[100, :, 0:3]
        _action, info = core.predict_with_info({"eef_pos": query})
        self.assertTrue(info["phase_tracked"])
        self.assertLessEqual(core.phase, 100)
        anchored = core.phase
        np.testing.assert_allclose(
            self.demo[anchored, :, 0:3], query, atol=1e-6
        )
        # A query from earlier in the demo must not move the phase backwards.
        core.predict_with_info({"eef_pos": self.demo[50, :, 0:3]})
        self.assertGreaterEqual(core.phase, anchored)

    def test_health_reports_trajectory(self):
        core = ReplayFollowCore(self.demo)
        health = core.health()
        self.assertEqual(health["trajectory_steps"], 7566)
        self.assertEqual(health["actions_dim"], 20)
        self.assertTrue(health["ready"])


class CoreWithSyntheticTrajectoryTests(unittest.TestCase):
    def test_open_loop_advances_then_holds_last_pose(self):
        traj = synthetic_trajectory(10)
        core = ReplayFollowCore(traj, lookahead=0)
        core.reset()
        for _ in range(15):
            action = core.predict()
        self.assertEqual(core.phase, 9)
        np.testing.assert_allclose(action[0:3], traj[9, 0, 0:3], atol=1e-6)

    def test_actions_dim_16_drops_wheels(self):
        core = ReplayFollowCore(synthetic_trajectory(3), actions_dim=16)
        core.reset()
        self.assertEqual(core.predict().shape, (16,))

    def test_quat_order_xyzw_reorders_on_output(self):
        traj = synthetic_trajectory(1)
        wxyz = np.array([0.5, 0.5, 0.5, 0.5])  # already unit norm
        traj[0, 0, 3:7] = wxyz
        core = ReplayFollowCore(traj, lookahead=0, action_quat_order="xyzw")
        core.reset()
        action = core.predict()
        np.testing.assert_allclose(action[3:7], wxyz[[1, 2, 3, 0]], atol=1e-6)

    def test_invalid_trajectory_rejected(self):
        with self.assertRaises(ValueError):
            ReplayFollowCore(np.zeros((5, 3, 7)))
        bad = synthetic_trajectory(2)
        bad[0, 0, 0] = np.nan
        with self.assertRaises(ValueError):
            ReplayFollowCore(bad)

    def test_close_blocks_predict_until_reset(self):
        core = ReplayFollowCore(synthetic_trajectory(3))
        core.reset()
        core.close()
        with self.assertRaises(RuntimeError):
            core.predict()
        core.reset()
        self.assertEqual(core.predict().shape, (20,))


if __name__ == "__main__":
    unittest.main()
