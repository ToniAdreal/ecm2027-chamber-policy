"""Unit tests for gripper replay (milestone 6).

The recorded Batch 2 ``gripper_pos`` is the Galbot gripper joint
position: 0.0 fully open, 1.703 fully closed (starter repo Galbot
config ``BinaryJointPositionActionCfg`` open/close_command_expr).
The policy thresholds it at the joint-range midpoint 0.8515 into the
binary gripper action (``< 0`` close, ``>= 0`` open, per the starter
repo's ``BinaryJointPositionAction`` convention). Demos without a
gripper channel keep the fixed configurable command.
"""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ecm_policy.core import (  # noqa: E402
    GRIPPER_CLOSE_CMD,
    GRIPPER_OPEN_CMD,
    GRIPPER_THRESHOLD,
    ReplayFollowCore,
)
from ecm_policy.multidemo import MultiDemoCore  # noqa: E402
from ecm_policy.trajectory import load_gripper_pos_npz  # noqa: E402

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PACKAGE_ROOT / "data"
B2_NPZ = DATA_DIR / "demo_b2_0_ee_poses.npz"
B2B_NPZ = DATA_DIR / "demo_b2_3_ee_poses.npz"
B1_NPZ = DATA_DIR / "demo_0_ee_poses.npz"


def synthetic_trajectory(steps: int = 4) -> np.ndarray:
    traj = np.zeros((steps, 2, 7), dtype=np.float64)
    for t in range(steps):
        traj[t, 0] = [0.1 * t, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0]
        traj[t, 1] = [0.1 * t, 0.5, 1.0, 1.0, 0.0, 0.0, 0.0]
    return traj


class ThresholdMappingTests(unittest.TestCase):
    def test_open_closed_and_mixed_rows(self):
        gripper = np.array(
            [
                [0.0, 0.0],        # both open
                [0.0, 1.703],      # left open, right closed
                [1.703, 0.0],      # left closed, right open
                [1.703, 1.703],    # both closed
            ]
        )
        core = ReplayFollowCore(
            synthetic_trajectory(4), actions_dim=16, lookahead=0,
            gripper_pos=gripper,
        )
        expected = [
            [GRIPPER_OPEN_CMD, GRIPPER_OPEN_CMD],
            [GRIPPER_OPEN_CMD, GRIPPER_CLOSE_CMD],
            [GRIPPER_CLOSE_CMD, GRIPPER_OPEN_CMD],
            [GRIPPER_CLOSE_CMD, GRIPPER_CLOSE_CMD],
        ]
        for want in expected:
            action = core.predict()
            self.assertEqual([action[14], action[15]], want)

    def test_threshold_boundary(self):
        # At the midpoint the demo gripper counts as closed; just below,
        # open. (Threshold is the 0.0-1.703 joint-range midpoint.)
        self.assertAlmostEqual(GRIPPER_THRESHOLD, 0.8515)
        gripper = np.array([[GRIPPER_THRESHOLD, GRIPPER_THRESHOLD - 1e-9]])
        core = ReplayFollowCore(
            synthetic_trajectory(1), actions_dim=16, lookahead=0,
            gripper_pos=gripper,
        )
        action = core.predict()
        self.assertEqual(action[14], GRIPPER_CLOSE_CMD)
        self.assertEqual(action[15], GRIPPER_OPEN_CMD)

    def test_lookahead_reads_target_index(self):
        gripper = np.array([[0.0, 0.0], [0.0, 0.0], [1.703, 1.703]])
        core = ReplayFollowCore(
            synthetic_trajectory(3), actions_dim=16, lookahead=2,
            gripper_pos=gripper,
        )
        action = core.predict()  # step 0, target index 2 -> closed
        self.assertEqual([action[14], action[15]],
                         [GRIPPER_CLOSE_CMD, GRIPPER_CLOSE_CMD])


class FixedFallbackTests(unittest.TestCase):
    def test_no_gripper_channel_keeps_fixed_commands(self):
        core = ReplayFollowCore(synthetic_trajectory(2), actions_dim=16)
        self.assertEqual(core.gripper_mode, "fixed")
        for _ in range(2):
            action = core.predict()
            self.assertEqual([action[14], action[15]], [1.0, 1.0])

    def test_custom_fixed_commands_respected_without_channel(self):
        core = ReplayFollowCore(
            synthetic_trajectory(2), actions_dim=16,
            gripper_left=-1.0, gripper_right=0.5,
        )
        action = core.predict()
        self.assertEqual([action[14], action[15]], [-1.0, 0.5])

    def test_validation_rejects_bad_gripper(self):
        traj = synthetic_trajectory(4)
        with self.assertRaises(ValueError):
            ReplayFollowCore(traj, gripper_pos=np.zeros((3, 2)))
        with self.assertRaises(ValueError):
            ReplayFollowCore(traj, gripper_pos=np.zeros((4, 3)))
        bad = np.zeros((4, 2))
        bad[1, 0] = np.nan
        with self.assertRaises(ValueError):
            ReplayFollowCore(traj, gripper_pos=bad)


class RealBatch2Tests(unittest.TestCase):
    def test_from_npz_loads_gripper_and_replays_it(self):
        recorded = load_gripper_pos_npz(B2_NPZ)
        self.assertIsNotNone(recorded)
        core = ReplayFollowCore.from_npz(B2_NPZ, actions_dim=16, lookahead=0)
        self.assertEqual(core.gripper_mode, "replay")
        self.assertEqual(core.num_steps, recorded.shape[0])
        emitted = []
        for _ in range(core.num_steps):
            action = core.predict()
            emitted.append([action[14], action[15]])
        emitted = np.array(emitted)
        want = np.where(
            recorded >= GRIPPER_THRESHOLD,
            GRIPPER_CLOSE_CMD, GRIPPER_OPEN_CMD,
        )
        np.testing.assert_array_equal(emitted, want)
        # The demo really does grasp: 8 open/close transitions, both arms
        # in lockstep, 58.1% of steps closed.
        transitions = int(np.abs(np.diff(want[:, 0])).sum() / 2)
        self.assertEqual(transitions, 8)
        self.assertAlmostEqual(float((want[:, 0] < 0).mean()), 0.581, places=2)

    def test_second_batch2_episode_replays_too(self):
        core = ReplayFollowCore.from_npz(B2B_NPZ, actions_dim=16, lookahead=0)
        self.assertEqual(core.gripper_mode, "replay")
        recorded = load_gripper_pos_npz(B2B_NPZ)
        want = np.where(
            recorded >= GRIPPER_THRESHOLD,
            GRIPPER_CLOSE_CMD, GRIPPER_OPEN_CMD,
        )
        transitions = int(np.abs(np.diff(want[:, 0])).sum() / 2)
        self.assertEqual(transitions, 6)
        first = core.predict()
        self.assertEqual([first[14], first[15]],
                         [GRIPPER_OPEN_CMD, GRIPPER_OPEN_CMD])

    def test_batch1_bundle_stays_fixed(self):
        self.assertIsNone(load_gripper_pos_npz(B1_NPZ))
        core = ReplayFollowCore.from_npz(B1_NPZ, actions_dim=16, lookahead=0)
        self.assertEqual(core.gripper_mode, "fixed")
        for _ in range(5):
            action = core.predict()
            self.assertEqual([action[14], action[15]], [1.0, 1.0])


class MultiDemoGripperTests(unittest.TestCase):
    def test_mixed_directory_modes(self):
        multi = MultiDemoCore.from_npz_dir(DATA_DIR, actions_dim=16, lookahead=0)
        self.assertEqual(len(multi.demo_names), 6)
        for name in ("demo_0", "demo_2", "demo_7", "demo_9"):
            self.assertEqual(multi._cores[name].gripper_mode, "fixed")
        for name in ("demo_b2_0", "demo_b2_3"):
            self.assertEqual(multi._cores[name].gripper_mode, "replay")

    def test_selected_batch2_core_replays_full_episode(self):
        multi = MultiDemoCore.from_npz_dir(DATA_DIR, actions_dim=16, lookahead=0)
        from ecm_policy.trajectory import load_ee_poses_npz

        first_poses = load_ee_poses_npz(B2_NPZ)
        multi.reset(observation={"eef_pos": first_poses[0, :, 0:3]})
        self.assertEqual(multi.selected, "demo_b2_0")
        recorded = load_gripper_pos_npz(B2_NPZ)
        want = np.where(
            recorded >= GRIPPER_THRESHOLD,
            GRIPPER_CLOSE_CMD, GRIPPER_OPEN_CMD,
        )
        emitted = np.array(
            [multi.predict()[14:16] for _ in range(multi.num_steps)]
        )
        np.testing.assert_array_equal(emitted, want)


if __name__ == "__main__":
    unittest.main()
