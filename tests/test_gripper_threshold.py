"""Gripper replay threshold: configurable, default-preserving, and
swept honestly.

The policy thresholds the recorded ``gripper_pos`` joint trajectory at
``gripper_threshold`` (default the joint-range midpoint 0.8515) into
the binary gripper action. The evaluator's ground-truth
recorded-closed labels always use the midpoint, so a policy threshold
different from it measures the agreement cost of a misplaced
transition point (M16 sweep, 2026-11-21).
"""

import math
import unittest

import numpy as np

from ecm_policy.core import (
    GRIPPER_CLOSE_CMD,
    GRIPPER_OPEN_CMD,
    GRIPPER_THRESHOLD,
    ReplayFollowCore,
    _gripper_commands_for,
)
from ecm_policy.evaluate import evaluate_lagged_follower


def _two_joint_traj(steps=20, gripper="split"):
    """A 2-arm trajectory whose EE x positions move monotonically
    (so the phase tracker can actually re-anchor), with a chosen
    gripper channel: 'split' (open then closed), 'open' (all open),
    or 'closed' (all closed)."""
    poses = np.zeros((steps, 2, 7))
    for i in range(steps):
        poses[i, 0, 0] = 0.01 * i
        poses[i, 1, 0] = 0.01 * i + 0.05
    poses[:, :, 3] = 1.0  # identity quaternions (w=1)
    if gripper == "open":
        grip = np.zeros((steps, 2))
    elif gripper == "closed":
        grip = np.full((steps, 2), 1.703)
    elif gripper == "split":
        grip = np.zeros((steps, 2))
        grip[steps // 2 :, :] = 1.703
    else:
        raise ValueError(f"unknown gripper mode: {gripper!r}")
    return poses, grip


class TestGripperThresholdCore(unittest.TestCase):
    def test_default_threshold_is_midpoint(self):
        self.assertAlmostEqual(GRIPPER_THRESHOLD, 0.8515)
        poses, grip = _two_joint_traj()
        core = ReplayFollowCore(poses, gripper_pos=grip)
        self.assertAlmostEqual(core.gripper_threshold, 0.8515)

    def test_threshold_helper_uses_given_threshold(self):
        row = np.array([0.5, 1.0])
        self.assertTrue(
            np.array_equal(
                _gripper_commands_for(row, 0.2),
                [GRIPPER_CLOSE_CMD, GRIPPER_CLOSE_CMD],
            )
        )
        self.assertTrue(
            np.array_equal(
                _gripper_commands_for(row, 0.8515),
                [GRIPPER_OPEN_CMD, GRIPPER_CLOSE_CMD],
            )
        )
        self.assertTrue(
            np.array_equal(
                _gripper_commands_for(row, 1.5),
                [GRIPPER_OPEN_CMD, GRIPPER_OPEN_CMD],
            )
        )

    def test_custom_threshold_changes_emitted_commands(self):
        poses, grip = _two_joint_traj()
        # At the closed half, gripper_pos is 1.703: closed at any
        # threshold below it, open at any threshold above it.
        low = ReplayFollowCore(poses, gripper_pos=grip, gripper_threshold=0.2)
        high = ReplayFollowCore(poses, gripper_pos=grip, gripper_threshold=1.8)
        low.reset()
        high.reset()
        # Drive both cores to the closed half (lookahead 1, no obs,
        # so the phase advances open-loop one demo step per predict).
        for _ in range(15):
            low_action = low.predict()
            high_action = high.predict()
        self.assertEqual(low_action[14], GRIPPER_CLOSE_CMD)
        self.assertEqual(low_action[15], GRIPPER_CLOSE_CMD)
        self.assertEqual(high_action[14], GRIPPER_OPEN_CMD)
        self.assertEqual(high_action[15], GRIPPER_OPEN_CMD)

    def test_nonfinite_threshold_rejected(self):
        poses, grip = _two_joint_traj()
        for bad in (math.nan, math.inf, -math.inf):
            with self.assertRaises(ValueError):
                ReplayFollowCore(poses, gripper_pos=grip, gripper_threshold=bad)

    def test_threshold_boundary_is_closed(self):
        # >= threshold means closed: a row exactly at the threshold
        # must emit the close command.
        row = np.array([GRIPPER_THRESHOLD, GRIPPER_THRESHOLD])
        self.assertTrue(
            np.array_equal(
                _gripper_commands_for(row),
                [GRIPPER_CLOSE_CMD, GRIPPER_CLOSE_CMD],
            )
        )


class TestGripperThresholdEvaluator(unittest.TestCase):
    def test_midpoint_threshold_reproduces_baseline(self):
        # All-open channel: at the default threshold the policy and
        # the ground-truth labels agree on every step.
        poses, grip = _two_joint_traj(gripper="open")
        result = evaluate_lagged_follower(
            poses, demo="threshold-baseline", alpha=1.0, gripper_pos=grip
        )
        self.assertAlmostEqual(result.gripper_agreement, 1.0, places=6)
        self.assertAlmostEqual(result.gripper_threshold, GRIPPER_THRESHOLD)

    def test_off_midpoint_threshold_loses_agreement(self):
        # All-closed channel, policy threshold 1.8 (above every
        # recorded value): the policy never closes while the
        # midpoint-labelled ground truth is closed on every step.
        poses, grip = _two_joint_traj(gripper="closed")
        result = evaluate_lagged_follower(
            poses,
            demo="threshold-off",
            alpha=1.0,
            gripper_pos=grip,
            gripper_threshold=1.8,
        )
        self.assertAlmostEqual(result.gripper_agreement, 0.0, places=6)
        self.assertAlmostEqual(result.gripper_threshold, 1.8)

    def test_threshold_does_not_change_position_error(self):
        # The gripper action does not feed the pose tracker, so the
        # threshold must not move position error at all.
        poses, grip = _two_joint_traj(steps=60)
        base = evaluate_lagged_follower(
            poses, demo="threshold-poserr", alpha=0.3, gripper_pos=grip
        )
        moved = evaluate_lagged_follower(
            poses,
            demo="threshold-poserr",
            alpha=0.3,
            gripper_pos=grip,
            gripper_threshold=0.2,
        )
        self.assertAlmostEqual(
            base.mean_position_error_m, moved.mean_position_error_m, places=9
        )

    def test_evaluator_determinism_with_threshold(self):
        poses, grip = _two_joint_traj(steps=60)
        kwargs = dict(
            demo="threshold-determinism", alpha=0.5, gripper_pos=grip,
            gripper_threshold=0.425, noise_sigma_m=0.005, noise_seed=20261114,
        )
        first = evaluate_lagged_follower(poses, **kwargs)
        second = evaluate_lagged_follower(poses, **kwargs)
        self.assertEqual(first.as_dict(), second.as_dict())


if __name__ == "__main__":
    unittest.main()
