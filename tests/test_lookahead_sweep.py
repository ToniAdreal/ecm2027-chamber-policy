"""Tests for the lookahead sweep (milestone 8).

The sweep measures whether the policy's ``lookahead`` — how many demo
steps ahead of the tracked phase the commanded pose sits — helps or
hurts tracking under the lagged-follower harness. Only the shortest
bundled demo (demo_b2_3, 4022 steps) is used here so the suite stays
fast; the full 6-demo sweep lives in ``tools/sweep_lookahead.py``.

What is asserted — and what is deliberately NOT asserted:

* lookahead 0 + alpha 1.0 STALLS the closed-loop tracker: the policy
  commands the demo pose at the tracked phase, the robot is already
  there, and the nearest-point tracker never advances (final phase 0,
  mean phase lag ~2010 steps on demo_b2_3). This is a real property of
  the nearest-point tracker, not a harness artifact — lookahead must
  be >= 1 for closed-loop tracking. The sweep is expected to show
  lookahead 0 as the worst setting.
* lookahead >= 2 + alpha 1.0 OUTRUNS the recording: the robot jumps
  several demo steps per policy step, so scored at the same step index
  it sits ahead of the recording (negative mean phase lag). Exact
  reproduction holds only at lookahead 1, which the M7 suite already
  pins.
* lookahead 1 + alpha 0.3 on demo_b2_3 reproduces the exact M7 number
  (mean position error 0.6984621786486186 m): the sweep reuses the same
  code path, so the M7 table must come back bit-for-bit.
* the sweep is deterministic: the same (demo, alpha, lookahead) run
  twice gives identical numbers.
* ``evaluate_all_demos`` accepts a lookahead and tags every result
  with it.
* NO trend in lookahead is asserted: whether a larger lookahead helps
  a lagging follower is an empirical question the sweep answers, not a
  property the tests may assume. The tool reports the measured best
  lookahead per alpha; the tests only pin the machinery.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ecm_policy.evaluate import (
    evaluate_all_demos,
    evaluate_lagged_follower,
)
from ecm_policy.trajectory import load_ee_poses_npz, load_gripper_pos_npz
from tools.sweep_lookahead import best_lookahead_by_alpha

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# Exact M7 measurement (evaluation/results.json, lookahead 1, alpha 0.3).
M7_B2_3_ALPHA03_MEAN_POS_ERR = 0.6984621786486186


def _load_b2_3():
    traj = load_ee_poses_npz(DATA_DIR / "demo_b2_3_ee_poses.npz")
    grip = load_gripper_pos_npz(DATA_DIR / "demo_b2_3_ee_poses.npz")
    assert grip is not None
    return traj, grip


def test_lookahead_zero_stalls_the_phase_tracker() -> None:
    """lookahead 0 stalls: the tracker commands the pose the robot is
    already on, so the nearest-point phase never advances."""
    traj, grip = _load_b2_3()
    result = evaluate_lagged_follower(
        traj, demo="demo_b2_3", alpha=1.0, lookahead=0, gripper_pos=grip
    )
    assert result.lookahead == 0
    assert result.final_phase == 0
    # Phase lag grows one step per policy step: mean (0 + 4021) / 2.
    assert result.mean_phase_lag == pytest.approx(2010.5, abs=1.0)
    assert result.max_phase_lag == 4021
    assert result.mean_position_error_m == pytest.approx(
        0.986636, abs=1e-6
    )


def test_lookahead_two_outruns_the_recording() -> None:
    """lookahead >= 2 at alpha 1.0: the robot jumps ahead of the
    recording, so the mean phase lag goes negative."""
    traj, grip = _load_b2_3()
    result = evaluate_lagged_follower(
        traj, demo="demo_b2_3", alpha=1.0, lookahead=2, gripper_pos=grip
    )
    assert result.lookahead == 2
    assert result.mean_phase_lag == pytest.approx(-1005.0, abs=1.0)
    assert result.mean_position_error_m == pytest.approx(
        0.680951, abs=1e-6
    )


def test_lookahead_one_reproduces_m7_number() -> None:
    traj, grip = _load_b2_3()
    result = evaluate_lagged_follower(
        traj, demo="demo_b2_3", alpha=0.3, lookahead=1, gripper_pos=grip
    )
    assert result.lookahead == 1
    assert result.mean_position_error_m == pytest.approx(
        M7_B2_3_ALPHA03_MEAN_POS_ERR, abs=1e-9
    )
    assert result.mean_phase_lag == pytest.approx(1419.2896568871208, abs=1e-6)
    assert result.gripper_agreement == pytest.approx(
        0.5129288910989558, abs=1e-9
    )


def test_sweep_is_deterministic() -> None:
    traj, grip = _load_b2_3()
    first = evaluate_lagged_follower(
        traj, demo="demo_b2_3", alpha=0.3, lookahead=2, gripper_pos=grip
    )
    second = evaluate_lagged_follower(
        traj, demo="demo_b2_3", alpha=0.3, lookahead=2, gripper_pos=grip
    )
    assert first.as_dict() == second.as_dict()


def test_evaluate_all_demos_tags_lookahead() -> None:
    results = evaluate_all_demos(DATA_DIR, alphas=[1.0], lookahead=4)
    assert len(results) == 6  # one row per bundled demo
    assert all(r.lookahead == 4 for r in results)
    assert all(r.alpha == 1.0 for r in results)
    assert {r.demo for r in results} == {
        "demo_0",
        "demo_2",
        "demo_7",
        "demo_9",
        "demo_b2_0",
        "demo_b2_3",
    }
    # At lookahead 4 the robot outruns the recording on every demo, so
    # the mean phase lag is negative everywhere (exact reproduction
    # holds only at lookahead 1, pinned by the M7 suite).
    assert all(r.mean_phase_lag < 0 for r in results)


def test_best_lookahead_picks_minimum() -> None:
    traj, grip = _load_b2_3()
    rows = [
        evaluate_lagged_follower(
            traj, demo="demo_b2_3", alpha=0.3, lookahead=la, gripper_pos=grip
        )
        for la in (0, 1, 2)
    ]
    best = best_lookahead_by_alpha(rows)
    assert set(best) == {0.3}
    expected = min(rows, key=lambda r: r.mean_position_error_m).lookahead
    assert best[0.3] == expected
