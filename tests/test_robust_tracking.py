"""Tests for noise-robust phase tracking (milestone 10).

The M9 noise sweep showed the phase tracker jumps forward up to the full
250-step window in a single step when the observation is noisy: it takes
the nearest demo pose in the window and never moves backward. Milestone
10 adds ``phase_advance_cap`` to ``ReplayFollowCore``: the phase may move
at most N demo steps per ``predict`` call.

What is asserted — and what is deliberately NOT asserted:

* cap bounds the advance: with an observation equal to a demo pose 150
  steps ahead, an uncapped core jumps straight to it, a cap-2 core
  advances at most 2 steps per call, and a cap-0 core holds its phase.
  This is a mechanical property of the clamp, tested on a synthetic
  straight-line trajectory so the numbers are exact.
* cap validation: a negative cap raises ValueError.
* the cap is off by default (``phase_advance_cap=None``), so every
  earlier milestone's numbers reproduce exactly: lookahead 1 + alpha
  0.3 on demo_b2_3 still gives the exact M7 number
  (0.6984621786486186 m), and the M9 baseline at sigma 0.005 / alpha
  0.5 still gives the M9 number for that demo.
* the sweep is deterministic: the same (demo, alpha, sigma, config) run
  twice gives identical numbers.
* ``evaluate_all_demos`` accepts the new knobs and tags every result
  with them.
* NO trend in cap or window is asserted: whether a cap or a smaller
  window helps a lagging follower under noise is an empirical question
  the sweep answers, not a property the tests may assume. The tool
  reports the measured best config per (alpha, sigma); the tests only
  pin the machinery.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ecm_policy.core import ReplayFollowCore
from ecm_policy.evaluate import (
    evaluate_all_demos,
    evaluate_lagged_follower,
)
from ecm_policy.trajectory import load_ee_poses_npz, load_gripper_pos_npz
from tools.sweep_robust import CONFIGS, best_config_by_alpha_sigma, config_label

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# Exact M7 measurement (evaluation/results.json, lookahead 1, alpha 0.3).
M7_B2_3_ALPHA03_MEAN_POS_ERR = 0.6984621786486186
# Exact M9 baseline measurement for demo_b2_3 (evaluation/noise_sweep.json,
# lookahead 2, alpha 0.5, sigma 0.005, seed 20261114).
M9_B2_3_ALPHA05_SIGMA0005_MEAN_POS_ERR = 0.548092579222239


def _straight_traj(steps: int = 200) -> np.ndarray:
    """Two arms sliding along +x at 0.01 m/step, identity orientation."""
    traj = np.zeros((steps, 2, 7), dtype=np.float64)
    x = np.arange(steps, dtype=np.float64) * 0.01
    traj[:, 0, 0] = x
    traj[:, 1, 0] = x
    traj[:, 1, 1] = 0.5  # right arm offset in y so the arms differ
    traj[:, :, 3] = 1.0  # w = 1 identity quats
    return traj


def _load_b2_3():
    traj = load_ee_poses_npz(DATA_DIR / "demo_b2_3_ee_poses.npz")
    grip = load_gripper_pos_npz(DATA_DIR / "demo_b2_3_ee_poses.npz")
    assert grip is not None
    return traj, grip


def test_cap_bounds_advance_per_step():
    """A far-ahead observation jumps an uncapped core, not a capped one."""
    traj = _straight_traj()
    far_ahead = traj[150, :, 0:3]  # observation = demo pose 150 steps ahead

    uncapped = ReplayFollowCore(traj, lookahead=1)
    _, info = uncapped.predict_with_info({"eef_pos": far_ahead})
    assert info["phase"] == 150

    capped = ReplayFollowCore(traj, lookahead=1, phase_advance_cap=2)
    _, info = capped.predict_with_info({"eef_pos": far_ahead})
    assert info["phase"] == 2
    # The cap applies per call: repeated calls walk forward 2 steps each.
    _, info = capped.predict_with_info({"eef_pos": far_ahead})
    assert info["phase"] == 4
    _, info = capped.predict_with_info({"eef_pos": far_ahead})
    assert info["phase"] == 6

    held = ReplayFollowCore(traj, lookahead=1, phase_advance_cap=0)
    _, info = held.predict_with_info({"eef_pos": far_ahead})
    assert info["phase"] == 0


def test_cap_does_not_block_small_advances():
    """A nearby observation advances a capped core exactly like uncapped."""
    traj = _straight_traj()
    near = traj[1, :, 0:3]
    for cap in (None, 2, 8):
        core = ReplayFollowCore(traj, lookahead=1, phase_advance_cap=cap)
        _, info = core.predict_with_info({"eef_pos": near})
        assert info["phase"] == 1


def test_negative_cap_rejected():
    with pytest.raises(ValueError, match="phase_advance_cap"):
        ReplayFollowCore(_straight_traj(), phase_advance_cap=-1)


def test_cap_off_by_default_reproduces_m7_and_m9():
    """Default (uncapped) runs reproduce the exact M7 and M9 numbers."""
    traj, grip = _load_b2_3()
    m7 = evaluate_lagged_follower(
        traj, demo="demo_b2_3", alpha=0.3, lookahead=1, gripper_pos=grip
    )
    assert m7.mean_position_error_m == pytest.approx(
        M7_B2_3_ALPHA03_MEAN_POS_ERR, abs=1e-12
    )
    m9 = evaluate_lagged_follower(
        traj,
        demo="demo_b2_3",
        alpha=0.5,
        lookahead=2,
        gripper_pos=grip,
        noise_sigma_m=0.005,
        noise_seed=20261114,
    )
    assert m9.mean_position_error_m == pytest.approx(
        M9_B2_3_ALPHA05_SIGMA0005_MEAN_POS_ERR, abs=1e-9
    )


def test_robust_run_is_deterministic():
    traj, grip = _load_b2_3()
    kwargs = dict(
        demo="demo_b2_3",
        alpha=0.5,
        lookahead=2,
        gripper_pos=grip,
        noise_sigma_m=0.005,
        noise_seed=20261114,
        phase_window=64,
        phase_advance_cap=8,
    )
    first = evaluate_lagged_follower(traj, **kwargs)
    second = evaluate_lagged_follower(traj, **kwargs)
    assert first.as_dict() == second.as_dict()


def test_evaluate_all_demos_tags_config():
    results = evaluate_all_demos(
        DATA_DIR,
        alphas=(0.5,),
        lookahead=2,
        noise_sigma_m=0.0,
        phase_window=32,
        phase_advance_cap=4,
    )
    assert len(results) == 6
    assert all(r.phase_window == 32 for r in results)
    assert all(r.phase_advance_cap == 4 for r in results)
    assert all(r.lookahead == 2 for r in results)


def test_config_labels_cover_known_pairs():
    assert config_label(250, None) == "baseline w250"
    assert config_label(250, 8) == "cap8 w250"
    assert config_label(64, 8) == "win64+cap8"
    assert config_label(32, None) == "win32"
    assert config_label(100, 3) == "w100/cap3"
    assert len(CONFIGS) == 6


def test_best_config_picks_lowest_error():
    rows = [
        type(
            "R",
            (),
            {"alpha": 0.5, "noise_sigma_m": 0.005, "phase_window": w,
             "phase_advance_cap": c, "mean_position_error_m": e},
        )()
        for (w, c, e) in [(250, None, 0.9), (250, 8, 0.4), (64, 8, 0.6)]
    ]
    assert best_config_by_alpha_sigma(rows, 0.005) == {0.5: "cap8 w250"}
