"""Tests for the observation-noise sweep (milestone 9).

The real Isaac Sim evaluator will not hand the policy perfect state:
its EE-position observations carry sensor / state-estimation noise,
and the policy re-anchors its phase from those observations. The
sweep measures how much noise the phase tracker tolerates. Only the
shortest bundled demo (demo_b2_3, 4022 steps) is used here so the
suite stays fast; the full 6-demo sweep lives in
``tools/sweep_noise.py``.

What is asserted — and what is deliberately NOT asserted:

* sigma = 0 reproduces the noiseless evaluation EXACTLY (same result
  object, field for field): noise is a pure addition, never a change
  to the baseline path.
* the zero-noise sweep row reproduces the exact M8 number at
  lookahead 2, alpha 0.3 on demo_b2_3 (mean position error
  0.5738896333694334 m, per-demo value from
  ``evaluation/lookahead_sweep.json``): the sweep reuses the same code
  path, so the M8 table must come back bit-for-bit.
* the sweep is deterministic: the same (demo, alpha, lookahead,
  sigma, seed) run twice gives identical numbers; a different seed
  gives different numbers (the noise is actually drawn, not ignored).
* a negative sigma is rejected, and ``evaluate_all_demos`` accepts a
  sigma and tags every result with it.
* noise changes the outcome: at sigma = 2 cm the mean position error
  differs from the noiseless baseline, and every reported error stays
  finite (no NaN / inf blow-up).
* NO trend in sigma is asserted: whether tracking degrades smoothly
  or cliffs at some sigma is an empirical question the sweep answers,
  not a property the tests may assume. The tool reports the measured
  table; the tests only pin the machinery.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from ecm_policy.evaluate import (
    evaluate_all_demos,
    evaluate_lagged_follower,
)
from ecm_policy.trajectory import load_ee_poses_npz, load_gripper_pos_npz
from tools.sweep_noise import NOISE_SEED, SWEEP_LOOKAHEAD

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
REPO_ROOT = Path(__file__).resolve().parent.parent

# Exact M8 measurement (evaluation/lookahead_sweep.json, lookahead 2,
# alpha 0.3, demo_b2_3).
M8_B2_3_ALPHA03_LA2_MEAN_POS_ERR = 0.5738896333694334


def _load_b2_3():
    traj = load_ee_poses_npz(DATA_DIR / "demo_b2_3_ee_poses.npz")
    grip = load_gripper_pos_npz(DATA_DIR / "demo_b2_3_ee_poses.npz")
    assert grip is not None
    return traj, grip


def _m8_b2_3_alpha03_la2() -> float:
    rows = json.loads(
        (REPO_ROOT / "evaluation" / "lookahead_sweep.json").read_text(
            encoding="utf-8"
        )
    )
    for row in rows:
        if (
            row["demo"] == "demo_b2_3"
            and row["alpha"] == 0.3
            and row["lookahead"] == 2
        ):
            return float(row["mean_position_error_m"])
    raise AssertionError("M8 lookahead_sweep.json has no demo_b2_3/0.3/la2 row")


def test_zero_noise_reproduces_noiseless_evaluation_exactly() -> None:
    traj, grip = _load_b2_3()
    baseline = evaluate_lagged_follower(
        traj, demo="demo_b2_3", alpha=0.3, lookahead=2, gripper_pos=grip
    )
    noisy_zero = evaluate_lagged_follower(
        traj,
        demo="demo_b2_3",
        alpha=0.3,
        lookahead=2,
        gripper_pos=grip,
        noise_sigma_m=0.0,
        noise_seed=NOISE_SEED,
    )
    assert noisy_zero.as_dict() == baseline.as_dict()


def test_zero_noise_reproduces_m8_lookahead2_number() -> None:
    traj, grip = _load_b2_3()
    result = evaluate_lagged_follower(
        traj,
        demo="demo_b2_3",
        alpha=0.3,
        lookahead=2,
        gripper_pos=grip,
        noise_sigma_m=0.0,
        noise_seed=NOISE_SEED,
    )
    assert result.mean_position_error_m == pytest.approx(
        _m8_b2_3_alpha03_la2(), abs=1e-12
    )
    assert result.mean_position_error_m == pytest.approx(
        M8_B2_3_ALPHA03_LA2_MEAN_POS_ERR, abs=1e-12
    )


def test_noise_sweep_is_deterministic_for_same_seed() -> None:
    traj, grip = _load_b2_3()
    first = evaluate_lagged_follower(
        traj,
        demo="demo_b2_3",
        alpha=0.3,
        lookahead=2,
        gripper_pos=grip,
        noise_sigma_m=0.01,
        noise_seed=NOISE_SEED,
    )
    second = evaluate_lagged_follower(
        traj,
        demo="demo_b2_3",
        alpha=0.3,
        lookahead=2,
        gripper_pos=grip,
        noise_sigma_m=0.01,
        noise_seed=NOISE_SEED,
    )
    assert first.as_dict() == second.as_dict()


def test_different_seed_draws_different_noise() -> None:
    traj, grip = _load_b2_3()
    seed_a = evaluate_lagged_follower(
        traj,
        demo="demo_b2_3",
        alpha=0.3,
        lookahead=2,
        gripper_pos=grip,
        noise_sigma_m=0.01,
        noise_seed=NOISE_SEED,
    )
    seed_b = evaluate_lagged_follower(
        traj,
        demo="demo_b2_3",
        alpha=0.3,
        lookahead=2,
        gripper_pos=grip,
        noise_sigma_m=0.01,
        noise_seed=NOISE_SEED + 1,
    )
    assert seed_a.as_dict() != seed_b.as_dict()


def test_noise_changes_outcome_but_stays_finite() -> None:
    traj, grip = _load_b2_3()
    baseline = evaluate_lagged_follower(
        traj, demo="demo_b2_3", alpha=0.3, lookahead=2, gripper_pos=grip
    )
    noisy = evaluate_lagged_follower(
        traj,
        demo="demo_b2_3",
        alpha=0.3,
        lookahead=2,
        gripper_pos=grip,
        noise_sigma_m=0.02,
        noise_seed=NOISE_SEED,
    )
    assert noisy.mean_position_error_m != baseline.mean_position_error_m
    for value in (
        noisy.mean_position_error_m,
        noisy.max_position_error_m,
        noisy.mean_orientation_error_deg,
        noisy.max_orientation_error_deg,
        noisy.mean_phase_lag,
    ):
        assert np.isfinite(value)


def test_negative_sigma_rejected() -> None:
    traj, grip = _load_b2_3()
    with pytest.raises(ValueError, match="noise_sigma_m"):
        evaluate_lagged_follower(
            traj,
            demo="demo_b2_3",
            alpha=0.3,
            lookahead=2,
            gripper_pos=grip,
            noise_sigma_m=-0.001,
        )


def test_evaluate_all_demos_tags_noise_sigma() -> None:
    results = evaluate_all_demos(
        DATA_DIR,
        alphas=(0.3,),
        lookahead=SWEEP_LOOKAHEAD,
        noise_sigma_m=0.005,
        noise_seed=NOISE_SEED,
    )
    assert results
    assert all(r.noise_sigma_m == 0.005 for r in results)
    assert all(r.lookahead == SWEEP_LOOKAHEAD for r in results)
