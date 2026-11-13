"""Tests for the offline lagged-follower evaluation harness.

The harness simulates a robot that moves a fraction ``alpha`` of the
way toward each commanded pose per step and scores the policy against
the recorded demonstration at the same step index. The shortest
bundled demo (demo_b2_3, 4022 steps) is used for the alpha sweeps so
the suite stays fast on a CPU-only host.

What is asserted here — and what is deliberately NOT asserted:

* alpha = 1.0 (perfect follower) reproduces the recording exactly:
  zero position error and orientation error at float-noise level
  (~1e-6 deg), zero phase lag. This is the harness's own sanity check.
* max phase lag strictly decreases as alpha grows: a faster follower
  falls less far behind the recording in wall-clock time.
* gripper agreement strictly increases with alpha on Batch 2 demos:
  the emitted gripper action is indexed by the lagging phase, so a
  slower follower's gripper state is further out of sync with the
  recording at the same step index.
* mean position error is NOT asserted monotonic in alpha: measured
  values are non-monotonic (0.52 m at alpha 0.1 vs 0.62 m at 0.5 on
  demo_0), because the phase tracker keeps re-anchoring the target to
  the robot's own neighbourhood. The numbers are reported, not forced
  into a trend they do not have.
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

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
ALPHAS = (0.1, 0.3, 0.5, 1.0)


def _load_b2_3():
    traj = load_ee_poses_npz(DATA_DIR / "demo_b2_3_ee_poses.npz")
    grip = load_gripper_pos_npz(DATA_DIR / "demo_b2_3_ee_poses.npz")
    assert grip is not None
    return traj, grip


def test_perfect_follower_reproduces_recording_exactly() -> None:
    traj, grip = _load_b2_3()
    result = evaluate_lagged_follower(
        traj, demo="demo_b2_3", alpha=1.0, gripper_pos=grip
    )
    assert result.steps == 4022
    assert result.mean_position_error_m == pytest.approx(0.0, abs=1e-9)
    assert result.max_position_error_m == pytest.approx(0.0, abs=1e-9)
    # Orientation error is float noise only (~1e-6 deg from the lerp +
    # renormalize path); the real signal at lower alphas is 40+ deg.
    assert result.mean_orientation_error_deg == pytest.approx(0.0, abs=1e-5)
    assert result.max_phase_lag == 0
    assert result.final_phase == 4021
    # Gripper is one lookahead step ahead of the recorded state, so
    # agreement is near-perfect but not exactly 1.0 at transitions.
    assert result.gripper_agreement == pytest.approx(1.0, abs=0.005)


def test_phase_lag_shrinks_as_follower_gets_faster() -> None:
    traj, grip = _load_b2_3()
    lags = [
        evaluate_lagged_follower(
            traj, demo="demo_b2_3", alpha=a, gripper_pos=grip
        ).max_phase_lag
        for a in ALPHAS
    ]
    assert lags == sorted(lags, reverse=True), lags
    assert all(b < a for a, b in zip(lags, lags[1:])), lags
    assert lags[-1] == 0


def test_gripper_agreement_grows_as_follower_gets_faster() -> None:
    traj, grip = _load_b2_3()
    agrees = [
        evaluate_lagged_follower(
            traj, demo="demo_b2_3", alpha=a, gripper_pos=grip
        ).gripper_agreement
        for a in ALPHAS
    ]
    assert all(b > a for a, b in zip(agrees, agrees[1:])), agrees
    assert agrees[-1] > 0.99


def test_evaluation_is_deterministic() -> None:
    traj, grip = _load_b2_3()
    first = evaluate_lagged_follower(
        traj, demo="demo_b2_3", alpha=0.3, gripper_pos=grip
    ).as_dict()
    second = evaluate_lagged_follower(
        traj, demo="demo_b2_3", alpha=0.3, gripper_pos=grip
    ).as_dict()
    # Float results agree to ~1e-9 relative (BLAS summation order can
    # vary the last bits run to run); ints and labels agree exactly.
    assert first.keys() == second.keys()
    for key, a in first.items():
        b = second[key]
        if isinstance(a, float):
            assert a == pytest.approx(b, rel=1e-9, abs=1e-12), key
        else:
            assert a == b, key


def test_all_demos_evaluate_cleanly_at_perfect_follow() -> None:
    results = evaluate_all_demos(DATA_DIR, alphas=(1.0,))
    assert len(results) == 6
    names = {r.demo for r in results}
    assert names == {
        "demo_0",
        "demo_2",
        "demo_7",
        "demo_9",
        "demo_b2_0",
        "demo_b2_3",
    }
    for r in results:
        assert np.isfinite(r.mean_position_error_m)
        assert np.isfinite(r.mean_orientation_error_deg)
        assert r.mean_position_error_m == pytest.approx(0.0, abs=1e-9)
        assert r.max_phase_lag == 0
        assert r.final_phase == r.steps - 1
        if r.demo.startswith("demo_b2"):
            assert r.gripper_agreement is not None
            assert r.gripper_agreement > 0.99
        else:
            assert r.gripper_agreement is None


def test_invalid_alpha_rejected() -> None:
    traj, _grip = _load_b2_3()
    for bad in (0.0, -0.5, 1.5):
        with pytest.raises(ValueError, match="alpha"):
            evaluate_lagged_follower(traj, demo="demo_b2_3", alpha=bad)
