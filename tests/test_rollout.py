"""Full-episode rollout integrity for the ECM 2027 chamber policy.

These tests drive a whole demonstration end to end through the policy
core (no simulator, no GPU) and assert the properties an organizer-side
evaluation would need to hold for every step, not just for one reset:

* every emitted action is finite, 20-dimensional, and carries unit-norm
  quaternions in the stored (w, x, y, z) order;
* gripper channels stay at the configured fixed command and wheel
  channels stay at zero for the whole episode (the released data has
  neither channel, so the policy must not invent values mid-episode);
* phase is monotonic and reaches the end of the trajectory;
* a multi-demo rollout never switches demos mid-episode.

The shortest bundled demo (demo_9, 5122 steps) is used so the suite
stays fast on a CPU-only host; the same invariants hold for the longer
demos because every core shares the same code path.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ecm_policy.core import BASE_ACTION_DIM, MOBILE_ACTION_DIM, ReplayFollowCore
from ecm_policy.multidemo import MultiDemoCore
from ecm_policy.trajectory import load_ee_poses_npz

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def _check_action(action: np.ndarray) -> None:
    assert action.shape == (MOBILE_ACTION_DIM,), action.shape
    assert np.all(np.isfinite(action)), "action contains NaN/inf"
    for arm in range(2):
        quat = action[arm * 7 + 3 : arm * 7 + 7]
        assert abs(float(np.linalg.norm(quat)) - 1.0) < 1e-5, quat
    assert float(action[14]) == 1.0 and float(action[15]) == 1.0
    assert np.all(action[BASE_ACTION_DIM:] == 0.0), "wheel channels must be zero"


def test_open_loop_full_episode_reaches_end() -> None:
    traj = load_ee_poses_npz(DATA_DIR / "demo_9_ee_poses.npz")
    core = ReplayFollowCore(traj)
    core.reset()
    prev_phase = -1
    last_action = None
    for _ in range(core.num_steps):
        action, info = core.predict_with_info()
        _check_action(action)
        assert info["phase"] >= prev_phase, "phase must be monotonic"
        prev_phase = info["phase"]
        last_action = action
    assert core.phase == core.num_steps - 1
    assert core.steps_taken == core.num_steps
    # Final commanded pose is the demo's last pose (lookahead clamped).
    np.testing.assert_allclose(
        last_action[:14], traj[-1].reshape(-1), rtol=0, atol=1e-6
    )


def test_closed_loop_tracking_follows_true_poses() -> None:
    traj = load_ee_poses_npz(DATA_DIR / "demo_9_ee_poses.npz")
    core = ReplayFollowCore(traj)
    core.reset()
    prev_phase = -1
    for step in range(core.num_steps):
        action, info = core.predict_with_info(
            {"eef_pos": traj[step, :, 0:3]}
        )
        _check_action(action)
        assert info["phase"] >= prev_phase, "phase must be monotonic"
        assert info["phase_tracked"] is True
        # Feeding the demo's own pose at step t must keep phase near t:
        # the tracker only searches forward, so it can lag but never run
        # ahead of the supplied pose index by more than the lookahead.
        assert info["phase"] <= step + 1, (info["phase"], step)
        prev_phase = info["phase"]
    assert core.phase == core.num_steps - 1


def test_multidemo_full_episode_never_switches_demo() -> None:
    core = MultiDemoCore.from_npz_dir(DATA_DIR)
    demo_7 = load_ee_poses_npz(DATA_DIR / "demo_7_ee_poses.npz")
    reset_info = core.reset(observation={"eef_pos": demo_7[0, :, 0:3]})
    assert reset_info["selected"] == "demo_7"
    steps = core.num_steps
    for _ in range(steps):
        action, info = core.predict_with_info()
        assert info["selected"] == "demo_7", "must not switch demos mid-episode"
        _check_action(action)
    assert core.phase == steps - 1
    assert core.selected == "demo_7"
