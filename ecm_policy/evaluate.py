"""Offline lagged-follower evaluation for the ECM 2027 chamber policy.

No simulator and no GPU are available on this host, so the harness
simulates the one failure mode the policy core is actually built to
survive: a robot whose end-effector positions lag the commanded poses.
At each step the simulated robot moves a fraction ``alpha`` of the way
toward the commanded pose (first-order lag on position; normalized
lerp on orientation with sign alignment), the policy observes the
robot's current EE positions, re-anchors its phase, and commands the
next demo pose. The harness then scores the rollout against the
recorded demonstration at the same step index:

* position error per arm (Euclidean distance, metres) — mean and max;
* orientation error per arm (angle between quaternions, degrees);
* phase lag (recorded step index minus policy phase) — mean and max;
* gripper agreement (Batch 2 demos only): fraction of steps where the
  emitted binary gripper action matches the recorded gripper state
  thresholded at the joint-range midpoint.

``alpha = 1.0`` is the perfect-follower sanity case: the robot reaches
each commanded pose exactly, so all errors vanish (up to ~1e-6 deg of
float noise in the quaternion path). Smaller ``alpha`` stresses the
closed-loop phase tracker: open-loop replay would drift and never
recover, while phase re-anchoring keeps the policy glued to the
robot's actual pose. All numbers are deterministic to ~1e-9 relative
— no randomness anywhere in the loop.
"""

from __future__ import annotations

import zlib
from dataclasses import asdict, dataclass

import numpy as np

from .core import (
    GRIPPER_THRESHOLD,
    MOBILE_ACTION_DIM,
    ReplayFollowCore,
    _normalize_quats,
)

# Lag fractions swept by the CLI: from a very sluggish follower to a
# perfect one. 0.1 means the robot closes 10% of the remaining gap per
# step; at the demo's recording rate that is a heavily lagging arm.
DEFAULT_ALPHAS = (0.1, 0.3, 0.5, 1.0)


@dataclass
class LaggedFollowerResult:
    """One evaluated rollout: a demo at one lag fraction."""

    demo: str
    alpha: float
    steps: int
    mean_position_error_m: float
    max_position_error_m: float
    mean_orientation_error_deg: float
    max_orientation_error_deg: float
    mean_phase_lag: float
    max_phase_lag: int
    final_phase: int
    gripper_agreement: float | None  # None for demos without a gripper channel
    lookahead: int = 1  # demo steps ahead of the tracked phase the policy commands
    noise_sigma_m: float = 0.0  # std of Gaussian noise added to the observed EE positions
    phase_window: int = 250  # forward search window of the phase tracker
    phase_advance_cap: int | None = None  # max phase jump per step (None = uncapped)
    gripper_threshold: float = GRIPPER_THRESHOLD  # policy replay threshold; the
    # ground-truth recorded-closed labels always use GRIPPER_THRESHOLD (the
    # documented midpoint convention), so varying this measures the
    # agreement lost when the policy's threshold differs from it.

    def as_dict(self) -> dict:
        return asdict(self)


def _quat_angle_deg(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Angle (degrees) between quaternions, per arm. (..., 4) in, (...) out."""
    dot = np.abs(np.sum(q1 * q2, axis=-1))
    dot = np.clip(dot, 0.0, 1.0)
    return np.degrees(2.0 * np.arccos(dot))


def _move_toward(current: np.ndarray, target: np.ndarray, alpha: float) -> np.ndarray:
    """First-order lag step: move ``alpha`` of the way from current to target.

    Positions lerp directly; quaternions lerp then renormalize, with the
    target's sign flipped first so the short arc is taken.
    """
    out = np.array(current, dtype=np.float64, copy=True)
    out[:, 0:3] += alpha * (target[:, 0:3] - current[:, 0:3])
    q_cur = current[:, 3:7]
    q_tgt = target[:, 3:7].copy()
    flip = np.sum(q_cur * q_tgt, axis=-1, keepdims=True) < 0
    q_tgt = np.where(flip, -q_tgt, q_tgt)
    q_new = q_cur + alpha * (q_tgt - q_cur)
    norms = np.linalg.norm(q_new, axis=-1, keepdims=True)
    out[:, 3:7] = q_new / np.where(norms < 1e-12, 1.0, norms)
    return out


def evaluate_lagged_follower(
    ee_poses: np.ndarray,
    *,
    demo: str = "demo",
    alpha: float = 0.3,
    lookahead: int = 1,
    gripper_pos: np.ndarray | None = None,
    noise_sigma_m: float = 0.0,
    noise_seed: int = 0,
    phase_window: int = 250,
    phase_advance_cap: int | None = None,
    gripper_threshold: float = GRIPPER_THRESHOLD,
) -> LaggedFollowerResult:
    """Run one demo end to end against a lagged follower and score it.

    ``noise_sigma_m`` adds zero-mean Gaussian noise (std ``noise_sigma_m``
    metres, per arm, per step) to the EE positions the policy *observes*.
    The true robot state and the recorded trajectory stay exact: only the
    observation is noisy, which is the failure mode a real evaluator's
    state estimate introduces. The noise stream is seeded per demo
    (``noise_seed`` plus a CRC of the demo name), so a run is exactly
    reproducible and different demos do not share one noise sequence.
    ``noise_sigma_m = 0`` reproduces the noiseless evaluation exactly.
    ``gripper_threshold`` is the policy's replay threshold only; the
    recorded-closed ground truth is always labelled at GRIPPER_THRESHOLD
    (the documented midpoint), so a threshold different from it measures
    the agreement cost of a misplaced transition point.
    """
    if not (0.0 < alpha <= 1.0):
        raise ValueError(f"alpha must be in (0, 1], got {alpha}")
    if noise_sigma_m < 0.0:
        raise ValueError(f"noise_sigma_m must be >= 0, got {noise_sigma_m}")
    # Seeded per demo so the stream is reproducible and demo-specific.
    rng = (
        np.random.default_rng(noise_seed + zlib.crc32(demo.encode()))
        if noise_sigma_m > 0.0
        else None
    )
    traj = np.array(ee_poses, dtype=np.float64)  # copy: normalizing below
    # must not mutate the caller's array, or a second run on the same
    # array would see already-normalized quats and differ in the 10th
    # decimal (caught by the M8 determinism test).
    # Score against the same normalized quaternions the core stores:
    # recorded quats are not exactly unit-norm (norms 0.9999999–
    # 1.0000003 in demo_b2_3), and scoring the normalized robot pose
    # against raw quats would report a ~0.001 deg artifact as error.
    traj[:, :, 3:7] = _normalize_quats(traj[:, :, 3:7])
    core = ReplayFollowCore(
        traj,
        lookahead=lookahead,
        gripper_pos=gripper_pos,
        gripper_threshold=gripper_threshold,
        phase_window=phase_window,
        phase_advance_cap=phase_advance_cap,
    )
    core.reset()

    # The robot starts exactly at the demo's first pose (the evaluation
    # scene's rest pose); the lag then accumulates from the policy's own
    # lookahead commands, not from a bogus initial offset.
    robot = traj[0].copy()
    steps = int(traj.shape[0])
    pos_errors: list[float] = []
    ori_errors: list[float] = []
    phase_lags: list[int] = []
    gripper_match = 0
    gripper_total = 0

    for step in range(steps):
        # The policy sees the robot's EE positions plus observation noise;
        # scoring below still uses the TRUE robot pose, so noise can only
        # hurt through the policy's phase re-anchoring, never by moving
        # the goalposts.
        obs_pos = robot[:, 0:3]
        if rng is not None:
            obs_pos = obs_pos + rng.normal(
                0.0, noise_sigma_m, size=obs_pos.shape
            )
        action, info = core.predict_with_info(
            {"eef_pos": obs_pos}
        )
        assert action.shape == (MOBILE_ACTION_DIM,), action.shape
        commanded = action[:14].reshape(2, 7)
        # Score the robot's pose BEFORE this step's move against the
        # recorded pose at the same index: that is the tracking error
        # the evaluator would actually see at step t.
        pos_errors.append(
            float(np.linalg.norm(robot[:, 0:3] - traj[step, :, 0:3], axis=-1).mean())
        )
        ori_errors.append(
            float(_quat_angle_deg(robot[:, 3:7], traj[step, :, 3:7]).mean())
        )
        phase_lags.append(int(step - info["phase"]))
        if gripper_pos is not None:
            # Compare against the recorded gripper state at the CURRENT
            # step index, not at the policy's target index: the emitted
            # action is indexed by phase + lookahead, so this measures
            # whether the gripper command is in sync with where the
            # recording is at wall-clock time. (Comparing at target_index
            # would be a tautology — the emitted action is derived from
            # that very row.)
            recorded_closed = gripper_pos[step] >= GRIPPER_THRESHOLD
            # Emitted gripper action: < 0 closes, >= 0 opens.
            emitted_closed = action[14:16] < 0
            gripper_match += int(np.all(recorded_closed == emitted_closed))
            gripper_total += 1
        robot = _move_toward(robot, commanded, alpha)

    pos = np.asarray(pos_errors)
    ori = np.asarray(ori_errors)
    lag = np.asarray(phase_lags, dtype=np.int64)
    return LaggedFollowerResult(
        demo=demo,
        alpha=float(alpha),
        steps=steps,
        mean_position_error_m=float(pos.mean()),
        max_position_error_m=float(pos.max()),
        mean_orientation_error_deg=float(ori.mean()),
        max_orientation_error_deg=float(ori.max()),
        mean_phase_lag=float(lag.mean()),
        max_phase_lag=int(lag.max()),
        final_phase=int(core.phase),
        gripper_agreement=(
            (gripper_match / gripper_total) if gripper_total else None
        ),
        lookahead=int(lookahead),
        noise_sigma_m=float(noise_sigma_m),
        phase_window=int(phase_window),
        phase_advance_cap=phase_advance_cap,
        gripper_threshold=float(gripper_threshold),
    )


def evaluate_all_demos(
    data_dir,
    *,
    alphas=DEFAULT_ALPHAS,
    lookahead: int = 1,
    noise_sigma_m: float = 0.0,
    noise_seed: int = 0,
    phase_window: int = 250,
    phase_advance_cap: int | None = None,
    gripper_threshold: float = GRIPPER_THRESHOLD,
) -> list[LaggedFollowerResult]:
    """Evaluate every bundled demo at every lag fraction, in filename order."""
    from pathlib import Path

    from .trajectory import load_ee_poses_npz, load_gripper_pos_npz

    results: list[LaggedFollowerResult] = []
    for path in sorted(Path(data_dir).glob("*_ee_poses.npz")):
        name = path.name[: -len("_ee_poses.npz")]
        traj = load_ee_poses_npz(path)
        grip = load_gripper_pos_npz(path)
        for alpha in alphas:
            results.append(
                evaluate_lagged_follower(
                    traj,
                    demo=name,
                    alpha=alpha,
                    lookahead=lookahead,
                    gripper_pos=grip,
                    noise_sigma_m=noise_sigma_m,
                    noise_seed=noise_seed,
                    phase_window=phase_window,
                    phase_advance_cap=phase_advance_cap,
                    gripper_threshold=gripper_threshold,
                )
            )
    return results


def results_markdown_table(results: list[LaggedFollowerResult]) -> str:
    """Render results as a Markdown table (one row per demo x alpha)."""
    lines = [
        "| demo | alpha | steps | mean pos err (m) | max pos err (m) | "
        "mean ori err (deg) | max phase lag | gripper agree |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        grip = (
            f"{r.gripper_agreement * 100:.1f}%"
            if r.gripper_agreement is not None
            else "n/a (no channel)"
        )
        lines.append(
            f"| {r.demo} | {r.alpha} | {r.steps} | "
            f"{r.mean_position_error_m:.4f} | {r.max_position_error_m:.4f} | "
            f"{r.mean_orientation_error_deg:.2f} | {r.max_phase_lag} | {grip} |"
        )
    return "\n".join(lines)
