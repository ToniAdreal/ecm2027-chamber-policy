"""Transport-agnostic policy core for the ECM 2027 Chamber Service task.

Strategy (v1): demonstration trajectory following.

The only demonstration modality released so far is dual-arm end-effector
pose trajectories (position + orientation, no gripper channel, no base
motion). The Galbot-Mobile embodiment in the official starter repository is
driven by *absolute* differential-IK pose commands
(``DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False)``),
so a recorded EE pose can be fed back as the commanded pose directly. This
core therefore:

* tracks a phase index into the demo trajectory, advancing monotonically;
* when the caller supplies the robot's current EE positions, re-anchors the
  phase to the demo point nearest to the current pose inside a forward
  search window (closed-loop phase tracking, tolerant to timing drift);
* otherwise advances the phase one demo step per policy step (open loop);
* commands the demo pose at ``phase + lookahead`` for both arms;
* replays the recorded gripper state when the demo carries one: Batch 2
  bundles store ``gripper_pos`` ``(T, 2)``, the Galbot gripper joint
  positions (0.0 fully open, 1.703 fully closed, per the starter repo's
  ``BinaryJointPositionActionCfg``), thresholded at the joint-range
  midpoint 0.8515 into the binary gripper action (``< 0`` close,
  ``>= 0`` open, per the starter repo's ``BinaryJointPositionAction``
  convention). Demos without a gripper channel (all of Batch 1) still
  hold both grippers at a fixed configurable command, and the policy
  commands zero wheel velocity throughout.

Action layout (Galbot-Mobile, 20 dims, from the starter repo):
``[left pose 7, right pose 7, left gripper 1, right gripper 1, wheels 4]``.
The fixed-base Galbot-Abs layout drops the wheels (16 dims). Gripper sign
convention follows the starter repo's Galbot config
(``galbot.py: preprocess_device_action``): the device-level ``+1.0``
close press is inverted to ``-1.0`` before the env action vector, i.e.
in the emitted action ``< 0`` closes and ``>= 0`` opens. Arm pose
quaternions are emitted in Isaac Lab's native ``(w, x, y, z)`` order by
default; set ``action_quat_order="xyzw"`` to reorder on output if the
official validator publishes a different convention (the official Docker
service schema was still unpublished as of 2026-10-31).

Known v1 limitations, stated plainly: Batch 1 demos carry no gripper
channel, so on those demos the grippers never change state and
grasp-dependent milestones cannot be earned by this policy as-is; and
replay assumes the evaluation scene's initial layout matches the demo
recording (the same fixed local scene the starter repo's own replay
tooling assumes). The binary gripper mapping itself is the starter
repo's own convention, but the official evaluation service has never
published its expected gripper action scale, so the mapping is
documented here rather than silently assumed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .trajectory import (
    load_ee_poses_hdf5,
    load_ee_poses_npz,
    load_gripper_pos_hdf5,
    load_gripper_pos_npz,
)

POSE_DIM = 7
NUM_ARMS = 2
BASE_ACTION_DIM = 16  # 2 arms * 7 + 2 grippers
MOBILE_ACTION_DIM = 20  # + 4 wheels

# Galbot gripper joint range, from the starter repo's Galbot config
# (BinaryJointPositionActionCfg open/close_command_expr): 0.0 is fully
# open, 1.703 fully closed. The binary gripper action is < 0 close,
# >= 0 open. The replay threshold is the joint-range midpoint; the two
# verified Batch 2 episodes are strongly bimodal around it (39.7% of
# samples < 0.05, most of the rest in 1.13-1.70, sparse transition band).
GRIPPER_JOINT_OPEN = 0.0
GRIPPER_JOINT_CLOSED = 1.703
GRIPPER_THRESHOLD = (GRIPPER_JOINT_OPEN + GRIPPER_JOINT_CLOSED) / 2.0
GRIPPER_CLOSE_CMD = -1.0
GRIPPER_OPEN_CMD = 1.0


def _gripper_commands_for(gripper_pos_row: np.ndarray) -> np.ndarray:
    """Binary gripper commands for one recorded gripper_pos row (2,).

    Joint position at or above the range midpoint means the demo's
    gripper was closed there, so emit the close command (< 0); below
    it, emit the open command (>= 0).
    """
    return np.where(
        np.asarray(gripper_pos_row, dtype=np.float64) >= GRIPPER_THRESHOLD,
        GRIPPER_CLOSE_CMD,
        GRIPPER_OPEN_CMD,
    )


def _normalize_quats(quats: np.ndarray) -> np.ndarray:
    """Normalize (..., 4) quaternions; replace zero-norm rows with identity."""
    norms = np.linalg.norm(quats, axis=-1, keepdims=True)
    safe = np.where(norms < 1e-12, 1.0, norms)
    out = quats / safe
    bad = (norms < 1e-12).reshape(-1)
    if np.any(bad):
        out = out.reshape(-1, 4)
        out[bad] = (1.0, 0.0, 0.0, 0.0)
        out = out.reshape(quats.shape)
    return out


class ReplayFollowCore:
    """Demo-trajectory-following policy core (pure NumPy, CPU only)."""

    def __init__(
        self,
        ee_poses: np.ndarray,
        *,
        actions_dim: int = MOBILE_ACTION_DIM,
        lookahead: int = 1,
        phase_window: int = 250,
        gripper_left: float = 1.0,
        gripper_right: float = 1.0,
        gripper_pos: np.ndarray | None = None,
        action_quat_order: str = "wxyz",
    ):
        traj = np.asarray(ee_poses, dtype=np.float64)
        if traj.ndim != 3 or tuple(traj.shape[1:]) != (NUM_ARMS, POSE_DIM):
            raise ValueError(f"ee_poses must have shape (T, 2, 7), got {traj.shape}")
        if not np.all(np.isfinite(traj)):
            raise ValueError("ee_poses contains NaN or infinite values")
        if gripper_pos is not None:
            grip = np.asarray(gripper_pos, dtype=np.float64)
            if grip.ndim != 2 or tuple(grip.shape) != (traj.shape[0], NUM_ARMS):
                raise ValueError(
                    "gripper_pos must have shape (T, 2) matching ee_poses, "
                    f"got {grip.shape} for {traj.shape[0]} steps"
                )
            if not np.all(np.isfinite(grip)):
                raise ValueError("gripper_pos contains NaN or infinite values")
            self._gripper_pos = grip.copy()
        else:
            self._gripper_pos = None
        if actions_dim < BASE_ACTION_DIM:
            raise ValueError(
                f"actions_dim must be >= {BASE_ACTION_DIM}, got {actions_dim}"
            )
        if action_quat_order not in ("wxyz", "xyzw"):
            raise ValueError(f"unknown action_quat_order: {action_quat_order!r}")
        self._traj = traj.copy()
        self._traj[:, :, 3:7] = _normalize_quats(self._traj[:, :, 3:7])
        self.actions_dim = int(actions_dim)
        self.lookahead = max(0, int(lookahead))
        self.phase_window = max(1, int(phase_window))
        self.gripper_left = float(gripper_left)
        self.gripper_right = float(gripper_right)
        self.action_quat_order = action_quat_order
        self._phase = 0
        self._steps = 0
        self._closed = False
        self._task_info: dict | None = None

    # -- construction helpers -------------------------------------------------

    @classmethod
    def from_npz(cls, path: str | Path, **kwargs) -> "ReplayFollowCore":
        if "gripper_pos" not in kwargs:
            kwargs["gripper_pos"] = load_gripper_pos_npz(path)
        return cls(load_ee_poses_npz(path), **kwargs)

    @classmethod
    def from_hdf5(cls, path: str | Path, episode_key: str, **kwargs) -> "ReplayFollowCore":
        if "gripper_pos" not in kwargs:
            try:
                kwargs["gripper_pos"] = load_gripper_pos_hdf5(path, episode_key)
            except ValueError:
                kwargs["gripper_pos"] = None
        return cls(load_ee_poses_hdf5(path, episode_key), **kwargs)

    # -- introspection --------------------------------------------------------

    @property
    def num_steps(self) -> int:
        return int(self._traj.shape[0])

    @property
    def first_ee_positions(self) -> np.ndarray:
        """First-frame EE positions (2, 3), used for multi-demo selection."""
        return self._traj[0, :, 0:3].copy()

    @property
    def phase(self) -> int:
        return self._phase

    @property
    def steps_taken(self) -> int:
        return self._steps

    @property
    def closed(self) -> bool:
        return self._closed

    def set_actions_dim(self, actions_dim: int) -> None:
        """Adjust the emitted action width (e.g. from evaluator metadata)."""
        if int(actions_dim) < BASE_ACTION_DIM:
            raise ValueError(
                f"actions_dim must be >= {BASE_ACTION_DIM}, got {actions_dim}"
            )
        self.actions_dim = int(actions_dim)

    @property
    def gripper_mode(self) -> str:
        """'replay' when a recorded gripper channel is consumed, else 'fixed'."""
        return "replay" if self._gripper_pos is not None else "fixed"

    def health(self) -> dict:
        return {
            "status": "ok",
            "ready": not self._closed,
            "closed": self._closed,
            "trajectory_steps": self.num_steps,
            "actions_dim": self.actions_dim,
            "gripper_mode": self.gripper_mode,
            "phase": self._phase,
            "steps_taken": self._steps,
        }

    # -- lifecycle ------------------------------------------------------------

    def reset(self, task_info: dict | None = None) -> dict:
        self._phase = 0
        self._steps = 0
        self._closed = False
        self._task_info = dict(task_info) if task_info else None
        return {"phase": self._phase, "steps_taken": self._steps}

    def close(self) -> None:
        self._closed = True

    # -- inference ------------------------------------------------------------

    def predict(self, observation: dict | None = None) -> np.ndarray:
        action, _info = self.predict_with_info(observation)
        return action

    def predict_with_info(self, observation: dict | None = None):
        if self._closed:
            raise RuntimeError("policy is closed; call reset() before predict()")
        ee_pos = self._extract_ee_pos(observation)
        if ee_pos is not None:
            self._phase = self._track_phase(ee_pos)
        else:
            self._phase = max(self._phase, min(self._steps, self.num_steps - 1))
        target = min(self._phase + self.lookahead, self.num_steps - 1)
        action = self._assemble(self._traj[target], target)
        self._steps += 1
        info = {
            "phase": self._phase,
            "target_index": int(target),
            "steps_taken": self._steps,
            "phase_tracked": ee_pos is not None,
        }
        return action, info

    # -- internals --------------------------------------------------------------

    @staticmethod
    def _extract_ee_pos(observation) -> np.ndarray | None:
        """Pull current EE positions (2, 3) out of an observation dict.

        Accepts the keys used by the starter repo's observation terms
        (``eef_pos``), optionally nested under ``policy`` or
        ``embodiment_general_obs``, with an optional leading env axis.
        Returns None when no usable positions are present.
        """
        if not isinstance(observation, dict):
            return None
        candidates = [observation]
        for key in ("policy", "embodiment_general_obs"):
            nested = observation.get(key)
            if isinstance(nested, dict):
                candidates.append(nested)
        for cand in candidates:
            raw = cand.get("eef_pos")
            if raw is None:
                continue
            arr = np.asarray(raw, dtype=np.float64)
            if arr.shape == (1, NUM_ARMS, 3):
                arr = arr[0]
            if arr.shape == (NUM_ARMS, 3) and np.all(np.isfinite(arr)):
                return arr
        return None

    def _track_phase(self, ee_pos: np.ndarray) -> int:
        """Nearest demo index to the current EE positions, searched forward."""
        start = self._phase
        stop = min(self.num_steps, start + self.phase_window)
        window = self._traj[start:stop, :, 0:3]
        dist = np.linalg.norm(window - ee_pos[None, :, :], axis=-1).mean(axis=-1)
        return start + int(np.argmin(dist))

    def _assemble(self, pose_pair: np.ndarray, target_index: int) -> np.ndarray:
        parts = []
        for arm in range(NUM_ARMS):
            pose = np.array(pose_pair[arm], dtype=np.float64)
            if self.action_quat_order == "xyzw":
                pose[3:7] = pose[[4, 5, 6, 3]]  # stored wxyz -> xyzw
            parts.append(pose)
        if self._gripper_pos is not None:
            parts.append(_gripper_commands_for(self._gripper_pos[target_index]))
        else:
            parts.append(np.array([self.gripper_left, self.gripper_right]))
        action = np.concatenate(parts)
        if self.actions_dim > BASE_ACTION_DIM:
            action = np.concatenate(
                [action, np.zeros(self.actions_dim - BASE_ACTION_DIM)]
            )
        return action.astype(np.float32)
