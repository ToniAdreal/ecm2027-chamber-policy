"""Trajectory IO and validation for ECM 2027 demonstration end-effector poses.

The demonstration data released for the Singapore Embodied AI Challenge for
Manufacturing (Hugging Face dataset ``rocochallenge2025/ecm2027_sim_data``)
stores, per episode, a ``data/<episode>/ee_poses`` array of shape
``(T, 2, 7)`` float32: for each timestep, the end-effector pose of both arms
of the Galbot robot. Each 7-vector is ``[x, y, z, qw, qx, qy, qz]``.

Convention notes (derived from the official starter repository, not assumed
silently):

* Positions come from the starter repo's ``ee_frame_pos`` observation term,
  which returns ``target_pos_w - env_origins`` (env-local frame, metres).
* Quaternions come from ``ee_frame_quat``, which returns the Isaac Lab
  FrameTransformer ``target_quat_w``. Isaac Lab stores frame quaternions in
  ``(w, x, y, z)`` order, and the Galbot config in the starter repo states
  the same ("FrameTransformer uses (w, x, y, z)").
* Arm index 0 is the left arm and index 1 is the right arm, matching the
  ``target_frames`` order in the Galbot scene config
  (``tool_left_hand`` first, ``tool_right_hand`` second).

Batch 2 layout (verified 2026-11-10 against two real Batch 2 recordings
from ``rocochallenge2025/ecm2027_sim_data``): Batch 2 files do NOT carry
the combined ``data/<episode>/ee_poses`` dataset. They carry the split
observation terms ``data/<episode>/obs/eef_pos`` ``(T, 2, 3)`` and
``data/<episode>/obs/eef_quat`` ``(T, 2, 4)`` instead, plus a real gripper
observation ``data/<episode>/obs/gripper_pos`` ``(T, 2)`` that Batch 1
files lack. The quat order is (w, x, y, z), same as Batch 1: at the shared
fixed rest pose the Batch 2 ``eef_quat`` values match the Batch 1
``ee_poses`` quaternions frame for frame, which would fail if the order
were (x, y, z, w). ``load_ee_poses_hdf5`` therefore composes
``ee_poses`` from the two split terms when the combined dataset is
absent, and validates the result with the same unit-norm check.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

EE_POSES_SHAPE_TAIL = (2, 7)


def validate_ee_poses(ee_poses: np.ndarray) -> np.ndarray:
    """Validate a raw ee_poses array and return it as float64.

    Raises ValueError on any structural problem. Quaternions are checked for
    unit norm within a loose tolerance (recorded data measured exactly 1.0),
    but the returned array is *not* renormalized here; the policy core
    normalizes defensively at load time so this function stays a pure check.
    """
    arr = np.asarray(ee_poses)
    if arr.ndim != 3 or tuple(arr.shape[1:]) != EE_POSES_SHAPE_TAIL:
        raise ValueError(
            f"ee_poses must have shape (T, 2, 7), got {arr.shape}"
        )
    if arr.shape[0] < 1:
        raise ValueError("ee_poses must contain at least one timestep")
    arr = arr.astype(np.float64, copy=False)
    if not np.all(np.isfinite(arr)):
        raise ValueError("ee_poses contains NaN or infinite values")
    norms = np.linalg.norm(arr[:, :, 3:7], axis=-1)
    if np.any(norms < 0.99) or np.any(norms > 1.01):
        raise ValueError(
            "ee_poses quaternion norms deviate from 1.0 by more than 1%; "
            f"observed range [{norms.min():.4f}, {norms.max():.4f}]"
        )
    return arr


def load_ee_poses_npz(path: str | Path) -> np.ndarray:
    """Load ee_poses from an .npz bundle produced by the extraction tool."""
    with np.load(Path(path), allow_pickle=False) as bundle:
        if "ee_poses" not in bundle:
            raise ValueError(f"{path} does not contain an 'ee_poses' array")
        return validate_ee_poses(bundle["ee_poses"])


def load_ee_poses_hdf5(path: str | Path, episode_key: str) -> np.ndarray:
    """Load ee_poses for one episode from a demonstration HDF5 file.

    Batch 1 layout: a single ``data/<episode>/ee_poses`` dataset.
    Batch 2 layout: the split observation terms
    ``data/<episode>/obs/eef_pos`` and ``data/<episode>/obs/eef_quat``,
    composed here into the same ``(T, 2, 7)`` ``[x, y, z, qw, qx, qy, qz]``
    array (quat order (w, x, y, z), verified against Batch 1 at the shared
    rest pose). Either layout is validated identically.
    """
    import h5py  # local import: only the extraction tool needs h5py

    with h5py.File(Path(path), "r") as handle:
        dataset_path = f"data/{episode_key}/ee_poses"
        if dataset_path in handle:
            return validate_ee_poses(handle[dataset_path][:])
        pos_path = f"data/{episode_key}/obs/eef_pos"
        quat_path = f"data/{episode_key}/obs/eef_quat"
        if pos_path in handle and quat_path in handle:
            pos = np.asarray(handle[pos_path][:], dtype=np.float64)
            quat = np.asarray(handle[quat_path][:], dtype=np.float64)
            if pos.shape[0] != quat.shape[0]:
                raise ValueError(
                    f"{path} episode {episode_key}: eef_pos has "
                    f"{pos.shape[0]} steps but eef_quat has {quat.shape[0]}"
                )
            return validate_ee_poses(
                np.concatenate([pos, quat], axis=-1)
            )
        raise ValueError(
            f"{path} does not contain {dataset_path} nor "
            f"{pos_path} + {quat_path}"
        )


def load_gripper_pos_npz(path: str | Path) -> np.ndarray | None:
    """Load gripper_pos from an .npz bundle, or None when it has none.

    Batch 1 bundles carry no gripper channel (returns None); Batch 2
    bundles carry ``gripper_pos`` ``(T, 2)``. Values are the Galbot
    gripper joint positions: 0.0 is fully open, 1.703 fully closed
    (``BinaryJointPositionActionCfg`` open/close_command_expr in the
    official starter repo's Galbot config).
    """
    with np.load(Path(path), allow_pickle=False) as bundle:
        if "gripper_pos" not in bundle:
            return None
        arr = np.asarray(bundle["gripper_pos"], dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(
            f"gripper_pos must have shape (T, 2), got {arr.shape}"
        )
    if arr.shape[0] < 1:
        raise ValueError("gripper_pos must contain at least one timestep")
    if not np.all(np.isfinite(arr)):
        raise ValueError("gripper_pos contains NaN or infinite values")
    return arr


def load_gripper_pos_hdf5(path: str | Path, episode_key: str) -> np.ndarray:
    """Load the Batch 2 gripper observation for one episode, shape (T, 2).

    Batch 1 files carry no gripper channel, so this raises ValueError
    there; Batch 2 files carry ``data/<episode>/obs/gripper_pos``. The
    values are returned as float64 and checked finite only: the policy
    maps them to the starter repo's binary gripper convention
    (joint 0.0 open / 1.703 closed, ``< 0`` close / ``>= 0`` open).
    Recorded range in the two verified Batch 2 episodes is roughly
    0 to 1.7 with 364 distinct rounded values, i.e. a real varying
    signal, not a constant.
    """
    import h5py  # local import: only the extraction tool needs h5py

    with h5py.File(Path(path), "r") as handle:
        dataset_path = f"data/{episode_key}/obs/gripper_pos"
        if dataset_path not in handle:
            raise ValueError(f"{path} does not contain {dataset_path}")
        arr = np.asarray(handle[dataset_path][:], dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(
            f"gripper_pos must have shape (T, 2), got {arr.shape}"
        )
    if arr.shape[0] < 1:
        raise ValueError("gripper_pos must contain at least one timestep")
    if not np.all(np.isfinite(arr)):
        raise ValueError("gripper_pos contains NaN or infinite values")
    return arr


def bundle_metadata(path: str | Path) -> dict:
    """Return the provenance metadata stored in an .npz bundle, if any."""
    with np.load(Path(path), allow_pickle=False) as bundle:
        if "meta_json" in bundle:
            return json.loads(str(bundle["meta_json"]))
    return {}
