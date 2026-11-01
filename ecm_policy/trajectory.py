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
    """Load ee_poses for one episode from a demonstration HDF5 file."""
    import h5py  # local import: only the extraction tool needs h5py

    with h5py.File(Path(path), "r") as handle:
        dataset_path = f"data/{episode_key}/ee_poses"
        if dataset_path not in handle:
            raise ValueError(f"{path} does not contain {dataset_path}")
        return validate_ee_poses(handle[dataset_path][:])


def bundle_metadata(path: str | Path) -> dict:
    """Return the provenance metadata stored in an .npz bundle, if any."""
    with np.load(Path(path), allow_pickle=False) as bundle:
        if "meta_json" in bundle:
            return json.loads(str(bundle["meta_json"]))
    return {}
