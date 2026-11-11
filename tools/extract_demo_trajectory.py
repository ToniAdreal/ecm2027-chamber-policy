#!/usr/bin/env python3
"""Extract one episode's ee_poses from a demonstration HDF5 into an .npz bundle.

The bundle stores the validated ``ee_poses`` array plus a ``meta_json``
provenance record. Run with a Python that has h5py + numpy (the ECM data
sample venv at ``hidden_files/ecm-data-sample/.venv`` works):

    python tools/extract_demo_trajectory.py \
        --input /path/to/replay_state_ee_poses.hdf5 \
        --episode demo_7 \
        --output data/demo_7_ee_poses.npz
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ecm_policy.trajectory import load_ee_poses_hdf5, load_gripper_pos_hdf5  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="source demonstration HDF5")
    parser.add_argument("--episode", default="demo_7", help="episode key under data/")
    parser.add_argument("--output", required=True, help="destination .npz")
    parser.add_argument(
        "--source-note",
        default="Sample episode from HF dataset rocochallenge2025/ecm2027_sim_data (Batch 1)",
        help="provenance note stored in the bundle metadata",
    )
    args = parser.parse_args()

    ee_poses = load_ee_poses_hdf5(args.input, args.episode)
    try:
        gripper_pos = load_gripper_pos_hdf5(args.input, args.episode)
    except ValueError:
        gripper_pos = None  # Batch 1 files carry no gripper channel
    meta = {
        "source_file": Path(args.input).name,
        "episode_key": args.episode,
        "shape": list(ee_poses.shape),
        "dtype": "float32 (stored), validated as float64",
        "layout": "[x, y, z, qw, qx, qy, qz] per arm; arm 0 = left, arm 1 = right",
        "source_note": args.source_note,
        "gripper_pos_included": gripper_pos is not None,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    arrays = {
        "ee_poses": ee_poses.astype(np.float32),
        "meta_json": np.array(json.dumps(meta)),
    }
    if gripper_pos is not None:
        arrays["gripper_pos"] = gripper_pos.astype(np.float32)
    np.savez(out, **arrays)
    print(
        f"wrote {out}: ee_poses {ee_poses.shape} from {args.input} "
        f"episode {args.episode}"
        + (" + gripper_pos " + str(gripper_pos.shape) if gripper_pos is not None else "")
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
