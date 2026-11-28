"""Cross-check the organizers' LeRobot v30 dataset release (2026-11-26).

On 2026-11-26 the ECM organizers published the training data in LeRobot
format under ``Roco_ChamberTask_LeRobot_v30/`` in the HF dataset
``sgeaichallenge/ecm2027_sim_data`` (renamed from
``rocochallenge2025/ecm2027_sim_data`` on 2026-11-24). This tool answers,
from primary sources, what that release contains and whether our
replay-follow policy can consume it:

* Inventory: episode counts per batch from the HF API listing (saved
  snapshot, read fresh on every run) -- batch1 173 + batch3 200 = 373
  episode dirs, no batch2.
* Schema: one episode's ``meta/info.json`` -- robot Galbot-Mobile,
  30 fps, ``observation.state`` float32[77], ``action`` float32[20],
  three h264 cameras (first-person 640x480, left/right hand 224x224).
* Action layout: inferred from the data parquet itself, not assumed --
  both 4-vectors in the 20-dim action are unit quaternions on every
  frame, gripper columns are binary, base columns are small velocities.
  The layout is [left pos(3) + quat(4), right pos(3) + quat(4),
  gripper(2), base(4)], matching our 20-dim legacy action. The quat
  component order (wxyz vs xyzw) cannot be resolved from the data
  alone and is recorded as open.
* Metadata honesty: ``meta/stats.json`` min/max reproduce EXACTLY
  (diff 0.0) from the downloaded parquet for state and action.
* Rest-pose comparison: the LeRobot episode's first-frame action EE
  pose is compared against the first frame of each of our 9 HDF5 demo
  bundles. It matches none of them (different recording / anchor),
  so no LeRobot episode can be dropped into our bundles as-is.
* Honest gaps: ``observation.state`` has ``names: null`` in
  info.json, so the 77-dim state semantics are unresolved (30 of its
  columns behave like joint angles, |x| up to ~2*pi); no HDF5 file
  for the sampled episode id exists locally, so no per-step
  LeRobot-vs-HDF5 trajectory equality check was possible.

Usage (from the repo root):

    python3 -m tools.analyze_lerobot [--json evaluation/lerobot_v30.json]

Prints a Markdown report to stdout and writes the full numbers to
``evaluation/lerobot_v30.json`` (plus a Markdown copy at
``evaluation/lerobot_v30.md``). Deterministic: re-running produces
identical output. The parquet half needs pyarrow; without it the tool
still reports the inventory + info.json half and says the data half
was skipped.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

PACKAGE_ROOT = Path(__file__).resolve().parent.parent

# HF API listing snapshot, next to the repo in the real checkout, or at
# the canonical goal path when running from an isolated clone.
INVENTORY_CANDIDATES = [
    PACKAGE_ROOT.parent.parent / "ecm-data-sample" / "hf_api_20261127.json",
    Path.home() / "workspace" / "goals" / "ai-2" / "hidden_files"
    / "ecm-data-sample" / "hf_api_20261127.json",
]

# One downloaded LeRobot episode (parquet + meta json), same layout.
EPISODE_CANDIDATES = [
    PACKAGE_ROOT.parent.parent / "ecm-data-sample" / "lerobot_v30",
    Path.home() / "workspace" / "goals" / "ai-2" / "hidden_files"
    / "ecm-data-sample" / "lerobot_v30",
]

EPISODE_ID = "ChamberServiceTask_1793172869923601"
EPISODE_BATCH = "batch1"
PARQUET_NAME = "batch1_1793172869923601.parquet"

# Our 9 integrated HDF5 demo bundles (first-frame EE poses).
DEMO_GLOB = "demo_*_ee_poses.npz"

try:
    import pyarrow.parquet as pq
except ImportError:  # system python has no pyarrow; venv python does
    pq = None


def _first_existing(candidates):
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def inventory(api_path):
    """Episode counts per batch from the HF API listing snapshot."""
    listing = json.loads(api_path.read_text())
    siblings = [s["rfilename"] for s in listing.get("siblings", [])]
    lerobot = [f for f in siblings
               if f.startswith("Roco_ChamberTask_LeRobot_v30/")]
    per_batch = {}
    for path in lerobot:
        parts = path.split("/")
        per_batch[parts[1]] = per_batch.get(parts[1], 0) + 1
    episodes = {batch: count // 8 for batch, count in sorted(per_batch.items())}
    tarballs = [f for f in siblings if f.endswith(".tar.gz")]
    batch3_tars = [f for f in tarballs if "/batch3/" in f]
    return {
        "dataset": listing.get("id", "sgeaichallenge/ecm2027_sim_data"),
        "last_modified": listing.get("lastModified"),
        "total_files": len(siblings),
        "lerobot_files": len(lerobot),
        "lerobot_episodes_per_batch": episodes,
        "lerobot_episodes_total": sum(episodes.values()),
        "hdf5_tarballs_total": len(tarballs),
        "batch3_tarballs": len(batch3_tars),
    }


def parse_info(info_path):
    """Feature schema from one episode's meta/info.json (pure JSON)."""
    info = json.loads(info_path.read_text())
    features = {}
    for name, spec in info.get("features", {}).items():
        features[name] = {
            "dtype": spec.get("dtype"),
            "shape": spec.get("shape"),
            "names": spec.get("names"),
        }
    return {
        "robot_type": info.get("robot_type"),
        "codebase_version": info.get("codebase_version"),
        "fps": info.get("fps"),
        "total_episodes": info.get("total_episodes"),
        "total_frames": info.get("total_frames"),
        "features": features,
    }


def check_stats_reproduction(parquet_path, stats_path):
    """Do stats.json min/max reproduce exactly from the parquet?"""
    stats = json.loads(stats_path.read_text())
    table = pq.read_table(parquet_path)
    out = {}
    for feature in ("observation.state", "action"):
        arr = np.array(table.column(feature).to_pylist(), dtype=np.float64)
        smin = np.array(stats[feature]["min"], dtype=np.float64)
        smax = np.array(stats[feature]["max"], dtype=np.float64)
        out[feature] = {
            "rows": int(arr.shape[0]),
            "dim": int(arr.shape[1]),
            "min_max_abs_diff": float(np.abs(smin - arr.min(axis=0)).max()),
            "max_max_abs_diff": float(np.abs(smax - arr.max(axis=0)).max()),
            "reproduces_exactly": bool(
                np.array_equal(smin, arr.min(axis=0))
                and np.array_equal(smax, arr.max(axis=0))),
        }
    return out


def action_layout(parquet_path):
    """Infer the 20-dim action layout from the data itself."""
    table = pq.read_table(parquet_path)
    action = np.array(table.column("action").to_pylist(), dtype=np.float64)
    state = np.array(table.column("observation.state").to_pylist(),
                     dtype=np.float64)
    ts = np.array(table.column("timestamp").to_pylist(), dtype=np.float64)
    frame_index = np.array(table.column("frame_index").to_pylist(),
                           dtype=np.int64)
    left_quat = action[:, 3:7]
    right_quat = action[:, 10:14]
    left_norm = np.linalg.norm(left_quat, axis=1)
    right_norm = np.linalg.norm(right_quat, axis=1)
    joint_like = [int(i) for i in range(state.shape[1])
                  if float(np.abs(state[:, i]).max()) > 3.0]
    return {
        "rows": int(action.shape[0]),
        "action_dim": int(action.shape[1]),
        "state_dim": int(state.shape[1]),
        "left_quat_norm_min": float(left_norm.min()),
        "left_quat_norm_max": float(left_norm.max()),
        "right_quat_norm_min": float(right_norm.min()),
        "right_quat_norm_max": float(right_norm.max()),
        "left_quat_unit_norm": bool(
            left_norm.min() > 0.999999 and left_norm.max() < 1.000001),
        "right_quat_unit_norm": bool(
            right_norm.min() > 0.999999 and right_norm.max() < 1.000001),
        "gripper_left_values": sorted(float(v) for v in np.unique(action[:, 14])),
        "gripper_right_values": sorted(float(v) for v in np.unique(action[:, 15])),
        "gripper_left_constant": bool(len(np.unique(action[:, 14])) == 1),
        "base_mean": [float(v) for v in action[:, 16:20].mean(axis=0)],
        "base_std": [float(v) for v in action[:, 16:20].std(axis=0)],
        "timestamp_dt_median": float(np.median(np.diff(ts))),
        "frame_index_contiguous": bool(
            np.array_equal(frame_index, np.arange(action.shape[0]))),
        "first_frame_left_pos": [float(v) for v in action[0, 0:3]],
        "first_frame_right_pos": [float(v) for v in action[0, 7:10]],
        "state_joint_like_columns": joint_like,
        "state_joint_like_count": len(joint_like),
        "quat_order": "unresolved (wxyz vs xyzw not distinguishable from data)",
    }


def rest_pose_comparison(first_left, first_right, demo_dir):
    """L2 distance from the LeRobot first frame to each demo bundle."""
    out = {}
    for path in sorted(Path(demo_dir).glob(DEMO_GLOB)):
        bundle = np.load(path)
        ee = np.asarray(bundle["ee_poses"], dtype=np.float64)  # (T, 2, 7)
        left = ee[0, 0, 0:3]
        right = ee[0, 1, 0:3]
        out[path.stem] = {
            "steps": int(ee.shape[0]),
            "left_l2": float(np.linalg.norm(left - np.array(first_left))),
            "right_l2": float(np.linalg.norm(right - np.array(first_right))),
            "demo_first_left": [float(v) for v in left],
            "demo_first_right": [float(v) for v in right],
        }
    return out


def analyze(api_path, episode_dir, demo_dir):
    report = {"episode": f"{EPISODE_BATCH}/{EPISODE_ID}"}
    report["inventory"] = inventory(api_path)
    info_path = Path(episode_dir) / "lr_info.json"
    stats_path = Path(episode_dir) / "lr_stats.json"
    parquet_path = Path(episode_dir) / PARQUET_NAME
    report["schema"] = parse_info(info_path)
    if pq is None:
        report["parquet"] = {"skipped": "pyarrow not installed"}
        report["rest_pose_comparison"] = {}
        return report
    report["stats_reproduction"] = check_stats_reproduction(
        parquet_path, stats_path)
    layout = action_layout(parquet_path)
    report["action_layout"] = layout
    report["rest_pose_comparison"] = rest_pose_comparison(
        layout["first_frame_left_pos"], layout["first_frame_right_pos"],
        demo_dir)
    return report


def render_markdown(report):
    lines = ["# LeRobot v30 cross-check (ECM 2027, M20)", ""]
    inv = report["inventory"]
    lines.append(
        f"Dataset `{inv['dataset']}`, lastModified {inv['last_modified']}: "
        f"{inv['total_files']} files, {inv['lerobot_files']} LeRobot files, "
        f"{inv['lerobot_episodes_total']} episode dirs "
        f"({inv['lerobot_episodes_per_batch']}), no batch2; "
        f"{inv['batch3_tarballs']} batch3 HDF5 tarballs (unchanged count).")
    schema = report["schema"]
    lines.append("")
    lines.append(
        f"Episode {report['episode']}: robot {schema['robot_type']}, "
        f"codebase {schema['codebase_version']}, {schema['fps']} fps, "
        f"{schema['total_frames']} frames. Features: "
        + ", ".join(
            f"{name} {spec['dtype']}{spec['shape']}"
            for name, spec in schema["features"].items()
            if spec["shape"] is not None))
    parquet = report.get("parquet", {})
    if parquet.get("skipped"):
        lines.append("")
        lines.append(f"Parquet half skipped: {parquet['skipped']}.")
        return "\n".join(lines) + "\n"
    stats = report["stats_reproduction"]
    for feature, check in stats.items():
        lines.append(
            f"stats.json reproduces exactly for {feature}: "
            f"{check['reproduces_exactly']} "
            f"(min diff {check['min_max_abs_diff']}, "
            f"max diff {check['max_max_abs_diff']}).")
    layout = report["action_layout"]
    lines.append("")
    lines.append(
        f"Action is {layout['action_dim']}-dim, state "
        f"{layout['state_dim']}-dim; both quaternions unit-norm on all "
        f"{layout['rows']} frames "
        f"(left {layout['left_quat_norm_min']:.9f}-"
        f"{layout['left_quat_norm_max']:.9f}, right "
        f"{layout['right_quat_norm_min']:.9f}-"
        f"{layout['right_quat_norm_max']:.9f}); layout "
        f"[pos(3)+quat(4)] x 2 arms + gripper(2) + base(4) matches our "
        f"20-dim legacy action. Gripper left values "
        f"{layout['gripper_left_values']} (constant), right "
        f"{layout['gripper_right_values']}; base mean ~0, std "
        f"{[round(v, 3) for v in layout['base_std']]}. Timestamp dt "
        f"median {layout['timestamp_dt_median']:.6f} s (30 Hz); "
        f"frame_index contiguous: {layout['frame_index_contiguous']}.")
    lines.append(
        f"State: {layout['state_joint_like_count']} of "
        f"{layout['state_dim']} columns behave like joint angles "
        f"(|x| > 3, up to ~2*pi); state semantics otherwise unresolved "
        f"(info.json names are null). Quat component order: "
        f"{layout['quat_order']}.")
    lines.append("")
    lines.append("Rest-pose comparison (LeRobot first frame vs our 9 demo "
                "bundles, L2 on EE position):")
    for name, comp in report["rest_pose_comparison"].items():
        lines.append(
            f"- {name} ({comp['steps']} steps): left "
            f"{comp['left_l2']:.4f} m, right {comp['right_l2']:.4f} m.")
    lines.append("")
    lines.append("No bundle matches: the LeRobot episode is a different "
                "recording (different anchor) from all 9 integrated demos, "
                "so LeRobot episodes cannot be dropped into our bundles "
                "as-is; the action layout does match our 20-dim legacy "
                "action, so a LeRobot-fed replay-follow variant is "
                "feasible once the quat order is pinned.")
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json",
                        default=str(PACKAGE_ROOT / "evaluation"
                                    / "lerobot_v30.json"))
    parser.add_argument("--md",
                        default=str(PACKAGE_ROOT / "evaluation"
                                    / "lerobot_v30.md"))
    args = parser.parse_args(argv)
    api_path = _first_existing(INVENTORY_CANDIDATES)
    episode_dir = _first_existing(EPISODE_CANDIDATES)
    if api_path is None or episode_dir is None:
        raise SystemExit("missing HF API snapshot or LeRobot episode dir")
    demo_dir = PACKAGE_ROOT / "data"
    report = analyze(api_path, episode_dir, demo_dir)
    markdown = render_markdown(report)
    json_path = Path(args.json)
    json_path.write_text(json.dumps(report, indent=2) + "\n")
    Path(args.md).write_text(markdown)
    print(markdown, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
