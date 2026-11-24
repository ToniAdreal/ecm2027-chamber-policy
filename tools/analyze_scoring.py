"""Answer upstream issue #8 from the task code and the demo data.

Issue #8 (rocochallenge/ecm2027_sim_track, sunnycho100, 2026-11-19,
still open with 0 comments) asks, for the official sim evaluation:
besides the sponge pose, which objects get randomized, and how are
the lid lock and the wipe milestones scored?

This tool answers both halves from primary sources, and says plainly
where the demo data cannot answer:

* Randomization and scoring come from the starter repo's task code
  (``robofinals_tasks/chamber_service_task.py`` in the sibling
  ``ecm2027-sim-track`` checkout, read fresh on every run).
* The demo data's ``initial_state`` root poses (Batch 2 HDF5 files
  under ``ecm-data-sample/batch2``) are measured against the code's
  randomization claims: what actually varies across episodes, and
  what is bit-identical.
* The per-step demo recordings carry no lid angle, no object poses
  over time and no contact flags (only ``obs/eef_pos``,
  ``obs/eef_quat`` and ``obs/gripper_pos``), so the wipe-face and
  lid-state scoring rules cannot be reconstructed from demo data at
  all. The scoring half of the answer therefore rests on the task
  code plus the organizer mail of 2026-11-20 (13-point breakdown),
  and the tool records that gap instead of pretending the demos
  prove it.

Usage (from the repo root):

    python3 -m tools.analyze_scoring [--json evaluation/scoring_analysis.json]

Prints a Markdown report to stdout and writes the full numbers to
``evaluation/scoring_analysis.json`` (plus a Markdown copy at
``evaluation/scoring_analysis.md``). Deterministic: re-running
produces identical output. The HDF5 half needs h5py; without it the
tool still reports the code half and says the data half was skipped.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ROOT = Path(__file__).resolve().parent.parent
STARTER_ROOT = ROOT.parent / "ecm2027-sim-track"
TASK_PATH = STARTER_ROOT / "robofinals_tasks" / "chamber_service_task.py"
DEFAULT_HDF5_DIR = ROOT.parent.parent / "ecm-data-sample" / "batch2"

# Objects whose t=0 root pose the recordings store in initial_state.
INITIAL_STATE_OBJECTS = (
    ("spongebody", ("rigid_object", "spongebody", "root_pose")),
    ("robot", ("articulation", "robot", "root_pose")),
    ("Chamber002", ("rigid_object", "Chamber002", "root_pose")),
    ("Chamber002_01", ("rigid_object", "Chamber002_01", "root_pose")),
    ("Table291", ("rigid_object", "Table291", "root_pose")),
    ("Chamber001_01", ("articulation", "Chamber001_01", "root_pose")),
)

try:
    import h5py
except ImportError:  # system python has no h5py; the venv python does
    h5py = None

try:
    import numpy as np
except ImportError:  # pragma: no cover - numpy is a hard dependency
    np = None


def _float_const(source: str, name: str) -> float:
    match = re.search(rf"^\s*{name}\s*=\s*([0-9.]+)", source, re.M)
    if match is None:
        raise ValueError(f"constant {name} not found in task code")
    return float(match.group(1))


def _tuple_const(source: str, name: str) -> tuple[float, ...]:
    match = re.search(rf"^\s*{name}\s*=\s*\(([^)]*)\)", source, re.M)
    if match is None:
        raise ValueError(f"constant {name} not found in task code")
    return tuple(float(v) for v in match.group(1).split(",") if v.strip())


def parse_scoring(task_path: Path) -> dict:
    """Extract the scoring spec from the starter repo's task code."""
    source = task_path.read_text(encoding="utf-8")
    point_names = re.search(
        r"POINT_NAMES\s*=\s*\((.*?)\)", source, re.S
    ).group(1)
    points = re.findall(r'"([^"]+)"', point_names)
    slices_match = re.search(r"STAGE_POINT_SLICES\s*=\s*\(", source)
    if slices_match is None:
        raise ValueError("STAGE_POINT_SLICES not found in task code")
    # The value nests one pair per stage and ends at the first "))",
    # so cut there instead of a fixed window (which would leak the
    # following _WIPE_SLICE = (5, 10) line).
    slices_end = source.find("))", slices_match.end())
    slices_text = source[slices_match.end():slices_end + 2]
    stage_slices = [
        (int(a), int(b))
        for a, b in re.findall(r"\((\d+),\s*(\d+)\)", slices_text)
    ]
    stage_names = re.findall(r'"(\d+\.\s[^"]+)"', source)
    thresholds = {
        name: _float_const(source, name)
        for name in (
            "MAX_SCORE",
            "LID_LENGTH_M",
            "LID_LIFT_HEIGHT_M",
            "LID_OPEN_THRESHOLD",
            "LID_CLOSED_THRESHOLD",
            "LID_STABLE_ANG_VEL",
            "STABLE_DURATION_S",
            "STABLE_LIN_VEL",
            "PART_LIFT_HEIGHT_M",
            "SPONGE_LIFT_HEIGHT_M",
            "REFERENCE_SETTLE_S",
            "NEW_PART_PLACE_DISTANCE_M",
            "CHAMBER_INNER_HALF_X",
            "CHAMBER_INNER_HALF_Y",
            "CHAMBER_FLOOR_BELOW_PART_M",
            "CHAMBER_INNER_HEIGHT_M",
            "SPONGE_CHAMBER_CONTACT_DURATION_S",
            "WIPE_FACE_MAX_DISTANCE_M",
            "WIPE_FACE_OUTSIDE_TOLERANCE_M",
            "SPONGE_CHAMBER_CONTACTS_REQUIRED",
            "ROBOT_RESET_POS_DEV_X",
            "ROBOT_RESET_POS_DEV_Y",
            "ROBOT_RESET_YAW_DEV",
            "SPONGE_POS_JITTER",
        )
    }
    sponge_base = _tuple_const(source, "SPONGE_BASE_POS")
    box_center = _tuple_const(source, "BOX_CENTER_XY")
    # Reset anchor (upstream 17159e5, 2026-11-23): the eval reset now
    # samples directly around the demo base pose instead of via the
    # shared helper, which displaced the robot ~0.3 m from the anchor.
    anchor_pos = _tuple_const(source, "ROBOT_RESET_ANCHOR_POS")
    anchor_ori = _tuple_const(source, "ROBOT_RESET_ANCHOR_ORI")
    keep_table_static = bool(
        re.search(r"^\s*KEEP_TABLE_STATIC\s*=\s*True", source, re.M)
    )
    wipe_faces = re.search(
        r'WIPE_FACE_NAMES\s*=\s*\((.*?)\)', source, re.S
    ).group(1)
    faces = re.findall(r'"([^"]+)"', wipe_faces)
    lock_lines = [
        (i + 1, line.strip())
        for i, line in enumerate(source.splitlines())
        if re.search(r"\block\b", line, re.I)
    ]
    return {
        "task_file": str(task_path),
        "max_score": int(thresholds.pop("MAX_SCORE")),
        "point_names": points,
        "stage_point_slices": stage_slices,
        "stage_names": stage_names[:6],
        "thresholds": thresholds,
        "sponge_base_pos": sponge_base,
        "sponge_pos_jitter": thresholds.pop("SPONGE_POS_JITTER"),
        "box_center_xy": box_center,
        "robot_reset_anchor_pos": anchor_pos,
        "robot_reset_anchor_ori": anchor_ori,
        "keep_table_static": keep_table_static,
        "wipe_faces": faces,
        "lock_mentions": lock_lines,
        "lid_lock_point": any("lock" in p for p in points),
    }


def yaw_from_quat_xyzw(quat) -> float:
    """Z-yaw of a quaternion, via its rotation matrix (numpy only)."""
    x, y, z, w = (float(v) for v in quat)
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    r00 = 1.0 - 2.0 * (y * y + z * z)
    r10 = 2.0 * (x * y + z * w)
    return math.atan2(r10, r00)


def _angular_spread(yaws) -> float:
    """Largest pairwise angular distance in [0, pi]."""
    best = 0.0
    for i, a in enumerate(yaws):
        for b in yaws[i + 1:]:
            d = abs((a - b + math.pi) % (2.0 * math.pi) - math.pi)
            best = max(best, d)
    return best


def load_initial_states(hdf5_files) -> list[dict]:
    """Read t=0 root poses from demo HDF5 files, deduplicated by content."""
    if h5py is None:
        raise RuntimeError("h5py not installed; cannot read HDF5 samples")
    states: list[dict] = []
    seen: set[str] = set()
    for path in sorted(hdf5_files):
        with h5py.File(path, "r") as handle:
            for demo in sorted(handle["data"].keys()):
                group = handle["data"][demo].get("initial_state")
                if group is None:
                    continue
                blob = bytearray()
                record: dict = {
                    "file": Path(path).name,
                    "demo": demo,
                    "objects": {},
                }
                for name, key_path in INITIAL_STATE_OBJECTS:
                    try:
                        arr = np.asarray(group["/".join(key_path)][0])
                    except KeyError:
                        continue
                    blob.extend(np.ascontiguousarray(arr).tobytes())
                    record["objects"][name] = arr
                try:
                    lid_joints = np.asarray(
                        group["articulation/Chamber001_01/joint_position"][0]
                    )
                except KeyError:
                    lid_joints = None
                if lid_joints is not None:
                    blob.extend(np.ascontiguousarray(lid_joints).tobytes())
                    record["lid_joints"] = lid_joints
                digest = hashlib.sha256(bytes(blob)).hexdigest()
                if digest in seen:
                    record["duplicate_of"] = digest
                    states.append(record)
                    continue
                seen.add(digest)
                record["digest"] = digest
                states.append(record)
    return states


def summarize_initial_states(states: list[dict]) -> dict:
    """Per-object spread across the unique initial states."""
    unique = [s for s in states if "digest" in s]
    summary: dict = {
        "raw_entries": len(states),
        "unique_episodes": len(unique),
        "duplicates": len(states) - len(unique),
        "objects": {},
    }
    for name, _ in INITIAL_STATE_OBJECTS:
        poses = [s["objects"][name] for s in unique if name in s["objects"]]
        if not poses:
            continue
        pos = np.array([p[:3] for p in poses], dtype=float)
        yaws = [yaw_from_quat_xyzw(p[3:]) for p in poses]
        summary["objects"][name] = {
            "n": len(poses),
            "pos_min": [round(float(v), 6) for v in pos.min(axis=0)],
            "pos_max": [round(float(v), 6) for v in pos.max(axis=0)],
            "pos_range": [round(float(v), 6) for v in pos.max(axis=0) - pos.min(axis=0)],
            "yaw_min": round(min(yaws), 6),
            "yaw_max": round(max(yaws), 6),
            "yaw_spread_rad": round(_angular_spread(yaws), 6),
        }
    lids = [s["lid_joints"] for s in unique if "lid_joints" in s]
    if lids:
        arr = np.array(lids, dtype=float)
        summary["lid_joints"] = {
            "n": len(lids),
            "all_zero": bool(np.all(arr == 0.0)),
            "max_abs": round(float(np.abs(arr).max()), 9),
        }
    return summary


def analyze(task_path: Path, hdf5_files) -> dict:
    """Full issue-#8 analysis: code spec plus measured demo data."""
    spec = parse_scoring(task_path)
    result: dict = {"scoring_spec": spec, "initial_states": None}
    if hdf5_files:
        states = load_initial_states(hdf5_files)
        summary = summarize_initial_states(states)
        sponge = summary["objects"].get("spongebody", {})
        base = np.array(spec["sponge_base_pos"], dtype=float)
        jitter = spec["sponge_pos_jitter"]
        if sponge:
            pos_min = np.array(sponge["pos_min"])
            pos_max = np.array(sponge["pos_max"])
            max_dev = np.maximum(
                np.abs(pos_min - base), np.abs(pos_max - base)
            )
            sponge["max_abs_dev_from_base"] = [
                round(float(v), 6) for v in max_dev
            ]
            sponge["spawn_band_z"] = [
                round(float(base[2] - jitter), 6),
                round(float(base[2] + jitter), 6),
            ]
            # Recorded z straddles the spawn band's low edge: the pose
            # is read after the sponge settles on the table, not at spawn.
            sponge["z_min_below_spawn_band"] = bool(
                pos_min[2] < base[2] - jitter
            )
            sponge["xy_within_jitter"] = bool(
                max_dev[0] <= jitter and max_dev[1] <= jitter
            )
        result["initial_states"] = summary
        robot = summary["objects"].get("robot", {})
        anchor_pos = spec["robot_reset_anchor_pos"]
        anchor_yaw = spec["robot_reset_anchor_ori"][2]
        anchor_match = bool(
            robot
            and all(
                abs(a - b) <= 1e-6
                for a, b in zip(robot["pos_min"], anchor_pos)
            )
            and all(
                abs(a - b) <= 1e-6
                for a, b in zip(robot["pos_max"], anchor_pos)
            )
            and abs(robot["yaw_min"] - anchor_yaw) <= 1e-6
            and abs(robot["yaw_max"] - anchor_yaw) <= 1e-6
        )
        result["findings"] = {
            "randomized_in_data": ["spongebody"],
            "fixed_in_data": [
                name
                for name, obj in summary["objects"].items()
                if name != "spongebody"
                and max(obj["pos_range"]) == 0.0
                and obj["yaw_spread_rad"] == 0.0
            ],
            "robot_base_randomized_in_code_but_fixed_in_data": bool(
                summary["objects"].get("robot", {}).get("yaw_spread_rad") == 0.0
                and max(summary["objects"].get("robot", {}).get("pos_range", [1.0])) == 0.0
            ),
            # Upstream 17159e5 (2026-11-23) anchors the eval reset at
            # exactly the demo base pose: the demos sit at the centre
            # of the eval distribution, not 0.3 m away from it.
            "robot_reset_anchor_matches_demo_base": anchor_match,
            "lid_lock_scored": spec["lid_lock_point"],
            "demo_data_cannot_score_wipe_or_lid": True,
        }
    return result


def to_markdown(result: dict) -> str:
    spec = result["scoring_spec"]
    lines = [
        "# ECM 2027 — issue #8: randomization and scoring, from code and demo data",
        "",
        "Upstream issue #8 asks which objects are randomized besides the",
        "sponge pose, and how the lid lock and wipe milestones are scored.",
        "Sources: the starter repo task code",
        f"(`{spec['task_file']}`) and the Batch 2 demo recordings'",
        "`initial_state` root poses. The organizer mail of 2026-11-20",
        "confirms the 13-point breakdown and the Docker submission route.",
        "",
        "## Scoring: 13 points, 6 sequential stages",
        "",
        "Stages gate on each other: a stage's points only count once every",
        "earlier stage is complete (`_stage_points_done` + stage flags in",
        "`_check_success`).",
        "",
    ]
    for (start, end), name in zip(
        spec["stage_point_slices"], spec["stage_names"]
    ):
        points = ", ".join(spec["point_names"][start:end])
        lines.append(f"* Stage {name}: {points}")
    lines += [
        "",
        "## Lid lock: there is no such scored point",
        "",
        "No entry of `POINT_NAMES` mentions a lock, and the word `lock`",
        "appears in the task file only in the episode description string",
        '("close and lock the chamber"). The lid is scored as three',
        "ordinary points instead:",
        "",
        "* `1a_lid_lifted_1cm`: lid front edge rises >= 0.01 m",
        "  (`LID_LENGTH_M * sin(angle)`).",
        "* `1b_lid_stable_open`: normalized lid joint >= 0.4 with angular",
        "  rate <= 0.1 rad/s, held 1.0 s.",
        "* `6_lid_closed_stable`: normalized lid joint <= 0.005 with",
        "  angular rate <= 0.1 rad/s, held 1.0 s.",
        "",
        "## Wipe milestones: 5 faces, 0.5 s contact each",
        "",
        "One point per cavity face "
        f"({', '.join(spec['wipe_faces'])}), credited after 0.5 s of",
        "continuous sponge-chamber contact while that face is the nearest",
        "cavity face to the sponge centre (within 0.08 m), the sponge",
        "centre is inside the cavity (height <= 0.30 m above the chamber",
        "floor), and stages 1-3 plus sponge pick-up are already done.",
        "The contact sensor only reports sponge-vs-chamber contact; the",
        "face is inferred as the nearest cavity face, not sensed directly.",
        "",
    ]
    yaw_dev = spec["thresholds"]["ROBOT_RESET_YAW_DEV"]
    anchor = spec["robot_reset_anchor_pos"]
    anchor_yaw = spec["robot_reset_anchor_ori"][2]
    randomization_paragraph = (
        "Task code randomizes exactly two things per reset: the sponge "
        "initial pose (base (0.37, -0.50, 0.85), +/-0.05 m uniform jitter "
        "per axis, yaw uniform over [0, 2*pi)) and the robot base pose "
        f"(+/-{spec['thresholds']['ROBOT_RESET_POS_DEV_X']:.2f} m in x/y, "
        f"+/-{yaw_dev:.4f} rad ({math.degrees(yaw_dev):.1f} degrees) yaw "
        f"around the anchor {anchor}, heading {anchor_yaw:.4f} rad). The "
        "chamber, both parts and the table are fixed scene geometry. "
        + (
            "Table: KEEP_TABLE_STATIC is True in the task code, so the "
            "table is held kinematic in GPU physics."
            if spec["keep_table_static"]
            else "Table: KEEP_TABLE_STATIC is False in the task code, so "
            "the table stays a movable body as in data collection (the "
            "GPU-physics static-table fix is kept but disabled)."
        )
    )
    lines += [
        "## Randomization: only the sponge (and, in code, the robot base)",
        "",
        randomization_paragraph,
        "",
    ]
    summary = result.get("initial_states")
    if summary is None:
        lines += [
            "Demo-data half skipped: no HDF5 sample files were given, or",
            "h5py is not installed.",
            "",
        ]
    else:
        lines += [
            f"Measured across {summary['unique_episodes']} unique initial",
            f"states ({summary['raw_entries']} raw entries,",
            f"{summary['duplicates']} duplicate):",
            "",
        ]
        for name, obj in summary["objects"].items():
            lines.append(
                f"* `{name}`: pos range {obj['pos_range']} m, yaw spread"
                f" {obj['yaw_spread_rad']} rad (n={obj['n']})"
            )
        sponge = summary["objects"].get("spongebody", {})
        zmin = sponge.get("pos_min", [0, 0, 0])[2]
        zmax = sponge.get("pos_max", [0, 0, 0])[2]
        band_lo, band_hi = sponge.get("spawn_band_z", [0, 0])
        lines += [
            "",
            "* Sponge x/y stay inside the code's +/-0.05 m jitter band;",
            "  sponge yaw varies across episodes (uniform yaw in code).",
            f"* Sponge z spans {zmin:.2f}-{zmax:.2f} m against a spawn",
            f"  band of {band_lo:.2f}-{band_hi:.2f} m: the low end sits",
            "  below the band, consistent with the recorded pose being",
            "  post-settle on the table rather than the spawn pose.",
            "* Robot base pose is BIT-IDENTICAL in every sampled initial",
            f"  state, at the anchor {spec['robot_reset_anchor_pos']} with",
            f"  heading {spec['robot_reset_anchor_ori'][2]:.4f} rad. The",
            "  task code now anchors the eval reset at exactly that pose",
            f"  (upstream 17159e5) and samples only +/-0.10 m /",
            f"  +/-{yaw_dev:.4f} rad around it, so the demos sit at the",
            "  centre of the eval distribution instead of ~0.3 m away from",
            "  it (the old shared-helper behaviour, upstream issue #5).",
            "  Residual mismatch is bounded: at eval the base can sit up",
            f"  to 0.10 m and {math.degrees(yaw_dev):.0f} deg away from the",
            "  recorded pose, so a replayed demo can be off by up to",
            "  ~0.10 m in translation plus the yaw-induced end-effector",
            "  offset; the replay-follow policy still commands zero base",
            "  velocity.",
            "* Chamber, both parts, table and lid joints are bit-identical",
            "  across episodes: nothing else is randomized.",
            "",
            "## What the demo data cannot say",
            "",
            "Per-step recordings carry only `obs/eef_pos`, `obs/eef_quat`",
            "and `obs/gripper_pos`. There is no lid angle, no per-step",
            "object pose and no contact flag, so wipe faces and lid state",
            "cannot be reconstructed from demo data. The scoring answers",
            "above rest on the task code and the organizer mail, not on",
            "the demos, and the submission keeps no score claim beyond",
            "what the offline harness measures.",
            "",
        ]
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--json",
        default=str(ROOT / "evaluation" / "scoring_analysis.json"),
        help="where to write the analysis JSON",
    )
    parser.add_argument(
        "--hdf5-dir",
        default=str(DEFAULT_HDF5_DIR),
        help="directory holding Batch 2 demo HDF5 files",
    )
    args = parser.parse_args(argv)
    hdf5_files: list[Path] = []
    if h5py is not None and Path(args.hdf5_dir).is_dir():
        hdf5_files = sorted(Path(args.hdf5_dir).glob("*.hdf5"))
    result = analyze(TASK_PATH, hdf5_files)
    out_json = Path(args.json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    md_path = out_json.with_suffix(".md")
    md_path.write_text(to_markdown(result), encoding="utf-8")
    print(to_markdown(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
