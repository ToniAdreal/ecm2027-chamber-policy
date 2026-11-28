"""LeRobot v30 cross-check: inventory, schema, action layout, stats.

The organizers published the ECM training data in LeRobot format on
2026-11-26 (``Roco_ChamberTask_LeRobot_v30/`` in
``sgeaichallenge/ecm2027_sim_data``). These tests pin what the
analysis tool extracts: the episode inventory per batch, the feature
schema from ``meta/info.json``, the 20-dim action layout inferred
from the data itself (unit quaternions, binary grippers, small base
velocities), the exact reproduction of ``meta/stats.json`` from the
parquet, and the finding that the sampled LeRobot episode matches
none of our 9 HDF5 demo bundles at the rest pose.

The fixture half builds a tiny synthetic LeRobot-like episode in a
temp dir and runs anywhere pyarrow is installed. The real-data half
needs pyarrow plus the downloaded episode in the goal workspace, and
is skipped otherwise (system python has neither).
"""

import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

PACKAGE_ROOT = Path(__file__).resolve().parent.parent

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError:  # system python has no pyarrow; venv python does
    pa = None
    pq = None

from tools.analyze_lerobot import (  # noqa: E402
    action_layout,
    analyze,
    check_stats_reproduction,
    inventory,
    parse_info,
    render_markdown,
    rest_pose_comparison,
)

# Downloaded episode: next to the repo in the real checkout, or at the
# canonical goal path when running from an isolated clone.
EPISODE_CANDIDATES = [
    PACKAGE_ROOT.parent.parent / "ecm-data-sample" / "lerobot_v30",
    Path.home() / "workspace" / "goals" / "ai-2" / "hidden_files"
    / "ecm-data-sample" / "lerobot_v30",
]
INVENTORY_CANDIDATES = [
    PACKAGE_ROOT.parent.parent / "ecm-data-sample" / "hf_api_20261127.json",
    Path.home() / "workspace" / "goals" / "ai-2" / "hidden_files"
    / "ecm-data-sample" / "hf_api_20261127.json",
]


def _first_existing(candidates):
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _synthetic_info(directory):
    """An info.json-only fixture: no parquet, runs everywhere."""
    directory = Path(directory)
    info = {
        "codebase_version": "v3.0",
        "robot_type": "Galbot-Mobile",
        "total_episodes": 1,
        "total_frames": 5,
        "fps": 30,
        "features": {
            "observation.state": {"dtype": "float32", "shape": [3],
                                  "names": None},
            "action": {"dtype": "float32", "shape": [20], "names": None},
        },
    }
    (directory / "lr_info.json").write_text(json.dumps(info))
    return directory


def _synthetic_episode(directory):
    """A 5-frame LeRobot-like episode with a known action layout."""
    directory = Path(directory)
    _synthetic_info(directory)
    state = np.zeros((5, 3), dtype=np.float32)
    state[:, 0] = [0.0, 6.28, -6.28, 0.5, -0.5]  # joint-like column
    action = np.zeros((5, 20), dtype=np.float32)
    # Left arm: pos (0.1, 0.2, 0.9), quat (1, 0, 0, 0) -- unit norm.
    action[:, 0:3] = [0.1, 0.2, 0.9]
    action[:, 3] = 1.0
    # Right arm: pos (0.3, -0.2, 0.9), quat (0, 0, 0, 1) -- unit norm.
    action[:, 7:10] = [0.3, -0.2, 0.9]
    action[:, 13] = 1.0
    action[:, 14] = 1.0   # left gripper constant
    action[:, 15] = [1.0, -1.0, 1.0, -1.0, 1.0]  # right gripper binary
    stats = {
        "observation.state": {
            "min": state.min(axis=0).tolist(),
            "max": state.max(axis=0).tolist(),
        },
        "action": {
            "min": action.min(axis=0).tolist(),
            "max": action.max(axis=0).tolist(),
        },
    }
    (directory / "lr_stats.json").write_text(json.dumps(stats))
    table = pa.table({
        "observation.state": pa.array(
            [row.tolist() for row in state],
            type=pa.list_(pa.float32(), 3)),
        "action": pa.array(
            [row.tolist() for row in action],
            type=pa.list_(pa.float32(), 20)),
        "timestamp": pa.array([i / 30.0 for i in range(5)],
                              type=pa.float32()),
        "frame_index": pa.array(list(range(5)), type=pa.int64()),
        "episode_index": pa.array([0] * 5, type=pa.int64()),
        "index": pa.array(list(range(5)), type=pa.int64()),
        "task_index": pa.array([0] * 5, type=pa.int64()),
    })
    pq.write_table(table, directory / "batch1_1793172869923601.parquet")
    return directory


class FixtureTests(unittest.TestCase):
    def test_parse_info_reads_schema(self):
        with tempfile.TemporaryDirectory() as tmp:
            episode = _synthetic_info(tmp)
            schema = parse_info(episode / "lr_info.json")
            self.assertEqual(schema["robot_type"], "Galbot-Mobile")
            self.assertEqual(schema["fps"], 30)
            self.assertEqual(schema["total_frames"], 5)
            self.assertEqual(
                schema["features"]["action"]["shape"], [20])
            self.assertEqual(
                schema["features"]["observation.state"]["shape"], [3])

    def test_inventory_counts_episodes_per_batch(self):
        # 8 files per episode dir: 2 batch1 episodes + 1 batch3 episode.
        siblings = []
        for i in range(2):
            for f in ("data/chunk-000/file-000.parquet",
                      "meta/info.json", "meta/stats.json",
                      "meta/tasks.parquet",
                      "meta/episodes/chunk-000/file-000.parquet",
                      "videos/a/chunk-000/file-000.mp4",
                      "videos/b/chunk-000/file-000.mp4",
                      "videos/c/chunk-000/file-000.mp4"):
                siblings.append({"rfilename":
                    f"Roco_ChamberTask_LeRobot_v30/batch1/ep{i}/{f}"})
        for f in ("data/chunk-000/file-000.parquet",
                  "meta/info.json", "meta/stats.json",
                  "meta/tasks.parquet",
                  "meta/episodes/chunk-000/file-000.parquet",
                  "videos/a/chunk-000/file-000.mp4",
                  "videos/b/chunk-000/file-000.mp4",
                  "videos/c/chunk-000/file-000.mp4"):
            siblings.append({"rfilename":
                f"Roco_ChamberTask_LeRobot_v30/batch3/ep0/{f}"})
        siblings.append({"rfilename": "Roco_ChamberTask/batch3/x.tar.gz"})
        listing = {"id": "sgeaichallenge/ecm2027_sim_data",
                   "lastModified": "2026-11-26T21:37:35.000Z",
                   "siblings": siblings}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "api.json"
            path.write_text(json.dumps(listing))
            inv = inventory(path)
            self.assertEqual(inv["lerobot_episodes_per_batch"],
                             {"batch1": 2, "batch3": 1})
            self.assertEqual(inv["lerobot_episodes_total"], 3)
            self.assertEqual(inv["lerobot_files"], 24)
            self.assertEqual(inv["batch3_tarballs"], 1)

    @unittest.skipIf(pq is None, "pyarrow not installed")
    def test_stats_reproduce_and_layout_inferred(self):
        with tempfile.TemporaryDirectory() as tmp:
            episode = _synthetic_episode(tmp)
            parquet = episode / "batch1_1793172869923601.parquet"
            checks = check_stats_reproduction(
                parquet, episode / "lr_stats.json")
            self.assertTrue(checks["action"]["reproduces_exactly"])
            self.assertTrue(
                checks["observation.state"]["reproduces_exactly"])
            layout = action_layout(parquet)
            self.assertTrue(layout["left_quat_unit_norm"])
            self.assertTrue(layout["right_quat_unit_norm"])
            self.assertEqual(layout["gripper_left_values"], [1.0])
            self.assertEqual(layout["gripper_right_values"], [-1.0, 1.0])
            self.assertTrue(layout["gripper_left_constant"])
            self.assertTrue(layout["frame_index_contiguous"])
            self.assertAlmostEqual(layout["timestamp_dt_median"],
                                   1.0 / 30.0, places=5)
            # float32 storage rounds the fixture values.
            for got, want in zip(layout["first_frame_left_pos"],
                                 [0.1, 0.2, 0.9]):
                self.assertAlmostEqual(got, want, places=6)
            self.assertEqual(layout["state_joint_like_columns"], [0])

    @unittest.skipIf(pq is None, "pyarrow not installed")
    def test_rest_pose_comparison_finds_match_and_miss(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            demo_dir = tmp / "data"
            demo_dir.mkdir()
            ee = np.zeros((10, 2, 7), dtype=np.float64)
            ee[0, 0, 0:3] = [0.1, 0.2, 0.9]   # matches synthetic left
            ee[0, 1, 0:3] = [0.3, -0.2, 0.9]  # matches synthetic right
            np.savez(demo_dir / "demo_match_ee_poses.npz", ee_poses=ee)
            ee2 = np.zeros((10, 2, 7), dtype=np.float64)
            ee2[0, 0, 0:3] = [1.0, 1.0, 1.0]
            ee2[0, 1, 0:3] = [1.0, 1.0, 1.0]
            np.savez(demo_dir / "demo_far_ee_poses.npz", ee_poses=ee2)
            comp = rest_pose_comparison([0.1, 0.2, 0.9],
                                        [0.3, -0.2, 0.9], demo_dir)
            self.assertAlmostEqual(comp["demo_match_ee_poses"]["left_l2"],
                                   0.0, places=9)
            self.assertGreater(comp["demo_far_ee_poses"]["left_l2"], 1.0)


@unittest.skipIf(pq is None, "pyarrow not installed")
class RealDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.episode_dir = _first_existing(EPISODE_CANDIDATES)
        cls.api_path = _first_existing(INVENTORY_CANDIDATES)
        if cls.episode_dir is None or cls.api_path is None:
            raise unittest.SkipTest("downloaded LeRobot episode missing")

    def test_real_inventory(self):
        inv = inventory(self.api_path)
        self.assertEqual(inv["lerobot_episodes_total"], 373)
        self.assertEqual(inv["lerobot_episodes_per_batch"],
                         {"batch1": 173, "batch3": 200})
        self.assertEqual(inv["batch3_tarballs"], 200)
        self.assertEqual(inv["last_modified"], "2026-11-26T21:37:35.000Z")

    def test_real_schema(self):
        schema = parse_info(self.episode_dir / "lr_info.json")
        self.assertEqual(schema["robot_type"], "Galbot-Mobile")
        self.assertEqual(schema["fps"], 30)
        self.assertEqual(schema["total_frames"], 7023)
        self.assertEqual(schema["features"]["action"]["shape"], [20])
        self.assertEqual(
            schema["features"]["observation.state"]["shape"], [77])

    def test_real_stats_and_layout(self):
        parquet = self.episode_dir / "batch1_1793172869923601.parquet"
        checks = check_stats_reproduction(
            parquet, self.episode_dir / "lr_stats.json")
        self.assertTrue(checks["action"]["reproduces_exactly"])
        self.assertTrue(
            checks["observation.state"]["reproduces_exactly"])
        layout = action_layout(parquet)
        self.assertEqual(layout["rows"], 7023)
        self.assertTrue(layout["left_quat_unit_norm"])
        self.assertTrue(layout["right_quat_unit_norm"])
        self.assertEqual(layout["gripper_left_values"], [1.0])
        self.assertEqual(layout["gripper_right_values"], [-1.0, 1.0])
        self.assertTrue(layout["frame_index_contiguous"])
        self.assertAlmostEqual(layout["timestamp_dt_median"],
                               1.0 / 30.0, places=4)
        self.assertEqual(layout["state_joint_like_count"], 30)

    def test_real_rest_pose_matches_no_demo(self):
        report = analyze(self.api_path, self.episode_dir,
                         PACKAGE_ROOT / "data")
        comp = report["rest_pose_comparison"]
        self.assertEqual(len(comp), 9)
        for name, entry in comp.items():
            self.assertGreater(entry["left_l2"], 0.01,
                               f"{name} unexpectedly matches")
            self.assertGreater(entry["right_l2"], 0.01,
                               f"{name} unexpectedly matches")
        md = render_markdown(report)
        self.assertIn("373 episode dirs", md)
        self.assertIn("reproduces exactly", md)
        self.assertIn("Quat component order", md)


if __name__ == "__main__":
    unittest.main()
