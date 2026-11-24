"""Issue #8: randomization and scoring, from task code and demo data.

Upstream issue #8 (rocochallenge/ecm2027_sim_track, still open with
0 comments) asks which objects are randomized besides the sponge pose,
and how the lid lock and wipe milestones are scored. These tests pin
the answers the analysis tool extracts: the scoring spec parsed from
the starter repo's task code, the initial-state spread measured from
the Batch 2 demo recordings, and the honest gap that per-step demo
data carries no lid or contact state at all.

The code half runs anywhere the sibling starter checkout exists. The
data half needs h5py (the data-sample venv python) and the Batch 2
HDF5 samples.
"""

import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
STARTER_ROOT = PACKAGE_ROOT.parent / "ecm2027-sim-track"
TASK_PATH = STARTER_ROOT / "robofinals_tasks" / "chamber_service_task.py"

# Batch 2 HDF5 samples: next to the repo in the real checkout, or at
# the canonical goal path when running from an isolated clone.
SAMPLE_CANDIDATES = [
    PACKAGE_ROOT.parent.parent / "ecm-data-sample" / "batch2",
    Path.home() / "workspace" / "goals" / "ai-2" / "hidden_files"
    / "ecm-data-sample" / "batch2",
]


def _sample_dir():
    for candidate in SAMPLE_CANDIDATES:
        if candidate.is_dir() and list(candidate.glob("*.hdf5")):
            return candidate
    return None


try:
    import h5py  # noqa: F401
except ImportError:  # system python has no h5py; venv python does
    h5py = None

from tools.analyze_scoring import (  # noqa: E402
    analyze,
    load_initial_states,
    parse_scoring,
    summarize_initial_states,
    to_markdown,
    yaw_from_quat_xyzw,
)


@unittest.skipUnless(
    TASK_PATH.exists(), "starter repo task code missing"
)
class ScoringSpecTests(unittest.TestCase):
    def test_thirteen_points_six_stages(self):
        spec = parse_scoring(TASK_PATH)
        self.assertEqual(spec["max_score"], 13)
        self.assertEqual(len(spec["point_names"]), 13)
        self.assertEqual(
            spec["point_names"],
            [
                "1a_lid_lifted_1cm",
                "1b_lid_stable_open",
                "2_part_lifted_1cm",
                "3_part_in_box_stable",
                "4a_sponge_picked_up",
                "4b_wipe_wall_x_neg",
                "4b_wipe_wall_x_pos",
                "4b_wipe_wall_y_neg",
                "4b_wipe_wall_y_pos",
                "4b_wipe_floor",
                "5a_new_part_lifted_1cm",
                "5b_new_part_placed_stable",
                "6_lid_closed_stable",
            ],
        )
        self.assertEqual(
            spec["stage_point_slices"],
            [(0, 2), (2, 3), (3, 4), (4, 10), (10, 12), (12, 13)],
        )
        self.assertEqual(len(spec["stage_names"]), 6)
        self.assertEqual(spec["wipe_faces"], [
            "wall_x_neg", "wall_x_pos", "wall_y_neg", "wall_y_pos",
            "floor",
        ])

    def test_scoring_thresholds(self):
        spec = parse_scoring(TASK_PATH)
        t = spec["thresholds"]
        self.assertAlmostEqual(t["LID_LIFT_HEIGHT_M"], 0.01)
        self.assertAlmostEqual(t["LID_OPEN_THRESHOLD"], 0.4)
        self.assertAlmostEqual(t["LID_CLOSED_THRESHOLD"], 0.005)
        self.assertAlmostEqual(t["LID_STABLE_ANG_VEL"], 0.1)
        self.assertAlmostEqual(t["STABLE_DURATION_S"], 1.0)
        self.assertAlmostEqual(t["STABLE_LIN_VEL"], 0.05)
        self.assertAlmostEqual(t["NEW_PART_PLACE_DISTANCE_M"], 0.05)
        self.assertAlmostEqual(t["SPONGE_CHAMBER_CONTACT_DURATION_S"], 0.5)
        self.assertAlmostEqual(t["WIPE_FACE_MAX_DISTANCE_M"], 0.08)
        self.assertAlmostEqual(t["CHAMBER_INNER_HEIGHT_M"], 0.30)

    def test_no_lid_lock_point(self):
        """The word 'lock' never appears in a scored point name."""
        spec = parse_scoring(TASK_PATH)
        self.assertFalse(spec["lid_lock_point"])
        self.assertEqual(len(spec["lock_mentions"]), 1)
        line_no, text = spec["lock_mentions"][0]
        self.assertIn("close and lock the chamber", text)
        self.assertGreater(line_no, 650)

    def test_randomization_spec(self):
        spec = parse_scoring(TASK_PATH)
        self.assertEqual(spec["sponge_base_pos"], (0.37, -0.50, 0.85))
        self.assertAlmostEqual(spec["sponge_pos_jitter"], 0.05)
        t = spec["thresholds"]
        self.assertAlmostEqual(t["ROBOT_RESET_POS_DEV_X"], 0.10)
        self.assertAlmostEqual(t["ROBOT_RESET_POS_DEV_Y"], 0.10)
        # Upstream 17159e5 (2026-11-23): yaw deviation narrowed from
        # +/-0.50 rad (+/-29 deg) to +/-10 deg around the demo anchor.
        self.assertAlmostEqual(
            t["ROBOT_RESET_YAW_DEV"], 0.17453292519943295
        )
        self.assertEqual(
            spec["robot_reset_anchor_pos"], (0.40, -0.85, 0.0)
        )
        self.assertAlmostEqual(
            spec["robot_reset_anchor_ori"][2], math.pi / 2
        )
        self.assertFalse(spec["keep_table_static"])


class YawTests(unittest.TestCase):
    def test_sponge_quat_form_gives_yaw(self):
        # Sponge quats are (cos(yaw/2), sin(yaw/2), 0, 0): Rx(pi) then Rz.
        # 4.58 rad is outside (-pi, pi], so compare modulo 2*pi.
        yaw = 4.58
        quat = (math.cos(yaw / 2), math.sin(yaw / 2), 0.0, 0.0)
        got = yaw_from_quat_xyzw(quat)
        diff = (got - yaw + math.pi) % (2.0 * math.pi) - math.pi
        self.assertAlmostEqual(diff, 0.0, places=6)

    def test_robot_quat_form_gives_yaw(self):
        quat = (0.0, 0.0, math.sin(math.pi / 4), math.cos(math.pi / 4))
        self.assertAlmostEqual(
            yaw_from_quat_xyzw(quat), math.pi / 2, places=6
        )


@unittest.skipUnless(h5py is not None, "h5py not installed")
@unittest.skipUnless(_sample_dir() is not None, "Batch 2 samples missing")
class InitialStateTests(unittest.TestCase):
    def setUp(self):
        self.files = sorted(_sample_dir().glob("*.hdf5"))
        self.states = load_initial_states(self.files)
        self.summary = summarize_initial_states(self.states)

    def test_duplicate_episode_deduplicated(self):
        # The same demo_0 recording ships in both Batch 2 files.
        self.assertEqual(self.summary["raw_entries"], 5)
        self.assertEqual(self.summary["unique_episodes"], 4)
        self.assertEqual(self.summary["duplicates"], 1)

    def test_sponge_varies_within_jitter_band(self):
        sponge = self.summary["objects"]["spongebody"]
        self.assertEqual(sponge["n"], 4)
        # x/y stay inside base +/- 0.05 m (base (0.37, -0.50)).
        self.assertGreaterEqual(sponge["pos_min"][0], 0.37 - 0.05)
        self.assertLessEqual(sponge["pos_max"][0], 0.37 + 0.05)
        self.assertGreaterEqual(sponge["pos_min"][1], -0.50 - 0.05)
        self.assertLessEqual(sponge["pos_max"][1], -0.50 + 0.05)
        # Yaw is uniform over [0, 2*pi) in code; the sample must show it.
        self.assertGreater(sponge["yaw_spread_rad"], 2.0)

    def test_sponge_z_straddles_spawn_band_low_edge(self):
        # Spawn z band is [0.80, 0.90]; recorded z is post-settle.
        sponge = self.summary["objects"]["spongebody"]
        self.assertLess(sponge["pos_min"][2], 0.80)
        self.assertLessEqual(sponge["pos_max"][2], 0.90)

    def test_robot_base_fixed_in_data(self):
        robot = self.summary["objects"]["robot"]
        self.assertEqual(robot["pos_range"], [0.0, 0.0, 0.0])
        self.assertEqual(robot["yaw_spread_rad"], 0.0)

    def test_reset_anchor_matches_demo_base(self):
        """Upstream 17159e5 anchors the eval reset at the demo base pose.

        The measured demo base pose is (0.40, -0.85, 0.0), yaw pi/2,
        exactly the ROBOT_RESET_ANCHOR_* constants in the task code, so
        the demos sit at the centre of the eval distribution.
        """
        robot = self.summary["objects"]["robot"]
        self.assertEqual(robot["pos_min"], [0.4, -0.85, 0.0])
        self.assertEqual(robot["pos_max"], [0.4, -0.85, 0.0])
        self.assertAlmostEqual(robot["yaw_min"], math.pi / 2, places=5)
        self.assertAlmostEqual(robot["yaw_max"], math.pi / 2, places=5)
        result = analyze(TASK_PATH, self.files)
        self.assertTrue(
            result["findings"]["robot_reset_anchor_matches_demo_base"]
        )

    def test_chamber_parts_table_and_lid_fixed(self):
        for name in ("Chamber002", "Chamber002_01", "Table291",
                     "Chamber001_01"):
            obj = self.summary["objects"][name]
            self.assertEqual(obj["pos_range"], [0.0, 0.0, 0.0], name)
            self.assertEqual(obj["yaw_spread_rad"], 0.0, name)
        self.assertTrue(self.summary["lid_joints"]["all_zero"])
        self.assertEqual(self.summary["lid_joints"]["max_abs"], 0.0)

    def test_analyze_findings(self):
        result = analyze(TASK_PATH, self.files)
        findings = result["findings"]
        self.assertEqual(findings["randomized_in_data"], ["spongebody"])
        self.assertIn("robot", findings["fixed_in_data"])
        self.assertIn("Chamber002", findings["fixed_in_data"])
        self.assertTrue(
            findings["robot_base_randomized_in_code_but_fixed_in_data"]
        )
        self.assertFalse(findings["lid_lock_scored"])
        self.assertTrue(findings["demo_data_cannot_score_wipe_or_lid"])
        self.assertTrue(
            result["initial_states"]["objects"]["spongebody"]
            ["xy_within_jitter"]
        )

    def test_analysis_is_deterministic(self):
        import contextlib
        import io

        import tools.analyze_scoring as mod

        with tempfile.TemporaryDirectory() as tmp:
            first = Path(tmp) / "first.json"
            second = Path(tmp) / "second.json"
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                mod.main(["--json", str(first)])
                mod.main(["--json", str(second)])
            self.assertEqual(
                first.read_text(encoding="utf-8"),
                second.read_text(encoding="utf-8"),
            )

    def test_markdown_states_the_gap(self):
        result = analyze(TASK_PATH, self.files)
        md = to_markdown(result)
        self.assertIn("no such scored point", md)
        self.assertIn("BIT-IDENTICAL", md)
        self.assertIn("cannot be reconstructed from demo data", md)
        # The prose derives the yaw deviation from the parsed code, so
        # it must show the new +/-10 deg value, not the old +/-0.50 rad.
        self.assertIn("0.1745", md)
        self.assertIn("10.0 degrees", md)
        self.assertIn("17159e5", md)
        self.assertNotIn("0.50 rad", md)
        json_text = json.dumps(result, sort_keys=True)
        self.assertIn("spongebody", json_text)


if __name__ == "__main__":
    unittest.main()
