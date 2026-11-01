"""Unit tests for MultiDemoCore (pure CPU, no simulator)."""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ecm_policy.multidemo import MultiDemoCore  # noqa: E402
from ecm_policy.trajectory import load_ee_poses_npz  # noqa: E402

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PACKAGE_ROOT / "data"


def synthetic_demo(start_x: float, steps: int = 10) -> np.ndarray:
    traj = np.zeros((steps, 2, 7), dtype=np.float64)
    for t in range(steps):
        traj[t, 0] = [start_x + 0.1 * t, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0]
        traj[t, 1] = [start_x + 0.1 * t, 0.5, 1.0, 1.0, 0.0, 0.0, 0.0]
    return traj


class MultiDemoSelectionTests(unittest.TestCase):
    def setUp(self):
        self.demos = {
            "alpha": synthetic_demo(0.0),
            "beta": synthetic_demo(5.0),
            "gamma": synthetic_demo(-5.0),
        }

    def test_reset_without_observation_selects_first_demo(self):
        core = MultiDemoCore(self.demos)
        info = core.reset()
        self.assertEqual(core.selected, "alpha")
        self.assertEqual(info["selected"], "alpha")

    def test_reset_with_observation_selects_nearest_start(self):
        core = MultiDemoCore(self.demos)
        obs = {"eef_pos": np.array([[5.05, 0.0, 1.0], [5.05, 0.5, 1.0]])}
        core.reset(observation=obs)
        self.assertEqual(core.selected, "beta")
        action, info = core.predict_with_info()
        self.assertEqual(info["selected"], "beta")
        # First action targets beta's frame at lookahead 1.
        np.testing.assert_allclose(action[0:7], self.demos["beta"][1, 0], atol=1e-6)

    def test_reset_re_selects_from_scratch(self):
        core = MultiDemoCore(self.demos)
        core.reset(observation={"eef_pos": np.array([[5.0, 0.0, 1.0], [5.0, 0.5, 1.0]])})
        self.assertEqual(core.selected, "beta")
        core.reset(observation={"eef_pos": np.array([[-5.0, 0.0, 1.0], [-5.0, 0.5, 1.0]])})
        self.assertEqual(core.selected, "gamma")
        core.reset()
        self.assertEqual(core.selected, "alpha")

    def test_tie_resolves_to_earliest_demo(self):
        demos = {"first": synthetic_demo(0.0), "second": synthetic_demo(0.0)}
        core = MultiDemoCore(demos)
        core.reset(observation={"eef_pos": np.array([[0.0, 0.0, 1.0], [0.0, 0.5, 1.0]])})
        self.assertEqual(core.selected, "first")

    def test_no_demo_raises(self):
        with self.assertRaises(ValueError):
            MultiDemoCore({})

    def test_close_blocks_predict_and_health_reports_selection(self):
        core = MultiDemoCore(self.demos)
        core.reset(observation={"eef_pos": np.array([[5.0, 0.0, 1.0], [5.0, 0.5, 1.0]])})
        health = core.health()
        self.assertEqual(health["selected"], "beta")
        self.assertEqual(health["num_demos"], 3)
        core.close()
        self.assertTrue(core.closed)
        with self.assertRaises(RuntimeError):
            core.predict()

    def test_predict_never_jumps_demos_mid_episode(self):
        core = MultiDemoCore(self.demos)
        core.reset(observation={"eef_pos": np.array([[0.0, 0.0, 1.0], [0.0, 0.5, 1.0]])})
        self.assertEqual(core.selected, "alpha")
        # Later observations near beta must NOT switch the demo mid-episode.
        for _ in range(5):
            _a, info = core.predict_with_info(
                {"eef_pos": np.array([[5.0, 0.0, 1.0], [5.0, 0.5, 1.0]])}
            )
            self.assertEqual(info["selected"], "alpha")


class MultiDemoRealDataTests(unittest.TestCase):
    def test_from_npz_dir_loads_four_real_demos(self):
        core = MultiDemoCore.from_npz_dir(DATA_DIR)
        self.assertEqual(core.demo_names, ["demo_0", "demo_2", "demo_7", "demo_9"])
        self.assertEqual(core.health()["num_demos"], 4)

    def test_real_demo_selection_picks_exact_source_demo(self):
        core = MultiDemoCore.from_npz_dir(DATA_DIR)
        demo_9 = load_ee_poses_npz(DATA_DIR / "demo_9_ee_poses.npz")
        core.reset(observation={"eef_pos": demo_9[0, :, 0:3]})
        self.assertEqual(core.selected, "demo_9")

    def test_empty_dir_raises(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                MultiDemoCore.from_npz_dir(tmp)


if __name__ == "__main__":
    unittest.main()
