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
    def test_from_npz_dir_loads_six_real_demos(self):
        core = MultiDemoCore.from_npz_dir(DATA_DIR)
        self.assertEqual(
            core.demo_names,
            ["demo_0", "demo_2", "demo_7", "demo_9", "demo_b2_0", "demo_b2_3"],
        )
        self.assertEqual(core.health()["num_demos"], 6)

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


try:
    import h5py  # noqa: F401
except ImportError:  # system python has no h5py; venv python does
    h5py = None


@unittest.skipUnless(h5py is not None, "h5py not installed")
class Batch2LayoutTests(unittest.TestCase):
    """Batch 2 HDF5 layout: split obs terms instead of combined ee_poses."""

    def _write_hdf5(self, path, episode="demo_0", with_combined=False,
                    with_gripper=False, steps=5, quat_steps=None):
        pos = np.zeros((steps, 2, 3), dtype=np.float32)
        pos[:, 0, 0] = np.arange(steps)
        quat = np.zeros((quat_steps if quat_steps is not None else steps, 2, 4),
                        dtype=np.float32)
        quat[..., 0] = 1.0  # (w, x, y, z) = identity rotation
        with h5py.File(path, "w") as handle:
            if with_combined:
                ee = np.concatenate([pos, quat], axis=-1)
                handle.create_dataset(f"data/{episode}/ee_poses", data=ee)
            handle.create_dataset(f"data/{episode}/obs/eef_pos", data=pos)
            handle.create_dataset(f"data/{episode}/obs/eef_quat", data=quat)
            if with_gripper:
                grip = np.zeros((steps, 2), dtype=np.float32)
                grip[:, 0] = np.linspace(0.0, 1.5, steps)
                handle.create_dataset(
                    f"data/{episode}/obs/gripper_pos", data=grip
                )
        return pos, quat

    def test_composes_ee_poses_from_split_obs(self):
        import tempfile

        from ecm_policy.trajectory import load_ee_poses_hdf5

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "b2.hdf5"
            pos, quat = self._write_hdf5(path)
            got = load_ee_poses_hdf5(path, "demo_0")
            self.assertEqual(got.shape, (5, 2, 7))
            np.testing.assert_allclose(got[:, :, 0:3], pos, atol=1e-6)
            np.testing.assert_allclose(got[:, :, 3:7], quat, atol=1e-6)

    def test_prefers_combined_ee_poses_when_present(self):
        import tempfile

        from ecm_policy.trajectory import load_ee_poses_hdf5

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "b1.hdf5"
            pos, quat = self._write_hdf5(path, with_combined=True)
            # Corrupt the split terms after writing the combined one:
            # the loader must still return the combined dataset.
            with h5py.File(path, "r+") as handle:
                handle["data/demo_0/obs/eef_pos"][:] = 99.0
            got = load_ee_poses_hdf5(path, "demo_0")
            np.testing.assert_allclose(got[:, :, 0:3], pos, atol=1e-6)

    def test_mismatched_obs_lengths_raise(self):
        import tempfile

        from ecm_policy.trajectory import load_ee_poses_hdf5

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.hdf5"
            self._write_hdf5(path, steps=5, quat_steps=4)
            with self.assertRaises(ValueError):
                load_ee_poses_hdf5(path, "demo_0")

    def test_gripper_pos_loads_and_missing_raises(self):
        import tempfile

        from ecm_policy.trajectory import load_gripper_pos_hdf5

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "grip.hdf5"
            self._write_hdf5(path, with_gripper=True)
            got = load_gripper_pos_hdf5(path, "demo_0")
            self.assertEqual(got.shape, (5, 2))
            self.assertAlmostEqual(float(got[-1, 0]), 1.5)
            path2 = Path(tmp) / "nogrip.hdf5"
            self._write_hdf5(path2, with_gripper=False)
            with self.assertRaises(ValueError):
                load_gripper_pos_hdf5(path2, "demo_0")

    def test_bundled_b2_npz_carries_gripper(self):
        for name, steps in [("demo_b2_0", 4106), ("demo_b2_3", 4022)]:
            with np.load(DATA_DIR / f"{name}_ee_poses.npz") as bundle:
                self.assertIn("gripper_pos", bundle)
                grip = bundle["gripper_pos"]
                self.assertEqual(grip.shape, (steps, 2))
                self.assertTrue(np.all(np.isfinite(grip)))
                # Recorded range is about 0 to 1.7 (tiny negative noise
                # at the closed end, about -0.001, is in the raw data).
                self.assertGreaterEqual(float(grip.min()), -0.01)
                self.assertLess(float(grip.max()), 2.0)
                self.assertEqual(
                    int(bundle["ee_poses"].shape[0]), steps
                )


if __name__ == "__main__":
    unittest.main()
