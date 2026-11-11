"""Multi-demonstration policy core for the ECM 2027 Chamber Service task.

Motivation (v2): the v1 core replays a single demonstration, so it can only
succeed when the evaluation scene's initial layout matches that one demo.
Batch 1 of the official demonstration dataset releases many independent
recordings of the same task, each holding exactly one demo's ``ee_poses``.
This module wraps one :class:`ReplayFollowCore` per demo and, at reset time,
selects the demo whose *first-frame* end-effector positions are nearest to
the robot's observed initial positions. Selection happens only at reset;
within a rollout the selected demo's own phase tracking applies, so the
policy never jumps between demos mid-episode.

Selection is deliberately conservative: with no observation at reset the
first demo (insertion order) is used, ties resolve to the earliest demo,
and a reset with an observation re-selects from scratch. Nothing here
claims the selected demo matches the evaluation scene; it only removes
the single-demo assumption that v1 baked in.

The gripper gap is narrowed but not closed by this module: Batch 1
recordings contain no gripper channel (verified across four independent
recordings: demo_0, demo_2, demo_7, demo_9), while Batch 2 recordings do
carry obs/gripper_pos, now stored in the demo_b2_* bundles. The policy
still holds its grippers at the configured fixed command because it does
not consume that channel yet.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .core import ReplayFollowCore
from .trajectory import load_ee_poses_npz


class MultiDemoCore:
    """Nearest-start demo selection over multiple ReplayFollowCore demos."""

    def __init__(
        self,
        demos: dict[str, np.ndarray],
        **core_kwargs,
    ):
        if not demos:
            raise ValueError("MultiDemoCore needs at least one demo")
        self._names: list[str] = list(demos.keys())
        self._cores: dict[str, ReplayFollowCore] = {
            name: ReplayFollowCore(np.asarray(traj), **core_kwargs)
            for name, traj in demos.items()
        }
        self._selected: str = self._names[0]
        self._closed = False
        self._last_distances: dict[str, float] | None = None

    # -- construction helpers -------------------------------------------------

    @classmethod
    def from_npz_dir(
        cls,
        directory: str | Path,
        pattern: str = "*_ee_poses.npz",
        **core_kwargs,
    ) -> "MultiDemoCore":
        """Load every matching .npz bundle in a directory as one demo each.

        Demo names are the bundle filenames without the ``_ee_poses`` suffix
        (e.g. ``data/demo_9_ee_poses.npz`` becomes ``demo_9``), sorted for
        deterministic order.
        """
        paths = sorted(Path(directory).glob(pattern))
        if not paths:
            raise ValueError(f"no demo bundles matching {pattern!r} in {directory}")
        demos = {}
        for path in paths:
            name = path.name
            if name.endswith("_ee_poses.npz"):
                name = name[: -len("_ee_poses.npz")]
            else:
                name = path.stem
            demos[name] = load_ee_poses_npz(path)
        return cls(demos, **core_kwargs)

    # -- introspection --------------------------------------------------------

    @property
    def demo_names(self) -> list[str]:
        return list(self._names)

    @property
    def selected(self) -> str:
        return self._selected

    @property
    def selected_core(self) -> ReplayFollowCore:
        return self._cores[self._selected]

    @property
    def phase(self) -> int:
        return self.selected_core.phase

    @property
    def num_steps(self) -> int:
        """Length of the currently selected demo's trajectory."""
        return self.selected_core.num_steps

    @property
    def closed(self) -> bool:
        return self._closed

    def set_actions_dim(self, actions_dim: int) -> None:
        """Resize every wrapped demo core (mirrors ReplayFollowCore)."""
        for core in self._cores.values():
            core.set_actions_dim(actions_dim)

    def health(self) -> dict:
        return {
            "status": "ok",
            "ready": not self._closed,
            "closed": self._closed,
            "num_demos": len(self._names),
            "demo_names": list(self._names),
            "selected": self._selected,
            "selected_trajectory_steps": self.selected_core.num_steps,
            "phase": self.selected_core.phase,
            "steps_taken": self.selected_core.steps_taken,
        }

    # -- lifecycle ------------------------------------------------------------

    def reset(
        self,
        task_info: dict | None = None,
        observation: dict | None = None,
    ) -> dict:
        """Reset all demos; select by observed initial EE positions if given."""
        self._closed = False
        for core in self._cores.values():
            core.reset(task_info)
        self._last_distances = None
        self._selected = self._names[0]
        if observation is not None:
            ee_pos = ReplayFollowCore._extract_ee_pos(observation)
            if ee_pos is not None:
                self._select_nearest(ee_pos)
        return {
            "selected": self._selected,
            "phase": 0,
            "steps_taken": 0,
        }

    def close(self) -> None:
        self._closed = True
        for core in self._cores.values():
            core.close()

    # -- inference ------------------------------------------------------------

    def predict(self, observation: dict | None = None) -> np.ndarray:
        action, _info = self.predict_with_info(observation)
        return action

    def predict_with_info(self, observation: dict | None = None):
        if self._closed:
            raise RuntimeError("policy is closed; call reset() before predict()")
        action, info = self.selected_core.predict_with_info(observation)
        info = {"selected": self._selected, **info}
        return action, info

    # -- internals --------------------------------------------------------------

    def _select_nearest(self, ee_pos: np.ndarray) -> str:
        """Pick the demo whose first-frame EE positions are nearest."""
        distances = {}
        for name, core in self._cores.items():
            first = core.first_ee_positions
            distances[name] = float(
                np.linalg.norm(first - ee_pos, axis=-1).mean()
            )
        self._last_distances = distances
        # min() keeps the earliest demo on ties (dict preserves insertion order).
        self._selected = min(distances, key=distances.get)
        return self._selected
