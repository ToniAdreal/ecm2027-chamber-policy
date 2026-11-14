"""BasePolicy adapter for the starter repo's local policy-evaluation harness.

The starter repository evaluates local policies through
``robofinals/applications/eval/entrypoints/policy.py``: it instantiates the
class named by ``policy_name: module:Class`` in the evaluation YAML, calls
``reset_model()``, then ``eval(task_env, observation, usr_args,
video_writer)`` per batch, and summarizes the returned per-episode booleans
with ``format_eval_results``.

``ChamberReplayPolicy`` wraps :class:`ReplayFollowCore` (single demo) or
:class:`MultiDemoCore` (v3: nearest-start demo selection over a directory
of demos, selected from the episode's initial observation) in that
interface.
Two integration details from the starter code are handled explicitly:

* The entrypoint only adds ``usr_args['actions_dim']`` (from simulator
  metadata) *after* constructing the policy, so the core is built with the
  documented Galbot-Mobile default (20) and re-sized in ``eval()``.
* ``BasePolicy.step_environment`` moves NumPy actions to CUDA. The README
  warns that with the supplied CPU physics config a custom policy should
  pass a tensor on the configured environment device or implement its own
  stepping; this adapter therefore builds CPU float32 tensors itself and
  calls ``task_env.step`` directly.

Success reporting: the chamber task registers its checker as the ``success``
termination term (``robofinals/modules/tasks/base.py``). This adapter
prefers an explicit ``extras["success"]`` signal when the server provides
one and otherwise falls back to the per-env termination flags returned by
the final step, documenting that fallback in the batch diagnostics it
writes under ``usr_args['save_path']`` when available.

Rollout budget: the simulator server disables the environment timeout
termination, and the README states the policy must enforce its own rollout
limit. The step budget defaults to the demo length plus 600 hold steps
(overridable via ``usr_args['policy_max_steps']``); ``time_out_limit`` from
the YAML is additionally honoured as a wall-clock safety cap in seconds.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np

try:  # torch exists in the simulator environment; absent on plain CPU hosts
    import torch
except ImportError:  # pragma: no cover - exercised only off-sim hosts
    torch = None

from policy.base import BasePolicy

from .core import ReplayFollowCore
from .multidemo import MultiDemoCore
from .trajectory import load_ee_poses_hdf5, load_ee_poses_npz

_PACKAGE_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_TRAJECTORY = _PACKAGE_ROOT / "data" / "demo_7_ee_poses.npz"
_CAMERA_KEYS = ["first_person_camera", "left_hand_camera", "right_hand_camera"]


def _to_numpy(value) -> np.ndarray:
    if torch is not None and torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


class ChamberReplayPolicy(BasePolicy):
    """Demonstration-trajectory-following policy for ChamberServiceTask."""

    # -- BasePolicy contract --------------------------------------------------

    def get_model(self, usr_args: Dict[str, Any]):
        """Build the policy core.

        Multi-demo mode (v3): when ``trajectory_dir`` / ``demo_dir`` is
        given, or ``trajectory_path`` / ``checkpoint`` points at a
        directory, or ``ECM_TRAJECTORY_DIR`` is set, build a
        MultiDemoCore over every ``*_ee_poses.npz`` bundle in that
        directory. Otherwise the v1 single-demo path is unchanged.
        """
        core_kwargs = dict(
            actions_dim=int(usr_args.get("actions_dim", 20)),
            # Default lookahead 2 (was 1): the M8 lookahead sweep
            # (2026-11-13, tools/sweep_lookahead.py, 6 demos x alpha
            # 0.3/0.5) found lookahead 2 beats 1 at both swept alphas
            # on demo-averaged mean position error (0.609 vs 0.618 m
            # at alpha 0.3; 0.010 vs 0.618 m at alpha 0.5), while
            # lookahead 4 wins only at 0.3 and loses badly at 0.5.
            # The evaluator can still override via policy_lookahead.
            lookahead=int(usr_args.get("policy_lookahead", 2)),
            phase_window=int(usr_args.get("policy_phase_window", 250)),
            gripper_left=float(usr_args.get("policy_gripper_left", 1.0)),
            gripper_right=float(usr_args.get("policy_gripper_right", 1.0)),
            action_quat_order=str(usr_args.get("policy_action_quat_order", "wxyz")),
        )
        demo_dir = self._resolve_demo_dir(usr_args)
        if demo_dir is not None:
            self.model = MultiDemoCore.from_npz_dir(demo_dir, **core_kwargs)
            return self.model
        trajectory_path = self._resolve_trajectory_path(usr_args)
        if trajectory_path.suffix == ".hdf5":
            episode = usr_args.get("trajectory_episode", "demo_7")
            ee_poses = load_ee_poses_hdf5(trajectory_path, episode)
        else:
            ee_poses = load_ee_poses_npz(trajectory_path)
        self.model = ReplayFollowCore(ee_poses, **core_kwargs)
        return self.model

    def get_action(self):
        """Return the next action for the current observation-less step."""
        return self.model.predict()

    def reset_model(self) -> None:
        if self.model is not None:
            self.model.reset()

    def eval(self, task_env, observation, usr_args, video_writer):
        core = self.model
        if usr_args.get("actions_dim"):
            core.set_actions_dim(int(usr_args["actions_dim"]))
        if isinstance(core, MultiDemoCore):
            # Select the demo from the episode's observed initial EE pose.
            ee = self._extract_ee(observation)
            if ee is not None:
                core.reset(observation={"eef_pos": ee})
            else:
                core.reset()
        else:
            core.reset()

        env_cfg = usr_args.get("env_cfg") or {}
        num_envs = max(1, int(env_cfg.get("num_envs", 1))) if isinstance(env_cfg, dict) else 1
        step_budget = int(usr_args.get("policy_max_steps", 0)) or (core.num_steps + 600)
        wall_cap_s = float(usr_args.get("time_out_limit", 0) or 0)

        terminated_flags = np.zeros(num_envs, dtype=bool)
        success_flags = np.zeros(num_envs, dtype=bool)
        saw_explicit_success = False
        obs = observation
        steps = 0
        start = time.monotonic()
        while steps < step_budget:
            action_np = self._batched_action(core, obs, num_envs)
            action = self._to_env_action(action_np)
            obs, _reward, terminated, _truncated, extras = task_env.step(action)
            steps += 1

            term = np.atleast_1d(_to_numpy(terminated).astype(bool))[:num_envs]
            if term.shape[0] < num_envs:
                term = np.pad(term, (0, num_envs - term.shape[0]))
            explicit = self._extract_success(extras, num_envs)
            if explicit is not None:
                saw_explicit_success = True
                success_flags |= explicit
            terminated_flags |= term

            if video_writer is not None:
                try:
                    self.add_video_frame(video_writer, obs, _CAMERA_KEYS)
                except Exception:
                    pass  # video is diagnostic only; never fail a rollout on it

            if bool(terminated_flags.all()):
                break
            if wall_cap_s > 0 and (time.monotonic() - start) > wall_cap_s:
                break

        if not saw_explicit_success:
            success_flags = terminated_flags.copy()
        diagnostics = {
            "steps": steps,
            "step_budget": step_budget,
            "terminated": terminated_flags.tolist(),
            "success": success_flags.tolist(),
            "success_source": "extras" if saw_explicit_success else "terminated_fallback",
            "final_phase": core.phase,
        }
        if isinstance(core, MultiDemoCore):
            diagnostics["selected_demo"] = core.selected
            diagnostics["demo_names"] = core.demo_names
        self._write_diagnostics(usr_args, diagnostics)
        return success_flags

    # -- internals --------------------------------------------------------------

    @staticmethod
    def _resolve_demo_dir(usr_args: Dict[str, Any]) -> Path | None:
        for key in ("trajectory_dir", "demo_dir"):
            value = usr_args.get(key)
            if value and Path(str(value)).is_dir():
                return Path(str(value))
        for key in ("trajectory_path", "checkpoint"):
            value = usr_args.get(key)
            if value and Path(str(value)).is_dir():
                return Path(str(value))
        env_dir = os.environ.get("ECM_TRAJECTORY_DIR")
        if env_dir and Path(env_dir).is_dir():
            return Path(env_dir)
        return None

    @staticmethod
    def _resolve_trajectory_path(usr_args: Dict[str, Any]) -> Path:
        for key in ("trajectory_path", "checkpoint"):
            value = usr_args.get(key)
            if value and Path(str(value)).suffix in (".npz", ".hdf5"):
                return Path(str(value))
        env_path = os.environ.get("ECM_TRAJECTORY_PATH")
        return Path(env_path) if env_path else _DEFAULT_TRAJECTORY

    def _batched_action(self, core: ReplayFollowCore, obs, num_envs: int) -> np.ndarray:
        ee = self._extract_ee(obs)
        observation = None
        if ee is not None:
            observation = {"eef_pos": ee}
        single = core.predict(observation)
        return np.repeat(single[None, :], num_envs, axis=0).astype(np.float32)

    @staticmethod
    def _extract_ee(obs) -> np.ndarray | None:
        """Current EE positions (2, 3) for env 0, from starter obs terms."""
        if not isinstance(obs, dict):
            return None
        for container_key in ("policy", "embodiment_general_obs"):
            container = obs.get(container_key)
            if not isinstance(container, dict) or "eef_pos" not in container:
                continue
            try:
                arr = _to_numpy(container["eef_pos"]).astype(np.float64)
            except Exception:
                continue
            if arr.shape == (1, 2, 3):
                return arr[0]
            if arr.shape == (2, 3):
                return arr
            if arr.ndim == 3 and arr.shape[1:] == (2, 3):
                return arr[0]
        return None

    @staticmethod
    def _extract_success(extras, num_envs: int) -> np.ndarray | None:
        if not isinstance(extras, dict) or "success" not in extras:
            return None
        try:
            flags = np.atleast_1d(_to_numpy(extras["success"]).astype(bool))
        except Exception:
            return None
        if flags.shape[0] < num_envs:
            flags = np.pad(flags, (0, num_envs - flags.shape[0]))
        return flags[:num_envs]

    @staticmethod
    def _to_env_action(action_np: np.ndarray):
        if torch is not None:
            return torch.as_tensor(action_np, dtype=torch.float32)
        return action_np

    @staticmethod
    def _write_diagnostics(usr_args: Dict[str, Any], payload: dict) -> None:
        save_path = usr_args.get("save_path")
        if not save_path:
            return
        try:
            out_dir = Path(save_path)
            out_dir.mkdir(parents=True, exist_ok=True)
            with open(out_dir / "replay_policy_diagnostics.json", "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
        except Exception:
            pass  # diagnostics must never break an evaluation batch
