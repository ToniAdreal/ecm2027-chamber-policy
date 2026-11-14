# ECM 2027 Policy v3 — TwinTeleop Multi-Demo Replay-Follow

Policy package for the **Singapore Embodied AI Challenge for Manufacturing**
(ECM 2027, simulation track), Team TwinTeleop (Team ID ECM-122). This is the
Plan B main line: development happens CPU-side on a GPU-less host, and
evaluation is done by the organizers from the submitted package.

## Strategy

**Multi-demonstration trajectory following (v2).** The v1 core replayed a
single demo; v2 wraps one `ReplayFollowCore` per demo and selects, at reset
time, the demo whose first-frame end-effector positions are nearest to the
robot's observed initial positions (`ecm_policy/multidemo.py`). Selection
happens only at reset, so a rollout never jumps between demos mid-episode;
without an observation at reset the first demo is used. The bundled set now
holds six real episodes: four Batch 1 (`demo_0`, `demo_2`, `demo_7`,
`demo_9`) and two Batch 2 (`demo_b2_0`, `demo_b2_3`), extracted from six
independent recordings of the official dataset
(`rocochallenge2025/ecm2027_sim_data`).

The released demonstration data contains dual-arm end-effector pose
trajectories (`ee_poses`, shape `(T, 2, 7)` =
`[x, y, z, qw, qx, qy, qz]` per arm). Batch 1 files carry the combined
`ee_poses` dataset; Batch 2 files (released 2026-11-10, three days early)
carry the split observation terms `obs/eef_pos` + `obs/eef_quat` instead,
composed into the same `(T, 2, 7)` shape by the loader
(`ecm_policy/trajectory.py`), plus a real gripper observation
`obs/gripper_pos` `(T, 2)` that Batch 1 files lack. The Galbot-Mobile embodiment in the
official starter repo is driven by *absolute* differential-IK pose commands
(`use_relative_mode=False`), so a recorded EE pose can be fed back as the
commanded pose directly.

The policy core (`ecm_policy/core.py`, pure NumPy):

* tracks a phase index into the demo trajectory, advancing monotonically;
* when the caller supplies the robot's current EE positions (`eef_pos`, the
  starter repo's own observation term), it re-anchors the phase to the demo
  point nearest the current pose within a forward window (closed-loop phase
  tracking); otherwise it advances one demo step per policy step;
* commands the demo pose at `phase + lookahead` for both arms;
* replays the recorded gripper state when the demo carries one
  (Batch 2): `gripper_pos` is the Galbot gripper joint position
  (0.0 fully open, 1.703 fully closed, per the starter repo's
  `BinaryJointPositionActionCfg`), thresholded at the range midpoint
  0.8515 into the binary gripper action (`< 0` close, `>= 0` open).
  Batch 1 demos carry no gripper channel and still hold both grippers
  at the fixed configurable command. Wheel velocity is zero throughout.

### Trade-offs and honest limitations

* **Gripper replay is built and tested (2026-11-11, milestone 6).**
  Batch 1 recordings carry no gripper channel (verified across four
  independent recordings: each file contains exactly one
  `data/demo_*/ee_poses` dataset and nothing else), so Batch 1 cores
  still hold the fixed command. Batch 2 recordings carry
  `data/demo_*/obs/gripper_pos` `(T, 2)`: in the two verified episodes
  the signal ranges 0 to 1.703 with hundreds of distinct values, is
  strongly bimodal (39.7% of samples < 0.05, most of the rest in
  1.13-1.70), and the two arms' thresholded states agree on 100% of
  steps. The mapping is the starter repo's own convention, verified
  in its code, not invented: `galbot.py` sets the gripper joint
  open/close positions to 0.0 / 1.703 and its `preprocess_device_action`
  states `action < 0` closes and `>= 0` opens. Full-episode open-loop
  rollouts replay the recorded gripper state exactly (demo_b2_0:
  8 open/close transitions, 58.1% closed; demo_b2_3: 6 transitions,
  68% closed). The official evaluation service has still never
  published its expected gripper action scale, so the mapping is
  documented as the starter repo's convention, not as a validated
  organizer claim.
* **Fixed layout assumption, weakened but not removed.** Multi-demo
  selection picks the closest recorded start layout instead of assuming one
  fixed demo, but replay still assumes the evaluation scene starts near a
  recorded layout (the fixed local scene, `scene04_1.usd`) — the same
  premise the starter repo's own replay tooling uses when it compares
  replayed EE poses against `replay_state_ee_poses`. The two Batch 2
  episodes start from the same fixed rest pose as the Batch 1 episodes,
  so they add trajectory diversity, not start-layout diversity.
* **Six demo episodes.** v2 bundles `demo_0` (7,471 steps), `demo_2`
  (5,632 steps), `demo_7` (7,566 steps) and `demo_9` (5,122 steps) from
  Batch 1, plus `demo_b2_0` (4,106 steps) and `demo_b2_3` (4,022 steps)
  from Batch 2 (recordings `ChamberServiceTask_1792826776425892` and
  `ChamberServiceTask_1793084008660268`). Batch 3 data (due 2026-11-28)
  can extend the set with the same extraction tool; the multi-demo core
  makes no per-episode assumptions.
* **Quaternion order.** Demo quaternions are interpreted as Isaac Lab
  FrameTransformer order `(w, x, y, z)` (per `ee_frame_quat` /
  `target_quat_w` in the starter repo) and emitted in the same order for
  the Isaac Lab diff-IK action. `policy_action_quat_order: xyzw` (or the
  core's `action_quat_order`) flips the output order in one line if the
  official validator publishes a different convention.

## Interfaces

Two adapters share the same core, and both now run the multi-demo core
(v3): the service defaults to multi-demo mode over `data/` (use
`--trajectory <single .npz>` for the old single-demo mode), and the
adapter builds a `MultiDemoCore` when `trajectory_dir` / `demo_dir` /
`ECM_TRAJECTORY_DIR` is set or when `trajectory_path` points at a
directory. In multi-demo mode the episode's initial observation selects
the demo: the service accepts an `observation` on `/reset`, and the
adapter passes the harness's first observation into `reset()`. Without
an observation both fall back to the first demo, exactly as v2 did.

1. **Docker policy service** (`ecm_policy/service.py`) — the planned
   official submission shape. The starter README names the service
   operations `health()`, `reset(task_info)`, `predict(observation)`,
   `close()` but states the transport and message schemas are still to be
   published, so this service implements them as small JSON-over-HTTP
   endpoints (stdlib only, offline at runtime):

   | Endpoint | Body | Response |
   | - | - | - |
   | `GET /health` | – | readiness, demo list, selected demo, phase |
   | `POST /reset` | `{"task_info": {...}}` (+ optional `observation`) | `{"status": "ok", "phase": 0, "selected": "demo_9"}` |
   | `POST /predict` | `{"observation": {"eef_pos": [[x,y,z],[x,y,z]]}}` | `{"action": [20 floats], "phase": n, "steps_taken": n, "selected": "demo_9"}` |
   | `POST /close` | `{}` | `{"status": "ok", "closed": true}`; the process then exits unless started with `--no-shutdown-on-close` |

   The observation is optional; without `eef_pos` the policy replays
   open-loop. `predict` before `reset` is allowed (initial state), and
   `predict` after `close` returns HTTP 409 until the next `reset`.

2. **Local harness adapter** (`ecm_policy/adapter.py`) —
   `ChamberReplayPolicy`, a subclass of the starter repo's real
   `policy.base.BasePolicy`, for `scripts/run_policy_eval.sh`. It sizes the
   action vector from the harness-provided `actions_dim`, steps the
   environment with CPU tensors (the base class's helper would force CUDA;
   the starter README explicitly tells custom policies to step themselves
   under the CPU physics config), enforces its own rollout budget (the
   server disables the env timeout), and reports success from
   `extras["success"]` when present, else from the termination flags.
   See `chamber_policy.local.yml.example`.

## Layout

```
ecm_policy/
  trajectory.py   ee_poses loading/validation + convention notes
  core.py         ReplayFollowCore (pure NumPy)
  multidemo.py    MultiDemoCore: nearest-start demo selection over cores
  service.py      JSON-over-HTTP health/reset/predict/close service
  adapter.py      BasePolicy adapter for the starter local harness
data/
  demo_0_ee_poses.npz   bundled demo episode (7,471 steps)
  demo_2_ee_poses.npz   bundled demo episode (5,632 steps)
  demo_7_ee_poses.npz   bundled demo episode (7,566 steps)
  demo_9_ee_poses.npz   bundled demo episode (5,122 steps)
tools/
  extract_demo_trajectory.py   HDF5 episode -> npz bundle
tests/
  test_core.py      core unit tests (real demo + synthetic trajectories)
  test_multidemo.py multi-demo selection tests (synthetic + real 4-demo set)
  test_service.py   live-server end-to-end tests
  test_adapter.py   real starter BasePolicy + mock-env rollout tests
Dockerfile          python:3.12-slim, runs `python -m ecm_policy.service`
chamber_policy.local.yml.example
```

## Run

Service (same command the Docker image runs):

```bash
pip install -r requirements.txt
PYTHONPATH=. python -m ecm_policy.service --host 0.0.0.0 --port 8080
```

Docker (image build not yet verified on a Docker-capable host):

```bash
docker build -t ecm2027-policy:v1 .
docker run --rm -p 8080:8080 ecm2027-policy:v1
```

Tests (CPU only; adapter tests stub torch/mediapy just to load the real
starter BasePolicy, and drive a mock env — no simulator is exercised):

```bash
python -m unittest discover -s tests -p 'test_*.py' -v
```

Extract another demo episode:

```bash
python tools/extract_demo_trajectory.py \
  --input /path/to/replay_state_ee_poses.hdf5 --episode demo_7 \
  --output data/demo_7_ee_poses.npz
```

## Verification status (2026-11-13, lookahead-sweep milestone 8)

Lookahead sweep (`ecm_policy/evaluate.py`, `tools/sweep_lookahead.py`):
every bundled demo at `alpha` 0.3 / 0.5 for lookaheads 0 / 1 / 2 / 4 / 8
(60 full rollouts). Full table in `evaluation/lookahead_sweep.md`
(also `lookahead_sweep.json`). Demo-averaged mean position error:

| alpha | lookahead 0 | 1 | 2 | 4 | 8 |
|---|---|---|---|---|---|
| 0.3 | 1.086 m | 0.618 m | 0.609 m | **0.514 m** | 0.719 m |
| 0.5 | 1.086 m | 0.618 m | **0.010 m** | 0.703 m | 0.759 m |

* **Lookahead 0 stalls the closed-loop tracker.** The policy commands
  the pose the robot is already on, so the nearest-point phase never
  advances: final phase 0 on every demo, mean phase lag ~2,826 steps,
  mean position error 1.086 m — identical at both alphas, because the
  robot never moves. Lookahead must be >= 1 for closed-loop tracking.
  This is a property of the nearest-point tracker, pinned by test.
* **Lookahead 2 is the robust submission default.** It beats lookahead 1
  at both swept alphas (0.609 vs 0.618 m at 0.3; 0.010 vs 0.618 m at
  0.5), and at alpha 0.5 it nearly eliminates tracking error on all 6
  demos independently (mean position error 0.003–0.016 m, orientation
  0.33–1.21 deg, gripper agreement 99.7–99.9% on the Batch 2 demos).
  Lookahead 4 wins only at alpha 0.3 and loses badly at 0.5, so it is
  not the default. The submission entry points (`ecm_policy/adapter.py`
  `policy_lookahead`, `ecm_policy/service.py` `--lookahead`) now default
  to 2; the evaluator can still override it. The core default stays 1.
* **Honest scope:** the near-perfect alpha-0.5 result is a property of
  the harness's first-order lag model (the robot closes a fixed fraction
  of the gap per step), not a measurement of real Isaac Sim dynamics.
  The real evaluator's follower lag is unknown, which is exactly why
  the sweep reports both alphas and the default is the robust choice
  across them, not the best at one. **Not verified:** any run inside
  Isaac Sim (no GPU on the development host), the Docker image build
  (no Docker on the development host), and the official submission
  validation (the official Docker service specification and validator
  had not been released as of 2026-11-13). No competition score is
  claimed.
* The sweep also fixed a real harness bug: `evaluate_lagged_follower`
  normalized quaternions in place on the caller's array, so a second run
  on the same array differed in the 10th decimal. It now copies first;
  determinism is exact (`as_dict()` equality, pinned by test).

The suite is now **62 tests**: 57 pass under the system Python
(`python3 -m pytest tests/ -q`), with 5 Batch 2 layout tests skipped
there because the system Python has no h5py; those 5 pass under the
data-sample venv Python. The 6 new tests cover the sweep: lookahead-0
stall, lookahead-2 outrun at alpha 1.0, exact reproduction of the M7
number at lookahead 1, determinism, lookahead tagging in
`evaluate_all_demos`, and best-lookahead selection.

## Verification status (2026-11-12, lagged-follower evaluation milestone 7)

New offline evaluation harness (`ecm_policy/evaluate.py`,
`tools/evaluate_policy.py`): the simulated robot moves a fraction
`alpha` of the way toward each commanded pose per step (first-order
lag on position, normalized lerp on orientation), the policy observes
the robot's actual EE positions and re-anchors its phase, and the
rollout is scored against the recorded demonstration at the same step
index. Full table in `evaluation/results.md` (also `results.json`).

Measured on the 6 bundled demos at `alpha` 0.1 / 0.3 / 0.5 / 1.0:

* **Perfect follower (`alpha = 1.0`) reproduces every recording
  exactly**: zero position error, orientation error at float-noise
  level (~1e-6 deg), zero phase lag, final phase at the last demo
  step. This is the harness's own sanity check, and it passes on all
  6 demos.
* **Max phase lag strictly decreases as the follower gets faster** on
  all 6 demos (e.g. demo_0: 6,780 → 5,265 → 3,730 → 0 steps):
  closed-loop re-anchoring keeps the policy glued to the robot's
  actual pose instead of drifting.
* **Gripper agreement rises with `alpha`** on both Batch 2 demos
  (demo_b2_0: 41.9% → 53.3% → 61.4% → 99.8%; demo_b2_3: 32.0% →
  51.3% → 68.0% → 99.9%), measured against the recorded gripper
  state at the same step index. The remaining ~0.1–0.2% at
  `alpha = 1.0` is the one-step lookahead at gripper transitions.
* **Honest baseline, not a success claim:** at `alpha` 0.1–0.5 the
  lagging follower cannot keep wall-clock pace with the recording —
  mean position error 0.50–0.70 m across demos, and at `alpha = 0.5`
  the policy only reaches about half the demo by the final step
  (final phase 2,053 of 4,106 on demo_b2_0). Mean position error is
  NOT monotonic in `alpha` (demo_0: 0.52 → 0.58 → 0.62 m) and is
  reported as measured, not forced into a trend. **Not verified:**
  any run inside Isaac Sim (no GPU on the development host), the
  Docker image build (no Docker on the development host), and the
  official submission validation (the official Docker service
  specification and validator had not been released as of 2026-11-12).
  No competition score is claimed.

The suite is now **56 tests**: 51 pass under the system Python
(`python3 -m pytest tests/ -q`), with 5 Batch 2 layout tests skipped
there because the system Python has no h5py; those 5 pass under the
data-sample venv Python. The 6 new tests cover the harness itself:
perfect-follower exactness, phase-lag and gripper-agreement
monotonicity in `alpha`, determinism to ~1e-9 relative, all-demo
clean runs, and invalid-`alpha` rejection.

## Verification status (2026-11-11, gripper-replay milestone 6)

Verified on a GPU-less VM: the suite is now **50 tests**: 45 pass under
the system Python (`python3 -m pytest tests/ -q`), with 5 Batch 2
layout tests skipped there because the system Python has no h5py; those
5 pass under the data-sample venv Python (`python -m unittest discover
-s tests`: 47 tests ran, OK), and the 3 pytest-style rollout functions
also pass when called under the venv Python. Coverage now includes the
6-demo set (4 Batch 1 + 2 Batch 2), Batch 2 split-obs composition,
gripper_pos bundling, gripper replay (threshold mapping, fixed fallback,
full-episode open-loop replay on both Batch 2 episodes, mixed-directory
multi-demo modes), and full-episode rollouts on the shortest demo
(`demo_b2_3`, 4,022 steps). **Not verified:** any run inside Isaac Sim
(no GPU on the development host), the Docker image build (no Docker on
the development host), and the official submission validation (the
official Docker service specification and validator had not been
released as of 2026-11-11). No competition score is claimed.

## Verification status (2026-11-10, Batch 2 milestone)

Verified on a GPU-less VM: the suite is now **39 tests**: 34 pass under
the system Python (`python3 -m pytest tests/ -q`), with 5 Batch 2
layout tests skipped there because the system Python has no h5py; those
5 pass under the data-sample venv Python (`python -m unittest discover
-s tests`: 36 tests ran, OK), and the 3 pytest-style rollout functions
also pass when called under the venv Python. Coverage now includes the
6-demo set (4 Batch 1 + 2 Batch 2), Batch 2 split-obs composition,
gripper_pos bundling, and full-episode rollouts on the shortest demo
(`demo_b2_3`, 4,022 steps). **Not verified:** any run inside Isaac Sim
(no GPU on the development host), the Docker image build (no Docker on
the development host), and the official submission validation (the
official Docker service specification and validator had not been
released as of 2026-11-10). No competition score is claimed.

## Verification status (2026-11-02, v3)

Verified on a GPU-less VM: core unit tests, multi-demo selection tests
(synthetic + real 4-demo set), live HTTP service tests for both the
single-demo and multi-demo cores, and mock-env adapter rollouts including
two new multi-demo adapter tests (selection from the initial observation,
first-demo fallback without EE data) — 31 tests, all passing, in both
the working copy and an isolated copy, plus the starter repo's
7 lightweight regression tests (all passing, before and after this work).
**Not verified:** any run inside Isaac Sim (no GPU on the development host),
the Docker image build (no Docker on the development host), and the
official submission validation (the official Docker service specification
and validator had not been released as of 2026-11-02). No competition
score is claimed for this policy.

## Verification status (2026-11-01, v2)

Verified on a GPU-less VM: core unit tests, multi-demo selection tests
(synthetic + real 4-demo set), live HTTP service tests, and mock-env
adapter rollouts (26 tests, all passing), plus the starter repo's
7 lightweight regression tests (all passing, before and after this work).
**Not verified:** any run inside Isaac Sim (no GPU on the development host),
the Docker image build (no Docker on the development host), and the
official submission validation (the official Docker service specification
and validator had not been released as of 2026-11-01). No competition
score is claimed for this policy.

## Verification status (2026-10-31, v1)

Verified on a GPU-less VM: core unit tests, live HTTP service tests, and
mock-env adapter rollouts (16 tests, all passing), plus the starter repo's
7 lightweight regression tests (all passing, before and after this work).
**Not verified:** any run inside Isaac Sim (no GPU on the development host),
the Docker image build (no Docker on the development host), and the
official submission validation (the official Docker service specification
and validator had not been released as of 2026-10-31). No competition
score is claimed for this policy.
