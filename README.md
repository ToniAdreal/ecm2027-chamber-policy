# ECM 2027 Policy v2 — TwinTeleop Multi-Demo Replay-Follow

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
holds four real Batch 1 episodes (`demo_0`, `demo_2`, `demo_7`, `demo_9`),
extracted from four independent recordings of the official dataset
(`rocochallenge2025/ecm2027_sim_data`).

The released demonstration data contains dual-arm end-effector pose
trajectories (`ee_poses`, shape `(T, 2, 7)` =
`[x, y, z, qw, qx, qy, qz]` per arm). The Galbot-Mobile embodiment in the
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
* holds both grippers at a fixed configurable command and commands zero
  wheel velocity, because the released data has neither channel.

### Trade-offs and honest limitations

* **No gripper data (re-confirmed in v2).** The demo trajectories carry no
  gripper channel, so v2 never changes gripper state (default: open, `+1.0`;
  the starter repo's binary gripper convention is `< 0` close, `>= 0`
  open). Grasp-dependent milestones cannot be earned until a gripper
  strategy exists. This is the single biggest known gap. The absence is now
  verified across four independent Batch 1 recordings, not just the sample:
  each recording file contains exactly one `data/demo_*/ee_poses` dataset
  and nothing else, so no gripper channel exists to extract.
* **Fixed layout assumption, weakened but not removed.** Multi-demo
  selection picks the closest recorded start layout instead of assuming one
  fixed demo, but replay still assumes the evaluation scene starts near a
  recorded layout (the fixed local scene, `scene04_1.usd`) — the same
  premise the starter repo's own replay tooling uses when it compares
  replayed EE poses against `replay_state_ee_poses`.
* **Four demo episodes.** v2 bundles `demo_0` (7,471 steps), `demo_2`
  (5,632 steps), `demo_7` (7,566 steps) and `demo_9` (5,122 steps) from
  Batch 1. Batch 2 data (2026-11-13) can extend the set with the same
  extraction tool; the multi-demo core makes no per-episode assumptions.
* **Quaternion order.** Demo quaternions are interpreted as Isaac Lab
  FrameTransformer order `(w, x, y, z)` (per `ee_frame_quat` /
  `target_quat_w` in the starter repo) and emitted in the same order for
  the Isaac Lab diff-IK action. `policy_action_quat_order: xyzw` (or the
  core's `action_quat_order`) flips the output order in one line if the
  official validator publishes a different convention.

## Interfaces

Two adapters share the same core:

1. **Docker policy service** (`ecm_policy/service.py`) — the planned
   official submission shape. The starter README names the service
   operations `health()`, `reset(task_info)`, `predict(observation)`,
   `close()` but states the transport and message schemas are still to be
   published, so this service implements them as small JSON-over-HTTP
   endpoints (stdlib only, offline at runtime):

   | Endpoint | Body | Response |
   | - | - | - |
   | `GET /health` | – | readiness, trajectory steps, actions dim, phase |
   | `POST /reset` | `{"task_info": {...}}` | `{"status": "ok", "phase": 0}` |
   | `POST /predict` | `{"observation": {"eef_pos": [[x,y,z],[x,y,z]]}}` | `{"action": [20 floats], "phase": n, "steps_taken": n}` |
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
