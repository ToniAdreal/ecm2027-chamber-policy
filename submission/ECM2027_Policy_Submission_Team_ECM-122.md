# Singapore Embodied AI Challenge for Manufacturing (ECM 2027) — Simulation Track Policy Submission

**Team:** TwinTeleop — **Team ID: ECM-122**
**Team lead:** Toni Adreal — toniadreal11@gmail.com
**Submission date:** 2026-11-11 (Beijing)
**Policy version:** v3.2 — multi-demo replay-follow (gripper replay)

## Where the code lives

Full policy code, tests, and this document are public at:

**https://github.com/ToniAdreal/ecm2027-chamber-policy**

Commit submitted: `d6f2af477af1fc72242c813f7309c3e0565486c6`
("ECM 2027: replay recorded gripper state (Batch 2, binary convention)", 2026-11-11)

This file is the submission README. The organizers asked for a policy file
with a detailed README; because the submission form's upload validator
rejects archive contents with code, data, and markdown extensions, this
single PDF carries the full README and points at the public repository
above for the runnable code. No other file is needed to evaluate the
policy: clone the repository at the commit above and follow §4.

## 1. What the policy does

Multi-demonstration trajectory following for the Galbot-Mobile dual-arm
chamber task in the official simulation (`scene04_1.usd`).

* At reset, the policy picks the recorded demonstration whose first-frame
  end-effector positions are nearest to the robot's observed initial
  positions. Selection happens only at reset, so a rollout never jumps
  between demonstrations mid-episode. Without an observation at reset the
  first demonstration is used.
* During the episode it tracks a phase index into the chosen demonstration
  and advances it monotonically. When the caller supplies the robot's
  current end-effector positions it re-anchors the phase to the
  demonstration point nearest the current pose within a forward window
  (closed-loop phase tracking); otherwise it advances one demonstration
  step per policy step.
* It commands the demonstration pose at `phase + lookahead` for both
  arms, as absolute differential-IK pose commands
  (`use_relative_mode=False`, the starter repository's own convention).
* On Batch 2 demonstrations it replays the recorded gripper state:
  `gripper_pos` is the Galbot gripper joint position (0.0 fully open,
  1.703 fully closed), thresholded at the range midpoint 0.8515 into
  the binary gripper action (`< 0` close, `>= 0` open), the starter
  repository's own `BinaryJointPositionAction` convention. Batch 1
  demonstrations carry no gripper channel, so on those the grippers
  are held at a fixed configurable command. Wheel velocity is
  commanded zero — see the honest limitation in §5.

The bundled set holds six real episodes: four Batch 1 (`demo_0`, `demo_2`,
`demo_7`, `demo_9`) and two Batch 2 (`demo_b2_0`, `demo_b2_3`), extracted
from six independent recordings of the official dataset
`rocochallenge2025/ecm2027_sim_data`. Episode lengths: 7,471 / 5,632 /
7,566 / 5,122 / 4,106 / 4,022 steps. Batch 2 files carry the split
observation terms `obs/eef_pos` + `obs/eef_quat` (composed into the same
`(T, 2, 7)` shape by the loader) plus a real gripper observation
`obs/gripper_pos` `(T, 2)`, stored in the `demo_b2_*` bundles.

## 2. Interfaces

Two adapters share the same core (`ecm_policy/core.py`, pure NumPy;
multi-demo selection in `ecm_policy/multidemo.py`):

1. **Docker policy service** (`ecm_policy/service.py`) — the planned
   official submission shape, implementing the starter README's service
   operations `health()`, `reset(task_info)`, `predict(observation)`,
   `close()`. Defaults to multi-demo mode over `data/`; `--trajectory
   <single .npz>` selects the old single-demo mode.
2. **In-process adapter** (`ecm_policy/adapter.py`) — builds a
   `MultiDemoCore` when `trajectory_dir` / `demo_dir` /
   `ECM_TRAJECTORY_DIR` is set or when `trajectory_path` points at a
   directory, for use with the starter repository's own evaluation
   harness.

A `Dockerfile` is included in the repository. The starter README still
lists the official Docker service transport and message schemas as
pending; the service follows the operation names the starter README
publishes.

## 3. Verification (all CPU-side; this host has no GPU)

* **56 / 56 tests pass** in the repository test suite, measured 2026-11-12
  (51 pass under the system Python; the 5 Batch 2 layout tests need h5py
  and pass under the data-sample venv Python), including:
  full-episode rollout integrity (finite 20-dim actions, unit-norm
  quaternions, replayed gripper state on Batch 2 demos and fixed gripper
  on Batch 1 demos, zero wheels throughout, monotonic phase
  to trajectory end, no mid-episode demo switch), multi-demo selection,
  adapter, service, and core tests.
* Offline lagged-follower evaluation (`ecm_policy/evaluate.py`, full
  table in `evaluation/results.md`): a simulated robot that moves a
  fraction `alpha` of the way toward each commanded pose per step.
  At `alpha = 1.0` the policy reproduces every bundled recording
  exactly (zero position error, zero phase lag); max phase lag strictly
  decreases as the follower gets faster on all 6 demos; gripper
  agreement on Batch 2 rises with `alpha` (32–42% at `alpha = 0.1`
  to 99.8–99.9% at `alpha = 1.0`). At `alpha` 0.1–0.5 the lagging
  follower cannot keep wall-clock pace (mean position error
  0.50–0.70 m) — reported as a baseline, not a score.
* The starter repository's 7 supplied lightweight regression tests also
  pass on this host (no simulator needed).
* **No score is claimed.** The policy has not been run in Isaac Sim on
  this host (no GPU; Isaac Sim 6.0.1 needs an RTX GPU) and has not been
  evaluated by the organizers. The numbers above are test-suite results
  only.

## 4. How the organizers run it

```bash
git clone https://github.com/ToniAdreal/ecm2027-chamber-policy.git
cd ecm2027-chamber-policy
git checkout d6f2af477af1fc72242c813f7309c3e0565486c6
pip install -r requirements.txt   # numpy only
python3 -m pytest tests/ -q      # expect: 51 passed, 5 skipped (the 5
                                 # Batch 2 layout tests need h5py)
docker build -t ecm2027-policy . # policy service image
```

Demonstration data is bundled under `data/` (`demo_*_ee_poses.npz`,
end-effector poses, shape `(T, 2, 7)` = `[x, y, z, qw, qx, qy, qz]` per
arm; the `demo_b2_*` bundles also carry the recorded `gripper_pos`
`(T, 2)`). The extraction tool is under `tools/`; Batch 2 data was
released 2026-11-10 and is integrated in this package (two episodes);
Batch 3 data (due 2026-11-28) can extend the set with the same tool.

## 5. Honest limitations

* **Gripper replay is built and tested; the mapping convention is
  unverified by the organizers.** Batch 1 recordings carry no gripper
  channel, verified across all four independent Batch 1 recordings
  (each file contains exactly one `data/demo_*/ee_poses` dataset and
  nothing else), so Batch 1 cores still hold the fixed command. Batch 2
  recordings carry `data/demo_*/obs/gripper_pos` `(T, 2)`, a real
  varying signal (0 to 1.703 in the two verified episodes), now consumed
  by the policy: full-episode open-loop rollouts replay the recorded
  gripper state exactly (demo_b2_0: 8 open/close transitions, 58.1%
  closed; demo_b2_3: 6 transitions, 68% closed). The mapping is the
  starter repository's own convention (`galbot.py`: gripper joint
  open/close positions 0.0 / 1.703; `action < 0` closes, `>= 0` opens),
  but the official evaluation service has never published its expected
  gripper action scale, so this remains a documented convention, not a
  validated organizer claim.
* **Fixed-layout assumption, weakened but not removed.** Multi-demo
  selection picks the closest recorded start layout instead of assuming
  one fixed demonstration, but replay still assumes the evaluation scene
  starts near a recorded layout — the same premise the starter
  repository's own replay tooling uses.
* **CPU-side development only.** All verification above ran without a
  simulator. Organizer-side evaluation is the intended validation step;
  the team welcomes the evaluation feedback the organizers send within
  5 working days and will iterate on it.
* **Resubmission.** The team understands resubmission is allowed before
  the policy submission deadline of **2026-12-30 12:00 SGT** and will
  resubmit an improved package (for example with Batch 3 demonstrations)
  if evaluation feedback calls for it.

## 6. Upstream issue filed

While setting up the starter repository the team found and reported a
missing-module defect in the official starter code:
https://github.com/rocochallenge/ecm2027_sim_track/issues/1
(`isaaclab_teleop` present in source but not installed; also missing
`mdp` exports). Open with no organizer reply as of 2026-11-04.

---

Team TwinTeleop (ECM-122) — Toni Adreal, toniadreal11@gmail.com.
Prepared 2026-11-06. Not yet uploaded to the submission form; the file
is ready for the team lead's go-ahead.
