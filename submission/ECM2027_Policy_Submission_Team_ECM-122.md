# Singapore Embodied AI Challenge for Manufacturing (ECM 2027) — Simulation Track Policy Submission

**Team:** TwinTeleop — **Team ID: ECM-122**
**Team lead:** Toni Adreal — toniadreal11@gmail.com
**Submission date:** 2026-11-24 (Beijing)
**Policy version:** v4.3 — multi-demo replay-follow (gripper replay + lagged-follower evaluation + lookahead sweep + observation-noise sweep + noise-robust phase tracking + official gRPC policy client + gripper-threshold sensitivity sweep + issue-#8 scoring analysis, re-derived from the upstream reset-anchor fix; submission defaults lookahead 2, phase advance cap 4, gripper threshold 0.8515; Batch 3 third demo integrated)

## Where the code lives

Full policy code, tests, and this document are public at:

**https://github.com/ToniAdreal/ecm2027-chamber-policy**

Commit submitted: `e29be4cf64a21196a9e0939f97bd18e984bb2528`
("ECM 2027 M18: re-derive issue #8 answers from upstream reset-anchor fix (17159e5)", 2026-11-24)

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
  step per policy step. The re-anchoring may move the phase at most 4
  demo steps per policy step (advance cap, §3): a noisy observation can
  otherwise yank the phase hundreds of steps forward in one step.
* It commands the demonstration pose at `phase + lookahead` for both
  arms, as absolute differential-IK pose commands
  (`use_relative_mode=False`, the starter repository's own convention).
* On Batch 2 and Batch 3 demonstrations it replays the recorded gripper state:
  `gripper_pos` is the Galbot gripper joint position (0.0 fully open,
  1.703 fully closed), thresholded at the range midpoint 0.8515 into
  the binary gripper action (`< 0` close, `>= 0` open), the starter
  repository's own `BinaryJointPositionAction` convention. Batch 1
  demonstrations carry no gripper channel, so on those the grippers
  are held at a fixed configurable command. Wheel velocity is
  commanded zero — see the honest limitation in §5.

The bundled set holds nine real episodes: four Batch 1 (`demo_0`, `demo_2`,
`demo_7`, `demo_9`), two Batch 2 (`demo_b2_0`, `demo_b2_3`), and three Batch 3
(`demo_b3_0`, `demo_b3_1`, `demo_b3_2`), extracted from nine independent recordings of the official
dataset `sgeaichallenge/ecm2027_sim_data` (formerly
`rocochallenge2025/ecm2027_sim_data`; the dataset moved to the
`sgeaichallenge` organization, last content change 2026-11-20). Episode lengths: 7,471 / 5,632 /
7,566 / 5,122 / 4,106 / 4,022 / 4,503 / 5,823 / 6,163 steps. Batch 2 files carry the split
observation terms `obs/eef_pos` + `obs/eef_quat` (composed into the same
`(T, 2, 7)` shape by the loader) plus a real gripper observation
`obs/gripper_pos` `(T, 2)`, stored in the `demo_b2_*` bundles. Batch 3
ships as one `.tar.gz` per recording (each holding a single
`galbot_success.hdf5`); it uses the same split layout plus a real gripper
observation, stored in the `demo_b3_0`, `demo_b3_1` and `demo_b3_2` bundles. Batch 3 was released
2026-11-17, eleven days before the announced 2026-11-28 date; all three Batch 3
tarballs are now downloaded, extracted, and integrated. (The Batch 3 listing on
the dataset page has since grown to 200 recordings, last checked 2026-11-24;
the dataset itself has had no new content since the Batch 3 release
2026-11-20, and this package integrates the third recording in sequence.)
One honest caveat: Batch 3's first frame is bit-identical to Batch 2's
rest pose (first-frame L2 difference 0.0 against `demo_b2_0`, `demo_b2_3`,
`demo_b3_0` and `demo_b3_1`), so nearest-start selection ties across the five and
resolves to the earliest demo per the documented tie rule; a Batch 3 demo
is selected only when the evaluation scene's start pose is nearer to it
than to the Batch 2 rest pose.

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
3. **Official gRPC policy client** (`ecm_policy/grpc_client.py`,
   new 2026-11-21) — speaks the participant interface the upstream
   starter repo published 2026-11-20 (`policy_v1.proto`, README
   "Participant policy interface"): the policy is a gRPC client
   connecting to the evaluator's server at `ECM27_EVAL_ENDPOINT`,
   exchanging `GetObservation` / `Step` calls with the preferred
   19-dimensional actions (arm target poses in `(x, y, z, w)` order,
   gripper negative-closes, zero base velocity). The 20-dimensional
   legacy core output is converted by dropping the wheel commands and
   appending a zero base velocity; quaternions are emitted in
   `(x, y, z, w)` order. The generated `policy_v1_pb2.py` is vendored
   from the upstream repo. A `Dockerfile.grpc` builds the policy image
   the interface asks for; the build is NOT verified on this host (no
   Docker here). The organizers published the Docker submission route
   by mail on 2026-11-20: the policy image is submitted through the
   official form (see §6).

A `Dockerfile` is included in the repository. The starter README now
publishes the official Docker service transport and message schemas
(2026-11-20); the service follows the operation names the starter README
publishes, and the gRPC client follows the published protocol.

## 3. Verification (all CPU-side; this host has no GPU)

* **114 / 114 tests pass** in the repository test suite, measured 2026-11-24
  in an isolated clone at the submitted commit (100 pass under the system
  Python; the 14 skipped tests need h5py — 5 Batch 2 layout tests and 9
  scoring-analysis tests — and pass under the data-sample venv Python,
  where the full suite runs 105/105 with the 9 gRPC tests skipped — that
  venv has h5py but no protobuf, and PyPI is unreachable from this host
  to install it; no single interpreter on this host runs all 114), including:
  full-episode rollout integrity (finite 20-dim actions, unit-norm
  quaternions, replayed gripper state on Batch 2 and Batch 3 demos and
  fixed gripper on Batch 1 demos, zero wheels throughout, monotonic phase
  to trajectory end, no mid-episode demo switch), multi-demo selection,
  adapter, service, core, and gripper-threshold tests.
* Official gRPC client tests (`tests/test_grpc_client.py`, 9 tests):
  a fake in-process evaluator replays real demo bundles as protocol
  observations through real protobuf serialization. A full 4,106-step
  episode of `demo_b2_0` completes with 0 action violations and 99.61%
  gripper agreement against the recording; a two-environment run (one
  env per demo) stays independent with per-env reset on termination;
  `FAILED_PRECONDITION` on `Step` recovers via re-`GetObservation`;
  transient `UNAVAILABLE` on `GetObservation` retries; the run is
  deterministic across repeats. The socket layer itself (`grpcio`) is
  NOT tested here: `grpcio` is not installable on this host (PyPI
  unreachable behind the proxy), so the tests drive a fake channel and
  the real `grpc.insecure_channel` path is untested. The frame
  assumption is documented in the module: recorded EE poses are passed
  through as-is with zero base velocity, and whether they coincide with
  the evaluator's robot-root frame depends on the spawn pose (upstream
  issue #5, still open).
* Offline lagged-follower evaluation (`ecm_policy/evaluate.py`, full
  table in `evaluation/results.md`): a simulated robot that moves a
  fraction `alpha` of the way toward each commanded pose per step.
  At `alpha = 1.0` the policy reproduces every bundled recording
  exactly (zero position error, zero phase lag); max phase lag strictly
  decreases as the follower gets faster on all 9 demos; gripper
  agreement on Batch 2 and Batch 3 rises with `alpha` (32–42% at
  `alpha = 0.1` to 99.8–99.9% at `alpha = 1.0`). At `alpha` 0.1–0.5 the
  lagging follower cannot keep wall-clock pace (mean position error
  0.50–0.70 m) — reported as a baseline, not a score.
* Lookahead sweep (`tools/sweep_lookahead.py`, full table in
  `evaluation/lookahead_sweep.md`): every bundled demo at `alpha` 0.3 /
  0.5 for lookaheads 0 / 1 / 2 / 4 / 8. Lookahead 2 is the submission
  default: it beats lookahead 1 at both swept alphas on demo-averaged
  mean position error (0.609 vs 0.618 m at `alpha = 0.3`; 0.010 vs
  0.618 m at `alpha = 0.5`), while lookahead 4 wins only at 0.3 and
  lookahead 0 stalls the phase tracker outright (final phase 0). The
  near-perfect `alpha = 0.5` / lookahead-2 result is a property of the
  harness's first-order lag model, not a measurement of real Isaac Sim
  dynamics, and is reported as such.
* Observation-noise sweep (`tools/sweep_noise.py`, full table in
  `evaluation/noise_sweep.md`): every bundled demo at `alpha` 0.3 /
  0.5 with lookahead 2, for Gaussian noise on the EE positions the
  policy observes (sigma 0 / 1 mm / 5 mm / 1 cm / 2 cm). The phase
  tracker is noise-fragile: at `alpha = 0.5`, 1 mm of noise breaks the
  near-perfect noiseless tracking (mean position error 0.010 -> 0.261
  m), and at 5 mm and above the tracker systematically outruns the
  recording (mean phase lag -918 steps at 5 mm, -1,948 at 2 cm) while
  gripper agreement on Batch 2 falls to 31.9%. The mechanism is the
  nearest-point phase search inside its 250-step forward window
  combined with monotonic phase: a noisy observation can make a later
  demo pose the nearest one, jumping phase forward, and the jump
  compounds. At `alpha = 0.3` the same jumps partially compensate lag
  at 5 mm (mean phase lag 1,117 -> 32 steps) but overshoot at 2 cm.
  Noise was added only to the observation, never to the true robot
  state or the recorded trajectory. The real evaluator's observation
  noise is unknown, so this bounds the sensitivity; noise-robust phase
  tracking is the next step, not a claim made here.
* Noise-robust phase-tracking sweep (`tools/sweep_robust.py`, full table
  in `evaluation/robust_sweep.md`): six tracker configs (uncapped
  baseline, advance caps 4 / 8 / 16, window 32, window 64 + cap 8) at
  `alpha` 0.3 / 0.5 with lookahead 2, for Gaussian observation noise
  (sigma 0 / 1 mm / 5 mm / 1 cm). The baseline column reproduces the M9
  numbers exactly. Advance cap 4 is the new submission default: it wins
  6 of 8 (alpha, sigma) cells on demo-averaged mean position error, it
  costs nothing with no noise (identical numbers at sigma 0), and it
  lifts Batch 2 gripper agreement at `alpha = 0.5` at every noise level
  (sigma 1 mm: 84.3% -> 93.4%; 5 mm: 55.2% -> 76.0%; 1 cm: 43.6% ->
  60.0%). The one exception is `alpha = 0.3` / sigma 5 mm, where the
  uncapped tracker wins (0.391 vs 0.593 m) because noisy forward jumps
  partly compensate the lag — a property of the harness's first-order
  lag model, not a real-evaluator result, and reported as such.
  Smaller windows alone help less than the cap, and cap 8 / cap 16 sit
  between cap 4 and the baseline at most noise levels.
* Batch 3 re-run of the same sweep (`tools/sweep_robust.py`, full table
  in `evaluation/robust_sweep_b3.md`, 2026-11-17): the six configs above
  re-swept over all 7 bundled demos (336 rollouts). Every one of the
  288 per-demo rows for the original six demos reproduces the M10
  numbers exactly (max difference 0.0); only the 7-demo averages move,
  because `demo_b3_0` joins them. Advance cap 4 still holds as the
  submission default: it wins 5 of 8 cells on 7-demo-averaged mean
  position error, costs nothing with no noise, and lifts gripper
  agreement at `alpha = 0.5` at every noise level (1 mm: 88.7% ->
  94.9%; 5 mm: 57.6% -> 72.8%; 1 cm: 48.4% -> 60.4%). The three cells
  it does not win: `alpha = 0.3` / sigma 5 mm, where the uncapped
  baseline wins (0.418 vs 0.589 m, the same lag-compensation mechanism
  as above); `alpha = 0.3` / sigma 1 mm, where cap 16 wins by 0.0009 m
  (a tie in practice); and `alpha = 0.5` / sigma 0, where baseline and
  cap 4 are numerically identical (0.0101 m). `demo_b3_0` alone behaves
  like the other demos: noiseless `alpha = 0.5` mean position error
  0.0108 m with 99.9% gripper agreement; at `alpha = 0.3` it lags like
  the rest (mean position error 0.666 m, mean phase lag 888 steps).
* Batch 3 second-demo re-run of the same sweep (`tools/sweep_robust.py`, full table
  in `evaluation/robust_sweep_b3_1.md`, 2026-11-18): the six configs above
  re-swept over all 8 bundled demos (384 rollouts). Every one of the
  336 per-demo rows for the original seven demos reproduces the M11
  numbers exactly (max difference 0.0); only the 8-demo averages move,
  because `demo_b3_1` joins them. Advance cap 4 still holds as the
  submission default: it wins 5 of 8 cells on 8-demo-averaged mean
  position error, costs nothing with no noise, and lifts gripper
  agreement at `alpha = 0.5` at every noise level (1 mm: 85.8% ->
  91.1%; 5 mm: 56.3% -> 69.4%; 1 cm: 48.6% -> 58.7%). The three cells
  it does not win: `alpha = 0.3` / sigma 5 mm, where the uncapped
  baseline wins (0.4155 vs 0.5666 m, the same lag-compensation mechanism
  as above); `alpha = 0.3` / sigma 1 mm, where cap 16 wins by 0.0014 m
  (a tie in practice); and `alpha = 0.5` / sigma 0, where baseline and
  cap 4 are numerically identical (0.0097 m). `demo_b3_1` alone behaves
  like the other demos: noiseless `alpha = 0.5` mean position error
  0.0065 m with 99.0% gripper agreement; at `alpha = 0.3` it lags like
  the rest (mean position error 0.444 m, mean phase lag 1,144 steps).
* Batch 3 third-demo re-run of the same sweep (`tools/sweep_robust.py`, full table
  in `evaluation/robust_sweep_b3_2.md`, 2026-11-20): the six configs above
  re-swept over all 9 bundled demos (432 rollouts). Every one of the
  384 per-demo rows for the original eight demos reproduces the M12
  numbers to float precision (max difference 1.4e-14); only the 9-demo averages move,
  because `demo_b3_2` joins them. Advance cap 4 still holds as the
  submission default: it wins 6 of 8 cells on 9-demo-averaged mean
  position error, costs nothing with no noise, and lifts gripper
  agreement at `alpha = 0.5` at every noise level (1 mm: 83.3% ->
  88.5%; 5 mm: 54.6% -> 66.9%; 1 cm: 48.9% -> 57.0%). The two cells
  it does not win: `alpha = 0.3` / sigma 5 mm, where the uncapped
  baseline wins (0.4161 vs 0.5571 m, the same lag-compensation mechanism
  as above); and `alpha = 0.3` / sigma 1 mm, where cap 16 wins by 0.0015 m
  (a tie in practice). At `alpha = 0.5` / sigma 0, cap 4 and baseline are
  numerically identical (0.0103 vs 0.0104 m). `demo_b3_2` alone behaves
  like the other demos: noiseless `alpha = 0.5` mean position error
  0.0154 m with 97.8% gripper agreement; at `alpha = 0.3` it lags like
  the rest (mean position error 0.539 m, mean phase lag 1,215 steps,
  gripper agreement 48.0%).
* Gripper-threshold sensitivity sweep (`tools/sweep_gripper_threshold.py`, full table
  in `evaluation/gripper_threshold_sweep.md`, 2026-11-21): five policy
  replay thresholds (0.2 / 0.425 / 0.8515 / 1.28 / 1.5) at `alpha` 0.3 /
  0.5 with lookahead 2 and advance cap 4, for Gaussian observation noise
  (sigma 0 / 1 mm / 5 mm / 1 cm), over all 9 bundled demos (360
  rollouts). The evaluator's ground-truth recorded-closed labels always
  use the joint-range midpoint 0.8515; only the policy's threshold
  varies. The midpoint column reproduces the M14 cap-4 numbers exactly
  (max difference 2.1e-14). Thresholds 0.2 and 0.425 match the midpoint
  within ~1 percentage point of gripper agreement in every cell, while
  1.28 and 1.5 collapse it by 20–45 points (e.g. noiseless `alpha = 0.5`:
  99.2% at the midpoint vs 58.3% at 1.28 and 51.8% at 1.5) — the
  recorded gripper signal is strongly bimodal, so any threshold in the
  low half of the joint range lands in the sparse transition band and
  performs the same, and only a transition point above ~1.1 would cost
  agreement. The midpoint stays the submission default: it is the
  documented convention and the best noiseless performer. Position
  error is exactly identical across thresholds in every cell, as
  expected (the gripper action does not feed the pose tracker).
* Issue-#8 scoring analysis (`tools/analyze_scoring.py`, full table in
  `evaluation/scoring_analysis.md`, 2026-11-23, re-derived 2026-11-24
  from the upstream reset-anchor fix `17159e5`): the 13-point scoring
  spec is read from the starter repo's task code, and the demo data's
  `initial_state` root poses are measured against the code's
  randomization claims. Only the sponge pose is randomized in the data
  (position jitter within the code's +/-0.05 m band in x/y, yaw spread
  2.80 rad; recorded z 0.76-0.81 m straddles the spawn band's low edge,
  i.e. the pose is post-settle on the table). The robot base pose is
  bit-identical in every sampled initial state, at exactly the anchor
  the task code now uses for the eval reset (0.40, -0.85, yaw pi/2):
  upstream `17159e5` (2026-11-23) samples the reset only +/-0.10 m /
  +/-10 deg around that anchor, so the demos sit at the centre of the
  eval distribution instead of ~0.3 m away from it (the old
  shared-helper behaviour, upstream issue #5). The residual mismatch
  is bounded: at eval the base can sit up to 0.10 m and 10 deg away
  from the recorded pose, so a replayed demo can be off by up to
  ~0.10 m in translation plus the yaw-induced end-effector offset;
  the policy still commands zero base velocity. The table stays a
  movable body as in data collection (`KEEP_TABLE_STATIC = False`),
  and the eval default physics device is now cpu (upstream `81eccab`).
  Chamber, both parts, table and lid joints are bit-identical: nothing else is
  randomized. There is no scored lid-lock point: the lid is scored as
  lift (1a), stable-open (1b) and stable-closed (6) only, and the word
  "lock" appears in the task file only in the episode description
  string. Each wipe face is worth 1 point after 0.5 s of continuous
  sponge-chamber contact while that face is the nearest cavity face to
  the sponge centre (within 0.08 m, inside the cavity). Per-step demo
  recordings carry no lid angle, object pose or contact flag, so the
  wipe and lid scoring rules cannot be reconstructed from demo data;
  the answers rest on the task code and the organizer mail.
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
git checkout e29be4cf64a21196a9e0939f97bd18e984bb2528
pip install -r requirements.txt   # numpy only
python3 -m pytest tests/ -q      # expect: 100 passed, 14 skipped (the 14
                                 # need h5py: 5 Batch 2 layout + 9
                                 # scoring-analysis tests; with h5py
                                 # installed: 105 passed, 9 skipped —
                                 # the 9 gRPC tests need protobuf)
docker build -t ecm2027-policy . # policy service image
docker build -f Dockerfile.grpc -t ecm27-policy:latest . # official
                                 # gRPC policy image (build NOT verified
                                 # on the dev host: no Docker there)
```

Demonstration data is bundled under `data/` (`demo_*_ee_poses.npz`,
end-effector poses, shape `(T, 2, 7)` = `[x, y, z, qw, qx, qy, qz]` per
arm; the `demo_b2_*` and `demo_b3_*` bundles also carry the recorded
`gripper_pos` `(T, 2)`). The extraction tool is under `tools/`; Batch 2
data was released 2026-11-10 and Batch 3 data was released 2026-11-17
(all three episodes integrated in this package). The organizer dataset
has since grown to 800 training episodes (mail of 2026-11-20); the
bundled 9 demos are a subset, not the full set.

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
  closed; demo_b2_3: 6 transitions, 68% closed; demo_b3_0: 14
  transitions per arm, 60.4% closed; demo_b3_1: 24 transitions per
  arm, 65.5% closed, both arms in agreement on every step; demo_b3_2: 20
  transitions per arm, 51.0% closed, both arms in agreement on every step). The mapping is the
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
* **Resubmission.** The organizers allow up to five submissions in
  total including resubmissions, before the policy submission deadline
  of **2026-12-30 12:00 SGT**, and email evaluation results within five
  working days of each submission. The team will resubmit an improved
  package if evaluation feedback calls for it.

## 6. Upstream issues filed and answered

While setting up the starter repository the team found and reported a
missing-module defect in the official starter code:
https://github.com/rocochallenge/ecm2027_sim_track/issues/1
(`isaaclab_teleop` present in source but not installed; also missing
`mdp` exports). Still open with no organizer reply as of 2026-11-24.
A second upstream issue appeared 2026-11-13 and is noted for the
organizers, not filed by this team:
https://github.com/rocochallenge/ecm2027_sim_track/issues/7
(sponge spawn z jitter can start the sponge inside the table). On
2026-11-21 the organizers committed a fix for that issue's root cause
(`667517b`: sponge spawn height raised from 0.78 to 0.85 m in
`robofinals_tasks/chamber_service_task.py`), followed the same day by
`07d962a` (keep the service table static on GPU); the issue itself is
still open with no organizer comment as of 2026-11-24, so it is not
treated as closed. On 2026-11-23 the organizers committed two more
fixes in the same file: `81eccab` (default eval physics device to cpu)
and `17159e5` (reset the robot at the demonstration base pose: anchor
(0.40, -0.85), yaw 90 deg, with only +/-0.10 m / +/-10 deg sampled
around it, replacing the shared helper that moved the robot ~0.3 m
away from the anchor — this answers the distribution question behind
issues #5 and #8 from the code side; the table-static fix is kept but
disabled by default, `KEEP_TABLE_STATIC = False`). On
2026-11-20 the organizers merged the participant policy interface and
evaluation into main (commits `a1dff75` / `aa1255e`: `policy_v1.proto`,
README "Participant policy interface", example Docker policy,
13-point chamber scoring — open chamber 2, extract old part 1, place
in box 1, sponge + wipe five faces 6, replacement part 2, close
chamber 1). The same day the organizers mailed all registered teams:
the simulation dataset now holds 800 training episodes, the updated
evaluation platform is live, and the Docker policy image is submitted
through the official form at
https://form.gov.sg/6ae35050aafe2da79edea132 (up to five submissions
per team including resubmissions; results emailed within five working
days, with a report where applicable). The image delivery channel is
therefore published; this PDF remains the human-readable policy
description, while the Docker image is the submission artifact (its
build is still unverified on the dev host, which has no Docker, and
upload still awaits the team lead's go-ahead). Issues #4 (whether
scripted / hybrid policies are allowed, and the end-effector pose
frame), #6 (ranked metric, seeds, submission format before the Docker
spec) and #8 (which objects are randomized, how the lid-lock and wipe
milestones are scored) are still open with no organizer reply as of
2026-11-24. Issue #8 is now answered from primary sources by this
team in M17 and re-derived in M18 (`tools/analyze_scoring.py`,
`evaluation/scoring_analysis.md`): only the sponge pose is randomized
in the demo data, the lid has no scored lock point, and each wipe face
is worth 1 point after 0.5 s of sustained sponge-chamber contact —
and since upstream `17159e5` the eval robot reset is anchored at
exactly the demo base pose, so the demos sit at the centre of the
eval distribution; the residual mismatch is bounded at 0.10 m / 10 deg.
But the answer is the team's reading of the task code, not an
organizer confirmation, and the issue stays open until the organizers
say so.

---

Team TwinTeleop (ECM-122) — Toni Adreal, toniadreal11@gmail.com.
Prepared 2026-11-24. Not yet uploaded to the submission form; the file
is ready for the team lead's go-ahead.
