# Singapore Embodied AI Challenge for Manufacturing (ECM 2027) — Simulation Track Policy Submission

**Team:** TwinTeleop — **Team ID: ECM-122**
**Team lead:** Toni Adreal — toniadreal11@gmail.com
**Submission date:** 2026-11-18 (Beijing)
**Policy version:** v3.8 — multi-demo replay-follow (gripper replay + lagged-follower evaluation + lookahead sweep + observation-noise sweep + noise-robust phase tracking; submission defaults lookahead 2, phase advance cap 4; Batch 3 second demo integrated)

## Where the code lives

Full policy code, tests, and this document are public at:

**https://github.com/ToniAdreal/ecm2027-chamber-policy**

Commit submitted: `b796d013a6342d9cd56272da107d24c2d4d014eb`
("ECM 2027 M12: Batch 3 second demo integrated (demo_b3_1, 5823 steps, split layout + gripper); robust sweep re-run on 8 demos, 384 rollouts, cap4 default holds; 80 tests green", 2026-11-18)

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

The bundled set holds eight real episodes: four Batch 1 (`demo_0`, `demo_2`,
`demo_7`, `demo_9`), two Batch 2 (`demo_b2_0`, `demo_b2_3`), and two Batch 3
(`demo_b3_0`, `demo_b3_1`), extracted from eight independent recordings of the official
dataset `rocochallenge2025/ecm2027_sim_data`. Episode lengths: 7,471 / 5,632 /
7,566 / 5,122 / 4,106 / 4,022 / 4,503 / 5,823 steps. Batch 2 files carry the split
observation terms `obs/eef_pos` + `obs/eef_quat` (composed into the same
`(T, 2, 7)` shape by the loader) plus a real gripper observation
`obs/gripper_pos` `(T, 2)`, stored in the `demo_b2_*` bundles. Batch 3
ships as one `.tar.gz` per recording (each holding a single
`galbot_success.hdf5`); it uses the same split layout plus a real gripper
observation, stored in the `demo_b3_0` and `demo_b3_1` bundles. Batch 3 was released
2026-11-17, eleven days before the announced 2026-11-28 date; both Batch 3
tarballs are now downloaded, extracted, and integrated.
One honest caveat: Batch 3's first frame is bit-identical to Batch 2's
rest pose (first-frame L2 difference 0.0 against `demo_b2_0`, `demo_b2_3`,
and `demo_b3_0`), so nearest-start selection ties across the four and
resolves to the earliest demo per the documented tie rule — neither
`demo_b3_0` nor `demo_b3_1`
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

A `Dockerfile` is included in the repository. The starter README still
lists the official Docker service transport and message schemas as
pending; the service follows the operation names the starter README
publishes.

## 3. Verification (all CPU-side; this host has no GPU)

* **80 / 80 tests pass** in the repository test suite, measured 2026-11-18
  in an isolated clone at the submitted commit (75 pass under the system
  Python; the 5 Batch 2 layout tests need h5py and pass under the
  data-sample venv Python, where the full suite runs 80/80), including:
  full-episode rollout integrity (finite 20-dim actions, unit-norm
  quaternions, replayed gripper state on Batch 2 and Batch 3 demos and
  fixed gripper on Batch 1 demos, zero wheels throughout, monotonic phase
  to trajectory end, no mid-episode demo switch), multi-demo selection,
  adapter, service, and core tests.
* Offline lagged-follower evaluation (`ecm_policy/evaluate.py`, full
  table in `evaluation/results.md`): a simulated robot that moves a
  fraction `alpha` of the way toward each commanded pose per step.
  At `alpha = 1.0` the policy reproduces every bundled recording
  exactly (zero position error, zero phase lag); max phase lag strictly
  decreases as the follower gets faster on all 8 demos; gripper
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
git checkout b796d013a6342d9cd56272da107d24c2d4d014eb
pip install -r requirements.txt   # numpy only
python3 -m pytest tests/ -q      # expect: 75 passed, 5 skipped (the 5
                                 # Batch 2 layout tests need h5py;
                                 # with h5py installed: 80 passed)
docker build -t ecm2027-policy . # policy service image
```

Demonstration data is bundled under `data/` (`demo_*_ee_poses.npz`,
end-effector poses, shape `(T, 2, 7)` = `[x, y, z, qw, qx, qy, qz]` per
arm; the `demo_b2_*` and `demo_b3_*` bundles also carry the recorded
`gripper_pos` `(T, 2)`). The extraction tool is under `tools/`; Batch 2
data was released 2026-11-10 and Batch 3 data was released 2026-11-17
(both episodes integrated in this package).

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
  arm, 65.5% closed, both arms in agreement on every step). The mapping is the
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
`mdp` exports). Still open with no organizer reply as of 2026-11-17.
A second upstream issue appeared 2026-11-13 and is noted for the
organizers, not filed by this team:
https://github.com/rocochallenge/ecm2027_sim_track/issues/7
(sponge spawn z jitter can start the sponge inside the table). The
starter README still lists the official Docker service transport and
message schemas as pending (last verified 2026-11-04; upstream commits
could not be re-checked 2026-11-17 because the GitHub API rate limit
was exhausted for unauthenticated requests).

---

Team TwinTeleop (ECM-122) — Toni Adreal, toniadreal11@gmail.com.
Prepared 2026-11-18. Not yet uploaded to the submission form; the file
is ready for the team lead's go-ahead.
