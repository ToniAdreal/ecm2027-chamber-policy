# ECM 2027 — issue #8: randomization and scoring, from code and demo data

Upstream issue #8 asks which objects are randomized besides the
sponge pose, and how the lid lock and wipe milestones are scored.
Sources: the starter repo task code
(`/home/hatch/workspace/goals/ai-2/hidden_files/projects/ecm2027-sim-track/robofinals_tasks/chamber_service_task.py`) and the Batch 2 demo recordings'
`initial_state` root poses. The organizer mail of 2026-11-20
confirms the 13-point breakdown and the Docker submission route.

## Scoring: 13 points, 6 sequential stages

Stages gate on each other: a stage's points only count once every
earlier stage is complete (`_stage_points_done` + stage flags in
`_check_success`).

* Stage 1. Lift off chamber top: 1a_lid_lifted_1cm, 1b_lid_stable_open
* Stage 2. Extract large part: 2_part_lifted_1cm
* Stage 3. Pack part into box: 3_part_in_box_stable
* Stage 4. Wiping of the chamber: 4a_sponge_picked_up, 4b_wipe_wall_x_neg, 4b_wipe_wall_x_pos, 4b_wipe_wall_y_neg, 4b_wipe_wall_y_pos, 4b_wipe_floor
* Stage 5. Unbox & place replacement: 5a_new_part_lifted_1cm, 5b_new_part_placed_stable
* Stage 6. Close up chamber: 6_lid_closed_stable

## Lid lock: there is no such scored point

No entry of `POINT_NAMES` mentions a lock, and the word `lock`
appears in the task file only in the episode description string
("close and lock the chamber"). The lid is scored as three
ordinary points instead:

* `1a_lid_lifted_1cm`: lid front edge rises >= 0.01 m
  (`LID_LENGTH_M * sin(angle)`).
* `1b_lid_stable_open`: normalized lid joint >= 0.4 with angular
  rate <= 0.1 rad/s, held 1.0 s.
* `6_lid_closed_stable`: normalized lid joint <= 0.005 with
  angular rate <= 0.1 rad/s, held 1.0 s.

## Wipe milestones: 5 faces, 0.5 s contact each

One point per cavity face (wall_x_neg, wall_x_pos, wall_y_neg, wall_y_pos, floor), credited after 0.5 s of
continuous sponge-chamber contact while that face is the nearest
cavity face to the sponge centre (within 0.08 m), the sponge
centre is inside the cavity (height <= 0.30 m above the chamber
floor), and stages 1-3 plus sponge pick-up are already done.
The contact sensor only reports sponge-vs-chamber contact; the
face is inferred as the nearest cavity face, not sensed directly.

## Randomization: only the sponge (and, in code, the robot base)

Task code randomizes exactly two things per reset: the sponge
initial pose (base (0.37, -0.50, 0.85), +/-0.05 m uniform jitter
per axis, yaw uniform over [0, 2*pi)) and the robot base pose
(+/-0.10 m in x/y, +/-0.50 rad yaw around the anchor). The
chamber, both parts and the table are fixed scene geometry.

Measured across 4 unique initial
states (5 raw entries,
1 duplicate):

* `spongebody`: pos range [0.031955, 0.090149, 0.051614] m, yaw spread 2.8033 rad (n=4)
* `robot`: pos range [0.0, 0.0, 0.0] m, yaw spread 0.0 rad (n=4)
* `Chamber002`: pos range [0.0, 0.0, 0.0] m, yaw spread 0.0 rad (n=4)
* `Chamber002_01`: pos range [0.0, 0.0, 0.0] m, yaw spread 0.0 rad (n=4)
* `Table291`: pos range [0.0, 0.0, 0.0] m, yaw spread 0.0 rad (n=4)
* `Chamber001_01`: pos range [0.0, 0.0, 0.0] m, yaw spread 0.0 rad (n=4)

* Sponge x/y stay inside the code's +/-0.05 m jitter band;
  sponge yaw varies across episodes (uniform yaw in code).
* Sponge z spans 0.76-0.81 m against a spawn
  band of 0.80-0.90 m: the low end sits
  below the band, consistent with the recorded pose being
  post-settle on the table rather than the spawn pose.
* Robot base pose is BIT-IDENTICAL in every sampled initial
  state, although the task code samples +/-0.10 m / +/-0.50
  rad per reset. The demos do not exhibit that spread: either
  collection used a fixed anchor, or the recorded pose predates
  resampling. The replay-follow policy's zero-base-velocity
  assumption matches the demo distribution, not necessarily
  the eval distribution (see upstream issue #5).
* Chamber, both parts, table and lid joints are bit-identical
  across episodes: nothing else is randomized.

## What the demo data cannot say

Per-step recordings carry only `obs/eef_pos`, `obs/eef_quat`
and `obs/gripper_pos`. There is no lid angle, no per-step
object pose and no contact flag, so wipe faces and lid state
cannot be reconstructed from demo data. The scoring answers
above rest on the task code and the organizer mail, not on
the demos, and the submission keeps no score claim beyond
what the offline harness measures.
