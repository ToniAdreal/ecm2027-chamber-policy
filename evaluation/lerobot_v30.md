# LeRobot v30 cross-check (ECM 2027, M20)

Dataset `sgeaichallenge/ecm2027_sim_data`, lastModified 2026-11-26T21:37:35.000Z: 5200 files, 2984 LeRobot files, 373 episode dirs ({'batch1': 173, 'batch3': 200}), no batch2; 200 batch3 HDF5 tarballs (unchanged count).

Episode batch1/ChamberServiceTask_1793172869923601: robot Galbot-Mobile, codebase v3.0, 30 fps, 7023 frames. Features: observation.state float32[77], action float32[20], observation.images.first_person_camera_rgb video[480, 640, 3], observation.images.left_hand_camera_rgb video[224, 224, 3], observation.images.right_hand_camera_rgb video[224, 224, 3], timestamp float32[1], frame_index int64[1], episode_index int64[1], index int64[1], task_index int64[1]
stats.json reproduces exactly for observation.state: True (min diff 0.0, max diff 0.0).
stats.json reproduces exactly for action: True (min diff 0.0, max diff 0.0).

Action is 20-dim, state 77-dim; both quaternions unit-norm on all 7023 frames (left 0.999999965-1.000000090, right 0.999999960-1.000000111); layout [pos(3)+quat(4)] x 2 arms + gripper(2) + base(4) matches our 20-dim legacy action. Gripper left values [1.0] (constant), right [-1.0, 1.0]; base mean ~0, std [0.226, 0.223, 0.31, 0.329]. Timestamp dt median 0.033333 s (30 Hz); frame_index contiguous: True.
State: 30 of 77 columns behave like joint angles (|x| > 3, up to ~2*pi); state semantics otherwise unresolved (info.json names are null). Quat component order: unresolved (wxyz vs xyzw not distinguishable from data).

Rest-pose comparison (LeRobot first frame vs our 9 demo bundles, L2 on EE position):
- demo_0_ee_poses (7471 steps): left 0.8406 m, right 0.4088 m.
- demo_2_ee_poses (5632 steps): left 0.8406 m, right 0.4088 m.
- demo_7_ee_poses (7566 steps): left 0.8406 m, right 0.4088 m.
- demo_9_ee_poses (5122 steps): left 0.8405 m, right 0.4087 m.
- demo_b2_0_ee_poses (4106 steps): left 0.8392 m, right 0.4055 m.
- demo_b2_3_ee_poses (4022 steps): left 0.8392 m, right 0.4055 m.
- demo_b3_0_ee_poses (4503 steps): left 0.8392 m, right 0.4055 m.
- demo_b3_1_ee_poses (5823 steps): left 0.8392 m, right 0.4055 m.
- demo_b3_2_ee_poses (6163 steps): left 0.8392 m, right 0.4055 m.

No bundle matches: the LeRobot episode is a different recording (different anchor) from all 9 integrated demos, so LeRobot episodes cannot be dropped into our bundles as-is; the action layout does match our 20-dim legacy action, so a LeRobot-fed replay-follow variant is feasible once the quat order is pinned.
