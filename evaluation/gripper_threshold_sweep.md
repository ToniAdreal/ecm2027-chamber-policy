# ECM 2027 policy — gripper replay threshold sensitivity sweep (lagged-follower harness)

Every bundled demo at alpha 0.3 / 0.5 with the submission-default lookahead 2 and phase advance cap 4, for Gaussian observation noise on the EE positions the policy sees (sigma 0 / 1 mm / 5 mm / 1 cm, seed 20261114). The evaluator's ground-truth recorded-closed labels always use the joint-range midpoint 0.8515; only the policy's replay threshold varies (0.2, 0.425, 0.8515, 1.28, 1.5). Gripper agreement averages over the 5 demos carrying a gripper channel; position error averages over all 9 demos. Deterministic, no GPU, no simulator.

| policy threshold | alpha | noise sigma (m) | demos | gripper demos | mean gripper agree | mean pos err (m) |
|---|---|---|---|---|---|---|
| 0.2 | 0.3 | 0.0 | 9 | 5 | 64.9% | 0.5890 |
| 0.2 | 0.3 | 0.001 | 9 | 5 | 63.7% | 0.5867 |
| 0.2 | 0.3 | 0.005 | 9 | 5 | 62.4% | 0.5571 |
| 0.2 | 0.3 | 0.01 | 9 | 5 | 68.2% | 0.3752 |
| 0.2 | 0.5 | 0.0 | 9 | 5 | 97.8% | 0.0103 |
| 0.2 | 0.5 | 0.001 | 9 | 5 | 88.4% | 0.1567 |
| 0.2 | 0.5 | 0.005 | 9 | 5 | 67.0% | 0.4894 |
| 0.2 | 0.5 | 0.01 | 9 | 5 | 57.4% | 0.5995 |
| 0.425 | 0.3 | 0.0 | 9 | 5 | 64.7% | 0.5890 |
| 0.425 | 0.3 | 0.001 | 9 | 5 | 63.6% | 0.5867 |
| 0.425 | 0.3 | 0.005 | 9 | 5 | 62.3% | 0.5571 |
| 0.425 | 0.3 | 0.01 | 9 | 5 | 68.0% | 0.3752 |
| 0.425 | 0.5 | 0.0 | 9 | 5 | 98.4% | 0.0103 |
| 0.425 | 0.5 | 0.001 | 9 | 5 | 88.5% | 0.1567 |
| 0.425 | 0.5 | 0.005 | 9 | 5 | 66.9% | 0.4894 |
| 0.425 | 0.5 | 0.01 | 9 | 5 | 57.2% | 0.5995 |
| 0.8515 | 0.3 | 0.0 | 9 | 5 | 64.3% | 0.5890 |
| 0.8515 | 0.3 | 0.001 | 9 | 5 | 63.2% | 0.5867 |
| 0.8515 | 0.3 | 0.005 | 9 | 5 | 62.1% | 0.5571 |
| 0.8515 | 0.3 | 0.01 | 9 | 5 | 67.7% | 0.3752 |
| 0.8515 | 0.5 | 0.0 | 9 | 5 | 99.2% | 0.0103 |
| 0.8515 | 0.5 | 0.001 | 9 | 5 | 88.5% | 0.1567 |
| 0.8515 | 0.5 | 0.005 | 9 | 5 | 66.9% | 0.4894 |
| 0.8515 | 0.5 | 0.01 | 9 | 5 | 57.0% | 0.5995 |
| 1.28 | 0.3 | 0.0 | 9 | 5 | 44.0% | 0.5890 |
| 1.28 | 0.3 | 0.001 | 9 | 5 | 44.0% | 0.5867 |
| 1.28 | 0.3 | 0.005 | 9 | 5 | 38.6% | 0.5571 |
| 1.28 | 0.3 | 0.01 | 9 | 5 | 44.0% | 0.3752 |
| 1.28 | 0.5 | 0.0 | 9 | 5 | 58.3% | 0.0103 |
| 1.28 | 0.5 | 0.001 | 9 | 5 | 53.4% | 0.1567 |
| 1.28 | 0.5 | 0.005 | 9 | 5 | 50.4% | 0.4894 |
| 1.28 | 0.5 | 0.01 | 9 | 5 | 44.1% | 0.5995 |
| 1.5 | 0.3 | 0.0 | 9 | 5 | 45.1% | 0.5890 |
| 1.5 | 0.3 | 0.001 | 9 | 5 | 45.0% | 0.5867 |
| 1.5 | 0.3 | 0.005 | 9 | 5 | 36.3% | 0.5571 |
| 1.5 | 0.3 | 0.01 | 9 | 5 | 39.5% | 0.3752 |
| 1.5 | 0.5 | 0.0 | 9 | 5 | 51.8% | 0.0103 |
| 1.5 | 0.5 | 0.001 | 9 | 5 | 48.1% | 0.1567 |
| 1.5 | 0.5 | 0.005 | 9 | 5 | 45.7% | 0.4894 |
| 1.5 | 0.5 | 0.01 | 9 | 5 | 40.4% | 0.5995 |

Reading the table: the 0.8515 (midpoint) column is the submission
default and reproduces the M10 robust-sweep gripper numbers. Any
other threshold's agreement is the cost of a misplaced transition
point against midpoint-labelled ground truth. Position error is
expected to be identical across thresholds (the gripper action
does not feed the pose tracker); a difference there would be a bug.

