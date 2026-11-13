# ECM 2027 policy — lagged-follower evaluation

Offline harness (`ecm_policy/evaluate.py`): the simulated robot moves a fraction `alpha` of the way toward each commanded pose per step; the policy observes the robot's actual EE positions and re-anchors its phase. Errors are scored against the recorded demonstration at the same step index. Deterministic, no GPU, no simulator.

| demo | alpha | steps | mean pos err (m) | max pos err (m) | mean ori err (deg) | max phase lag | gripper agree |
|---|---|---|---|---|---|---|---|
| demo_0 | 0.1 | 7471 | 0.5234 | 1.2470 | 46.34 | 6780 | n/a (no channel) |
| demo_0 | 0.3 | 7471 | 0.5784 | 1.9058 | 44.58 | 5265 | n/a (no channel) |
| demo_0 | 0.5 | 7471 | 0.6173 | 1.8580 | 42.57 | 3730 | n/a (no channel) |
| demo_0 | 1.0 | 7471 | 0.0000 | 0.0000 | 0.00 | 0 | n/a (no channel) |
| demo_2 | 0.1 | 5632 | 0.5953 | 1.1279 | 58.65 | 5110 | n/a (no channel) |
| demo_2 | 0.3 | 5632 | 0.6424 | 1.5273 | 51.84 | 3960 | n/a (no channel) |
| demo_2 | 0.5 | 5632 | 0.6194 | 1.5351 | 71.99 | 2814 | n/a (no channel) |
| demo_2 | 1.0 | 5632 | 0.0000 | 0.0000 | 0.00 | 0 | n/a (no channel) |
| demo_7 | 0.1 | 7566 | 0.6733 | 1.5797 | 48.60 | 6864 | n/a (no channel) |
| demo_7 | 0.3 | 7566 | 0.5317 | 1.7530 | 48.35 | 5325 | n/a (no channel) |
| demo_7 | 0.5 | 7566 | 0.5473 | 1.5644 | 54.46 | 3779 | n/a (no channel) |
| demo_7 | 1.0 | 7566 | 0.0000 | 0.0000 | 0.00 | 0 | n/a (no channel) |
| demo_9 | 0.1 | 5122 | 0.6522 | 1.2890 | 55.77 | 4647 | n/a (no channel) |
| demo_9 | 0.3 | 5122 | 0.6022 | 1.7136 | 51.57 | 3601 | n/a (no channel) |
| demo_9 | 0.5 | 5122 | 0.6003 | 1.4162 | 51.32 | 2558 | n/a (no channel) |
| demo_9 | 1.0 | 5122 | 0.0000 | 0.0000 | 0.00 | 0 | n/a (no channel) |
| demo_b2_0 | 0.1 | 4106 | 0.6367 | 1.2552 | 41.08 | 3727 | 41.9% |
| demo_b2_0 | 0.3 | 4106 | 0.6529 | 1.5904 | 37.95 | 2896 | 53.3% |
| demo_b2_0 | 0.5 | 4106 | 0.6833 | 1.3324 | 36.94 | 2052 | 61.4% |
| demo_b2_0 | 1.0 | 4106 | 0.0000 | 0.0000 | 0.00 | 0 | 99.8% |
| demo_b2_3 | 0.1 | 4022 | 0.4976 | 1.0357 | 57.94 | 3652 | 32.0% |
| demo_b2_3 | 0.3 | 4022 | 0.6985 | 1.5256 | 45.78 | 2837 | 51.3% |
| demo_b2_3 | 0.5 | 4022 | 0.6422 | 1.3266 | 47.11 | 2010 | 68.0% |
| demo_b2_3 | 1.0 | 4022 | 0.0000 | 0.0000 | 0.00 | 0 | 99.9% |
