# ECM 2027 policy — observation-noise sweep (lagged-follower harness)

Every bundled demo at alpha 0.3 / 0.5 with the submission-default lookahead 2, for Gaussian observation noise on the EE positions the policy sees (sigma 0 / 1 mm / 5 mm / 1 cm / 2 cm, seed 20261114). Errors are scored against the recorded demonstration at the same step index using the TRUE robot pose; the table below averages each metric over the 6 demos. Deterministic, no GPU, no simulator.

| alpha | noise sigma (m) | demos | mean pos err (m) | mean ori err (deg) | mean phase lag | stalled demos | mean gripper agree |
|---|---|---|---|---|---|---|---
| 0.3 | 0.0 | 6 | 0.6087 | 49.79 | 1117 | 0 | 69.9% |
| 0.3 | 0.001 | 6 | 0.6154 | 48.43 | 1006 | 0 | 68.6% |
| 0.3 | 0.005 | 6 | 0.3914 | 32.36 | 32 | 0 | 66.0% |
| 0.3 | 0.01 | 6 | 0.5848 | 58.04 | -800 | 0 | 72.7% |
| 0.3 | 0.02 | 6 | 0.7466 | 76.84 | -1621 | 0 | 39.4% |
| 0.5 | 0.0 | 6 | 0.0100 | 0.90 | -2 | 0 | 99.8% |
| 0.5 | 0.001 | 6 | 0.2610 | 24.41 | -154 | 0 | 84.3% |
| 0.5 | 0.005 | 6 | 0.6259 | 61.13 | -918 | 0 | 55.2% |
| 0.5 | 0.01 | 6 | 0.7179 | 74.56 | -1432 | 0 | 43.6% |
| 0.5 | 0.02 | 6 | 0.7375 | 85.43 | -1948 | 0 | 31.9% |
