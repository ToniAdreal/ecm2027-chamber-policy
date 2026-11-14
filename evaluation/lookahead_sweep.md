# ECM 2027 policy — lookahead sweep (lagged-follower harness)

Every bundled demo at alpha 0.3 / 0.5 for lookaheads 0 / 1 / 2 / 4 / 8. Errors are scored against the recorded demonstration at the same step index; the table below averages each metric over the 6 demos. Deterministic, no GPU, no simulator.

| alpha | lookahead | demos | mean pos err (m) | mean ori err (deg) | mean phase lag | mean gripper agree |
|---|---|---|---|---|---|---|
| 0.3 | 0 | 6 | 1.0860 | 57.86 | 2826 | 37.0% |
| 0.3 | 1 | 6 | 0.6177 | 46.68 | 1991 | 52.3% |
| 0.3 | 2 | 6 | 0.6087 | 49.79 | 1117 | 69.9% |
| 0.3 | 4 | 6 | 0.5137 | 48.20 | -443 | 74.7% |
| 0.3 | 8 | 6 | 0.7185 | 79.69 | -1691 | 35.6% |
| 0.5 | 0 | 6 | 1.0860 | 57.86 | 2826 | 37.0% |
| 0.5 | 1 | 6 | 0.6183 | 50.73 | 1413 | 64.7% |
| 0.5 | 2 | 6 | 0.0100 | 0.90 | -2 | 99.8% |
| 0.5 | 4 | 6 | 0.7032 | 72.65 | -1422 | 46.0% |
| 0.5 | 8 | 6 | 0.7587 | 89.93 | -2139 | 30.0% |

Best lookahead by alpha (lowest demo-averaged mean position error):

- alpha 0.3: lookahead 4
- alpha 0.5: lookahead 2
