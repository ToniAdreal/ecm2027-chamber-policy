"""Sweep the gripper replay threshold under the lagged-follower harness.

The policy replays the recorded gripper state by thresholding the
recorded ``gripper_pos`` joint trajectory (0.0 fully open, 1.703 fully
closed) into the binary gripper action. The submission default is the
joint-range midpoint 0.8515 (the starter repo's
``BinaryJointPositionActionCfg`` convention), but the official
evaluation service has never published its expected gripper action
scale, so the midpoint is a documented assumption, not a verified
fact. This tool measures what that assumption costs: the evaluator's
ground-truth recorded-closed labels always stay at the midpoint, while
the policy's replay threshold varies, so each cell reports the
gripper agreement lost when the policy's threshold differs from the
ground-truth transition point.

Grid: 5 thresholds x alpha 0.3 / 0.5 x noise sigma 0 / 1 mm / 5 mm /
1 cm, lookahead 2 and phase advance cap 4 (the submission defaults),
noise seed 20261114 (same as the M9/M10 sweeps, so the 0.8515 column
reproduces the M10 robust-sweep numbers). Gripper agreement is
defined only for the 5 demos carrying a gripper channel (Batch 2 +
Batch 3); the 4 Batch 1 demos have none and contribute position error
only, which should be identical across thresholds because the gripper
action does not feed the pose tracker.

Usage (from the repo root):

    python3 -m tools.sweep_gripper_threshold [--json evaluation/gripper_threshold_sweep.json]

Prints a Markdown table to stdout and writes the full per-run numbers
to ``evaluation/gripper_threshold_sweep.json`` (plus a Markdown copy at
``evaluation/gripper_threshold_sweep.md``). Deterministic: re-running
produces identical numbers. Runtime is ~12 minutes on a CPU-only host
(360 full rollouts); run it in the background.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ecm_policy.core import GRIPPER_THRESHOLD
from ecm_policy.evaluate import evaluate_all_demos

# The lagging regime, where the phase tracker actually has work to do.
SWEEP_ALPHAS = (0.3, 0.5)
SWEEP_LOOKAHEAD = 2  # the submission default, chosen by the M8 sweep
SWEEP_SIGMAS_M = (0.0, 0.001, 0.005, 0.01)
NOISE_SEED = 20261114
# Thresholds: near-open, quarter-range, the midpoint default,
# three-quarter-range, near-closed.
SWEEP_THRESHOLDS = (0.2, 0.425, GRIPPER_THRESHOLD, 1.28, 1.5)


def sweep_table(results) -> str:
    """One row per (threshold, alpha, sigma): demo-averaged metrics."""
    lines = [
        "| policy threshold | alpha | noise sigma (m) | demos | gripper demos | "
        "mean gripper agree | mean pos err (m) |",
        "|---|---|---|---|---|---|---|",
    ]
    for threshold in SWEEP_THRESHOLDS:
        for alpha in sorted({r.alpha for r in results}):
            for sigma in sorted({r.noise_sigma_m for r in results if r.alpha == alpha}):
                rows = [
                    r
                    for r in results
                    if r.alpha == alpha
                    and r.noise_sigma_m == sigma
                    and r.gripper_threshold == threshold
                ]
                if not rows:
                    continue
                grip = [r.gripper_agreement for r in rows if r.gripper_agreement is not None]
                grip_str = (
                    f"{100.0 * sum(grip) / len(grip):.1f}%"
                    if grip
                    else "n/a (no channel)"
                )
                lines.append(
                    f"| {threshold} | {alpha} | {sigma} | {len(rows)} | {len(grip)} | "
                    f"{grip_str} | "
                    f"{sum(r.mean_position_error_m for r in rows) / len(rows):.4f} |"
                )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--json",
        type=Path,
        default=Path(__file__).resolve().parent.parent
        / "evaluation"
        / "gripper_threshold_sweep.json",
        help="where to write the JSON results",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    results = []
    for threshold in SWEEP_THRESHOLDS:
        for sigma in SWEEP_SIGMAS_M:
            print(
                f"sweeping threshold={threshold} noise_sigma_m={sigma} ...",
                file=sys.stderr,
                flush=True,
            )
            results.extend(
                evaluate_all_demos(
                    repo_root / "data",
                    alphas=SWEEP_ALPHAS,
                    lookahead=SWEEP_LOOKAHEAD,
                    noise_sigma_m=sigma,
                    noise_seed=NOISE_SEED,
                    phase_window=250,
                    phase_advance_cap=4,
                    gripper_threshold=threshold,
                )
            )

    table = sweep_table(results)
    print(table)

    num_demos = len({r.demo for r in results})
    grip_demos = len({r.demo for r in results if r.gripper_agreement is not None})

    verdict_lines = [
        "",
        "Reading the table: the 0.8515 (midpoint) column is the submission",
        "default and reproduces the M10 robust-sweep gripper numbers. Any",
        "other threshold's agreement is the cost of a misplaced transition",
        "point against midpoint-labelled ground truth. Position error is",
        "expected to be identical across thresholds (the gripper action",
        "does not feed the pose tracker); a difference there would be a bug.",
        "",
    ]
    verdict = "\n".join(verdict_lines)
    print(verdict)

    out_dir = args.json.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    args.json.write_text(
        json.dumps([r.as_dict() for r in results], indent=2) + "\n",
        encoding="utf-8",
    )
    summary = (
        "# ECM 2027 policy — gripper replay threshold sensitivity sweep (lagged-follower harness)\n\n"
        "Every bundled demo at alpha 0.3 / 0.5 with the submission-default "
        "lookahead 2 and phase advance cap 4, for Gaussian observation "
        "noise on the EE positions the policy sees (sigma 0 / 1 mm / "
        f"5 mm / 1 cm, seed {NOISE_SEED}). The evaluator's ground-truth "
        "recorded-closed labels always use the joint-range midpoint "
        f"{GRIPPER_THRESHOLD}; only the policy's replay threshold varies "
        f"({', '.join(str(t) for t in SWEEP_THRESHOLDS)}). Gripper "
        f"agreement averages over the {grip_demos} demos carrying a "
        f"gripper channel; position error averages over all {num_demos} "
        "demos. Deterministic, no GPU, no simulator.\n\n" + table + "\n" + verdict + "\n"
    )
    md_path = args.json.with_suffix(".md")
    md_path.write_text(summary, encoding="utf-8")
    print(f"\nwrote {args.json} and {md_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
