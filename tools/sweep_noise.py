"""Sweep observation noise under the lagged-follower harness.

The real Isaac Sim evaluator will not hand the policy perfect state:
its EE-position observations carry sensor / state-estimation noise,
and the policy re-anchors its phase from those observations. This tool
measures how much noise the phase tracker tolerates before tracking
degrades: every bundled demo at alpha 0.3 / 0.5 (the lagging regime)
with the submission-default lookahead 2, for observation-noise sigmas
0 (baseline), 1 mm, 5 mm, 1 cm and 2 cm.

Noise is added ONLY to the observed EE positions (see
``ecm_policy.evaluate.evaluate_lagged_follower``); the true robot state
and the recorded trajectory stay exact, and scoring is against the
true robot pose, so noise can only hurt through the policy's phase
re-anchoring. The noise stream is seeded per demo (seed 20261114),
so re-running produces identical numbers.

Usage (from the repo root):

    python3 -m tools.sweep_noise [--json evaluation/noise_sweep.json]

Prints a Markdown table to stdout and writes the full per-run numbers
to ``evaluation/noise_sweep.json`` (plus a Markdown copy at
``evaluation/noise_sweep.md``). Deterministic: re-running produces
identical numbers. Runtime is minutes on a CPU-only host (60 full
rollouts); run it in the background.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ecm_policy.evaluate import evaluate_all_demos

# The lagging regime, where the phase tracker actually has work to do.
# alpha = 1.0 is excluded on purpose: a perfect follower reproduces the
# recording exactly at any noise level, so it carries no signal.
SWEEP_ALPHAS = (0.3, 0.5)
SWEEP_LOOKAHEAD = 2  # the submission default, chosen by the M8 sweep
SWEEP_SIGMAS_M = (0.0, 0.001, 0.005, 0.01, 0.02)
NOISE_SEED = 20261114


def sweep_table(results) -> str:
    """One row per (alpha, sigma): demo-averaged errors."""
    lines = [
        "| alpha | noise sigma (m) | demos | mean pos err (m) | mean ori err (deg) | "
        "mean phase lag | stalled demos | mean gripper agree |",
        "|---|---|---|---|---|---|---|---",
    ]
    for alpha in sorted({r.alpha for r in results}):
        for sigma in sorted({r.noise_sigma_m for r in results if r.alpha == alpha}):
            rows = [
                r
                for r in results
                if r.alpha == alpha and r.noise_sigma_m == sigma
            ]
            grip = [r.gripper_agreement for r in rows if r.gripper_agreement is not None]
            grip_str = (
                f"{100.0 * sum(grip) / len(grip):.1f}%"
                if grip
                else "n/a (no channel)"
            )
            stalled = sum(1 for r in rows if r.final_phase == 0)
            lines.append(
                f"| {alpha} | {sigma} | {len(rows)} | "
                f"{sum(r.mean_position_error_m for r in rows) / len(rows):.4f} | "
                f"{sum(r.mean_orientation_error_deg for r in rows) / len(rows):.2f} | "
                f"{sum(r.mean_phase_lag for r in rows) / len(rows):.0f} | "
                f"{stalled} | {grip_str} |"
            )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--json",
        type=Path,
        default=Path(__file__).resolve().parent.parent
        / "evaluation"
        / "noise_sweep.json",
        help="where to write the JSON results",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    results = []
    for sigma in SWEEP_SIGMAS_M:
        print(
            f"sweeping noise_sigma_m={sigma} ...",
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
            )
        )

    table = sweep_table(results)
    print(table)

    out_dir = args.json.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    args.json.write_text(
        json.dumps([r.as_dict() for r in results], indent=2) + "\n",
        encoding="utf-8",
    )
    summary = (
        "# ECM 2027 policy — observation-noise sweep (lagged-follower harness)\n\n"
        "Every bundled demo at alpha 0.3 / 0.5 with the submission-default "
        "lookahead 2, for Gaussian observation noise on the EE positions "
        "the policy sees (sigma 0 / 1 mm / 5 mm / 1 cm / 2 cm, seed "
        f"{NOISE_SEED}). Errors are scored against the recorded "
        "demonstration at the same step index using the TRUE robot pose; "
        "the table below averages each metric over the 6 demos. "
        "Deterministic, no GPU, no simulator.\n\n" + table + "\n"
    )
    (out_dir / "noise_sweep.md").write_text(summary, encoding="utf-8")
    print(f"\nwrote {args.json} and {out_dir / 'noise_sweep.md'}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
