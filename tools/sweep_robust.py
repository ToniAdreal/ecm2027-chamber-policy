"""Sweep noise-robust phase-tracking configs under the lagged-follower harness.

The M9 noise sweep (2026-11-14) showed the phase tracker is noise-fragile:
at alpha 0.5, 1 mm of observation noise breaks the near-perfect noiseless
tracking (mean position error 0.010 -> 0.261 m), and at 5 mm+ the tracker
systematically outruns the recording. The mechanism is in
``ReplayFollowCore._track_phase``: it takes the nearest demo pose inside a
250-step forward window and phase never moves backward, so a noisy
observation can yank the phase forward up to 250 steps in a single policy
step.

This tool tests the two cheap mitigations, alone and combined:

* ``phase_advance_cap`` — the phase may move at most N demo steps per
  policy step (4 / 8 / 16). The robot can only advance a few demo steps
  per step anyway, so a small cap should cost nothing noiseless and bound
  the damage of a noisy jump.
* a smaller ``phase_window`` (32 / 64) — limits how far ahead a jump can
  reach at all.

Grid: 6 configs x alpha 0.3 / 0.5 x noise sigma 0 / 1 mm / 5 mm / 1 cm,
lookahead 2 (the submission default), noise seed 20261114 (same as M9,
so the baseline column reproduces the M9 numbers). Noise is added ONLY
to the observed EE positions; the true robot state and the recorded
trajectory stay exact.

Usage (from the repo root):

    python3 -m tools.sweep_robust [--json evaluation/robust_sweep.json]

Prints a Markdown table to stdout and writes the full per-run numbers
to ``evaluation/robust_sweep.json`` (plus a Markdown copy at
``evaluation/robust_sweep.md``). Deterministic: re-running produces
identical numbers. Runtime is ~15 minutes on a CPU-only host (288 full
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
SWEEP_SIGMAS_M = (0.0, 0.001, 0.005, 0.01)
NOISE_SEED = 20261114

# (label, phase_window, phase_advance_cap)
CONFIGS = (
    ("baseline w250", 250, None),
    ("cap4 w250", 250, 4),
    ("cap8 w250", 250, 8),
    ("cap16 w250", 250, 16),
    ("win32", 32, None),
    ("win64+cap8", 64, 8),
)


def config_label(phase_window: int, phase_advance_cap) -> str:
    for label, window, cap in CONFIGS:
        if window == phase_window and cap == phase_advance_cap:
            return label
    return f"w{phase_window}/cap{phase_advance_cap}"


def sweep_table(results) -> str:
    """One row per (config, alpha, sigma): demo-averaged errors."""
    lines = [
        "| config | alpha | noise sigma (m) | demos | mean pos err (m) | mean ori err (deg) | "
        "mean phase lag | stalled demos | mean gripper agree |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for label, window, cap in CONFIGS:
        for alpha in sorted({r.alpha for r in results}):
            for sigma in sorted({r.noise_sigma_m for r in results if r.alpha == alpha}):
                rows = [
                    r
                    for r in results
                    if r.alpha == alpha
                    and r.noise_sigma_m == sigma
                    and r.phase_window == window
                    and r.phase_advance_cap == cap
                ]
                if not rows:
                    continue
                grip = [r.gripper_agreement for r in rows if r.gripper_agreement is not None]
                grip_str = (
                    f"{100.0 * sum(grip) / len(grip):.1f}%"
                    if grip
                    else "n/a (no channel)"
                )
                stalled = sum(1 for r in rows if r.final_phase == 0)
                lines.append(
                    f"| {label} | {alpha} | {sigma} | {len(rows)} | "
                    f"{sum(r.mean_position_error_m for r in rows) / len(rows):.4f} | "
                    f"{sum(r.mean_orientation_error_deg for r in rows) / len(rows):.2f} | "
                    f"{sum(r.mean_phase_lag for r in rows) / len(rows):.0f} | "
                    f"{stalled} | {grip_str} |"
                )
    return "\n".join(lines)


def best_config_by_alpha_sigma(results, sigma: float):
    """Cheapest-error config per alpha at one noise level (mean pos error)."""
    best = {}
    for alpha in sorted({r.alpha for r in results}):
        candidates = {}
        for label, window, cap in CONFIGS:
            rows = [
                r
                for r in results
                if r.alpha == alpha
                and r.noise_sigma_m == sigma
                and r.phase_window == window
                and r.phase_advance_cap == cap
            ]
            if rows:
                candidates[label] = sum(
                    r.mean_position_error_m for r in rows
                ) / len(rows)
        if candidates:
            best[alpha] = min(candidates, key=candidates.get)
    return best


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--json",
        type=Path,
        default=Path(__file__).resolve().parent.parent
        / "evaluation"
        / "robust_sweep.json",
        help="where to write the JSON results",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    results = []
    for label, window, cap in CONFIGS:
        for sigma in SWEEP_SIGMAS_M:
            print(
                f"sweeping {label} noise_sigma_m={sigma} ...",
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
                    phase_window=window,
                    phase_advance_cap=cap,
                )
            )

    table = sweep_table(results)
    print(table)

    verdict_lines = [
        "",
        "Best config by (alpha, noise sigma), lowest demo-averaged mean position error:",
        "",
    ]
    for sigma in SWEEP_SIGMAS_M:
        best = best_config_by_alpha_sigma(results, sigma)
        verdict_lines.append(
            f"- sigma {sigma}: "
            + ", ".join(f"alpha {a}: {c}" for a, c in sorted(best.items()))
        )
    verdict = "\n".join(verdict_lines)
    print(verdict)

    out_dir = args.json.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    args.json.write_text(
        json.dumps([r.as_dict() for r in results], indent=2) + "\n",
        encoding="utf-8",
    )
    summary = (
        "# ECM 2027 policy — noise-robust phase-tracking sweep (lagged-follower harness)\n\n"
        "Every bundled demo at alpha 0.3 / 0.5 with the submission-default "
        "lookahead 2, for Gaussian observation noise on the EE positions "
        "the policy sees (sigma 0 / 1 mm / 5 mm / 1 cm, seed "
        f"{NOISE_SEED}). Each config is one (phase_window, "
        "phase_advance_cap) pair; the baseline column reproduces the M9 "
        "noise sweep numbers. Errors are scored against the recorded "
        "demonstration at the same step index using the TRUE robot pose; "
        "the table below averages each metric over the 6 demos. "
        "Deterministic, no GPU, no simulator.\n\n" + table + "\n" + verdict + "\n"
    )
    (out_dir / "robust_sweep.md").write_text(summary, encoding="utf-8")
    print(f"\nwrote {args.json} and {out_dir / 'robust_sweep.md'}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
