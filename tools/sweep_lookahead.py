"""Sweep the policy lookahead under the lagged-follower harness.

The submission policy commands the demo pose at ``phase + lookahead``.
Lookahead 1 is the current default, chosen without measurement; a larger
lookahead lets a lagging follower aim further ahead, but it also
commands poses the robot has not reached yet, which can overshoot when
the follower is fast. This tool measures the trade-off instead of
guessing it: every bundled demo at alpha 0.3 and 0.5 (the lagging
regime where the choice matters) for lookaheads 0, 1, 2, 4 and 8, and
reports the lookahead that minimizes the mean position error averaged
over demos at each alpha.

Usage (from the repo root):

    python3 -m tools.sweep_lookahead [--json evaluation/lookahead_sweep.json]

Prints a Markdown table to stdout and writes the full per-run numbers
to ``evaluation/lookahead_sweep.json`` (plus a Markdown copy at
``evaluation/lookahead_sweep.md``). Deterministic: re-running produces
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

# The lagging regime, where lookahead actually changes the outcome.
# alpha = 1.0 is excluded on purpose: a perfect follower reproduces the
# recording exactly at any lookahead, so it carries no signal.
SWEEP_ALPHAS = (0.3, 0.5)
SWEEP_LOOKAHEADS = (0, 1, 2, 4, 8)


def best_lookahead_by_alpha(results) -> dict[float, int]:
    """Per alpha, the lookahead minimizing mean position error.

    Averaged over demos first (each demo contributes its own mean), so a
    long demo cannot outvote a short one.
    """
    best: dict[float, int] = {}
    for alpha in sorted({r.alpha for r in results}):
        per_lookahead: dict[int, list[float]] = {}
        for r in results:
            if r.alpha == alpha:
                per_lookahead.setdefault(r.lookahead, []).append(
                    r.mean_position_error_m
                )
        means = {
            la: sum(v) / len(v) for la, v in per_lookahead.items()
        }
        best[alpha] = min(means, key=means.get)
    return best


def sweep_table(results) -> str:
    """One row per (alpha, lookahead): demo-averaged errors."""
    lines = [
        "| alpha | lookahead | demos | mean pos err (m) | mean ori err (deg) | "
        "mean phase lag | mean gripper agree |",
        "|---|---|---|---|---|---|---|",
    ]
    for alpha in sorted({r.alpha for r in results}):
        for la in sorted({r.lookahead for r in results if r.alpha == alpha}):
            rows = [r for r in results if r.alpha == alpha and r.lookahead == la]
            grip = [r.gripper_agreement for r in rows if r.gripper_agreement is not None]
            grip_str = (
                f"{100.0 * sum(grip) / len(grip):.1f}%"
                if grip
                else "n/a (no channel)"
            )
            lines.append(
                f"| {alpha} | {la} | {len(rows)} | "
                f"{sum(r.mean_position_error_m for r in rows) / len(rows):.4f} | "
                f"{sum(r.mean_orientation_error_deg for r in rows) / len(rows):.2f} | "
                f"{sum(r.mean_phase_lag for r in rows) / len(rows):.0f} | "
                f"{grip_str} |"
            )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--json",
        type=Path,
        default=Path(__file__).resolve().parent.parent
        / "evaluation"
        / "lookahead_sweep.json",
        help="where to write the JSON results",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    results = []
    for lookahead in SWEEP_LOOKAHEADS:
        print(
            f"sweeping lookahead={lookahead} ...",
            file=sys.stderr,
            flush=True,
        )
        results.extend(
            evaluate_all_demos(
                repo_root / "data",
                alphas=SWEEP_ALPHAS,
                lookahead=lookahead,
            )
        )

    best = best_lookahead_by_alpha(results)
    table = sweep_table(results)
    print(table)
    print(
        "\nBest lookahead by alpha (lowest demo-averaged mean position error):",
    )
    for alpha in sorted(best):
        print(f"  alpha {alpha}: lookahead {best[alpha]}")

    out_dir = args.json.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    args.json.write_text(
        json.dumps([r.as_dict() for r in results], indent=2) + "\n",
        encoding="utf-8",
    )
    summary = (
        "# ECM 2027 policy — lookahead sweep (lagged-follower harness)\n\n"
        "Every bundled demo at alpha 0.3 / 0.5 for lookaheads 0 / 1 / 2 / 4 / 8. "
        "Errors are scored against the recorded demonstration at the same "
        "step index; the table below averages each metric over the 6 demos. "
        "Deterministic, no GPU, no simulator.\n\n" + table + "\n\n"
        "Best lookahead by alpha (lowest demo-averaged mean position error):\n\n"
        + "".join(f"- alpha {a}: lookahead {la}\n" for a, la in sorted(best.items()))
    )
    (out_dir / "lookahead_sweep.md").write_text(summary, encoding="utf-8")
    print(f"\nwrote {args.json} and {out_dir / 'lookahead_sweep.md'}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
