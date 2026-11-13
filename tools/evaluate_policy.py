"""Run the lagged-follower evaluation over every bundled demo.

Usage (from the repo root):

    python3 -m tools.evaluate_policy [--alphas 0.1 0.3 0.5 1.0] [--json evaluation/results.json]

Prints a Markdown table to stdout and, by default, writes the full
per-run numbers to ``evaluation/results.json`` (plus a Markdown copy
at ``evaluation/results.md``) for the submission package and the README.
Everything is deterministic: re-running produces identical numbers.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ecm_policy.evaluate import (
    DEFAULT_ALPHAS,
    evaluate_all_demos,
    results_markdown_table,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--alphas",
        type=float,
        nargs="+",
        default=list(DEFAULT_ALPHAS),
        help="lag fractions to sweep (each in (0, 1])",
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=Path(__file__).resolve().parent.parent
        / "evaluation"
        / "results.json",
        help="where to write the JSON results",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    results = evaluate_all_demos(repo_root / "data", alphas=args.alphas)
    table = results_markdown_table(results)
    print(table)

    out_dir = args.json.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    args.json.write_text(
        json.dumps([r.as_dict() for r in results], indent=2) + "\n",
        encoding="utf-8",
    )
    (out_dir / "results.md").write_text(
        "# ECM 2027 policy — lagged-follower evaluation\n\n"
        "Offline harness (`ecm_policy/evaluate.py`): the simulated robot "
        "moves a fraction `alpha` of the way toward each commanded pose "
        "per step; the policy observes the robot's actual EE positions "
        "and re-anchors its phase. Errors are scored against the recorded "
        "demonstration at the same step index. Deterministic, no GPU, no "
        "simulator.\n\n" + table + "\n",
        encoding="utf-8",
    )
    print(f"\nwrote {args.json} and {out_dir / 'results.md'}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
