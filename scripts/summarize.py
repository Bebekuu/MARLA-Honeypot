"""Summarise the per-seed results: mean trap/breach rates with 95% confidence intervals.

Example:
    python scripts/summarize.py --results results
"""
import argparse
import json
from pathlib import Path

import numpy as np

ATTACKERS = ["Random", "Greedy", "Targeted", "Mix"]


def load(folder):
    files = sorted(Path(folder).glob("seed*.json"), key=lambda p: int(p.stem[4:]))
    if not files:
        raise SystemExit(f"No seed*.json files found in {folder}")
    return [json.loads(f.read_text()) for f in files]


def summarise(runs):
    methods = list(runs[0]["methods"])
    table = {}
    for m in methods:
        table[m] = {}
        for a in ATTACKERS:
            trap = np.array([r["methods"][m][a]["trapped"] for r in runs]) * 100
            breach = np.array([r["methods"][m][a]["compromised"] for r in runs]) * 100
            ci = 1.96 * trap.std(ddof=1) / np.sqrt(len(trap)) if len(trap) > 1 else 0.0
            table[m][a] = {"trapped": trap.mean(), "trapped_ci95": ci, "breached": breach.mean()}
    return table


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default="results")
    parser.add_argument("--out", default="results/summary.json")
    args = parser.parse_args()

    runs = load(args.results)
    table = summarise(runs)
    print(f"{len(runs)} networks\n")
    print(f"{'Method':18s}{'Trapped % (mean ± 95% CI)':>28s}{'Breached %':>12s}")
    for m, rows in table.items():
        r = rows["Mix"]
        print(f"{m:18s}{r['trapped']:>18.1f} ± {r['trapped_ci95']:<6.1f}{r['breached']:>12.1f}")
    print("\nTrapped % by attacker type")
    print(f"{'Method':18s}" + "".join(f"{a:>10s}" for a in ATTACKERS[:3]))
    for m, rows in table.items():
        print(f"{m:18s}" + "".join(f"{rows[a]['trapped']:>10.1f}" for a in ATTACKERS[:3]))
    Path(args.out).write_text(json.dumps(table, indent=1))
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
