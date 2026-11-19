"""Draw the result figures used on the poster from the per-seed results.

Example:
    python scripts/make_figures.py --results results --out figures
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#a3a29c"
LABEL = {
    "Random-static": "Random placement",
    "Critical-static": "Guard critical subnets",
    "IPG": "Learned: IPG",
    "MAPG-CC": "Learned: MAPG-CC",
    "Alert-follow": "Follow the alert",
}


def style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 15, "axes.edgecolor": GRID,
        "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
        "axes.spines.top": False, "axes.spines.right": False,
    })


def load(folder):
    files = sorted(Path(folder).glob("seed*.json"), key=lambda p: int(p.stem[4:]))
    if not files:
        raise SystemExit(f"No seed*.json files found in {folder}")
    return [json.loads(f.read_text()) for f in files]


def stat(runs, method, attacker):
    v = np.array([r["methods"][method][attacker]["trapped"] for r in runs]) * 100
    ci = 1.96 * v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0
    return v.mean(), ci


def fig_methods(runs, out):
    order = ["Random-static", "Critical-static", "IPG", "MAPG-CC", "Alert-follow"]
    rnd = stat(runs, "Random-static", "Mix")[0]
    fig, ax = plt.subplots(figsize=(7.2, 3.7))
    ys = np.arange(len(order))[::-1]
    ax.axvline(rnd, color=INK2, ls="--", lw=1.3, zorder=1)
    ax.text(rnd + 0.8, 4.45, "random level", ha="left", va="bottom", fontsize=12.5, color=INK2)
    for y, m in zip(ys, order):
        mu, ci = stat(runs, m, "Mix")
        learned = m in ("IPG", "MAPG-CC")
        ax.barh(y, mu, height=0.62, color=BLUE if learned else GRAY, zorder=2)
        ax.errorbar(mu, y, xerr=ci, fmt="none", ecolor=INK, elinewidth=1.3, capsize=4, zorder=3)
        ax.text(mu + ci + 1.5, y, f"{mu:.1f}%", va="center", color=INK, fontsize=15,
                fontweight="bold" if learned else "normal")
        if learned:
            ax.text(2, y, f"+{mu - rnd:.1f} pts vs random", va="center", ha="left", color="white",
                    fontsize=12.5, fontweight="bold", zorder=4)
    ax.set_yticks(ys, [LABEL[m] for m in order], fontsize=15, color=INK)
    ax.set_ylim(-0.6, 4.75)
    ax.set_xlim(0, 82)
    ax.set_xlabel("Attacks trapped (%), longer is better")
    ax.xaxis.grid(True, color=GRID, zorder=0)
    ax.tick_params(axis="y", length=0)
    fig.tight_layout()
    fig.savefig(out / "fig1_trap_rate_by_method.png", dpi=300, facecolor="white")
    plt.close(fig)


def fig_attackers(runs, out):
    attackers = ["Random", "Greedy", "Targeted"]
    methods = [("Random-static", "Random placement", GRAY),
               ("Critical-static", "Guard critical subnets", AQUA),
               ("MAPG-CC", "Learned (MAPG-CC)", BLUE)]
    fig, ax = plt.subplots(figsize=(7.2, 3.7))
    w, x = 0.26, np.arange(3)
    for i, (m, lab, c) in enumerate(methods):
        stats = [stat(runs, m, a) for a in attackers]
        mus = [s[0] for s in stats]
        cis = [s[1] for s in stats]
        xs = x + (i - 1) * (w + 0.02)
        ax.bar(xs, mus, width=w, color=c, label=lab, zorder=2)
        ax.errorbar(xs, mus, yerr=cis, fmt="none", ecolor=INK, elinewidth=1.2, capsize=3, zorder=3)
        for xi, mu, ci in zip(xs, mus, cis):
            ax.text(xi, min(mu + max(ci, 2.5), 104) + 1, f"{mu:.0f}", ha="center", va="bottom",
                    fontsize=11.5, color=INK)
    ax.set_xticks(x, ["Random walk", "Greedy", "Targeted"], color=INK, fontsize=15)
    ax.set_ylim(0, 118)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.set_ylabel("Attacks trapped (%)")
    ax.yaxis.grid(True, color=GRID, zorder=0)
    ax.tick_params(axis="x", length=0)
    ax.legend(frameon=False, ncol=3, loc="upper center", fontsize=11.5, bbox_to_anchor=(0.5, 1.13),
              handlelength=1.2, columnspacing=1.0)
    fig.tight_layout()
    fig.savefig(out / "fig2_trap_rate_by_attacker.png", dpi=300, facecolor="white")
    plt.close(fig)


def fig_learning(runs, out):
    fig, ax = plt.subplots(figsize=(7.2, 3.7))
    for m, c in [("IPG", BLUE), ("MAPG-CC", ORANGE)]:
        curves = np.array([[v for _, v in r["curves"][m]] for r in runs]) * 100
        x = np.array([e for e, _ in runs[0]["curves"][m]]) / 1000
        ax.plot(x, curves.mean(0), color=c, lw=2.6, label=LABEL[m])
    for m in ("Random-static", "Critical-static"):
        v = stat(runs, m, "Mix")[0]
        ax.axhline(v, color=GRAY, ls="--", lw=1.4, zorder=1)
        ax.text(x[-1] * 1.01, v, f" {LABEL[m].lower()} {v:.0f}%", va="center", fontsize=12, color=INK2)
    ax.set_xlabel("Training episodes (thousands)")
    ax.set_ylabel("Attacks trapped (%)")
    ax.yaxis.grid(True, color=GRID, zorder=0)
    ax.legend(frameon=False, loc="lower right", fontsize=12.5, bbox_to_anchor=(0.75, 0.05))
    fig.tight_layout()
    fig.savefig(out / "fig3_learning_curve.png", dpi=300, facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default="results")
    parser.add_argument("--out", default="figures")
    args = parser.parse_args()
    style()
    runs = load(args.results)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    fig_methods(runs, out)
    fig_attackers(runs, out)
    fig_learning(runs, out)
    print(f"Saved 3 figures to {out}/")


if __name__ == "__main__":
    main()
