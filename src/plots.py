"""Report figures. Colors: fixed role per ranker, same in every chart."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"
COLORS = {
    "Random": "#9a9893",
    "LC sub-grade only": "#eb6834",
    "Champion scorecard": "#2a78d6",
    "Challenger GBM": "#1baf7a",
}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "axes.titlecolor": INK,
    "xtick.color": INK_2, "ytick.color": INK_2, "text.color": INK,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "font.size": 11, "axes.titlesize": 13, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "legend.frameon": False,
})


def profit_curves(curves: dict[str, pd.DataFrame], approve_all: float,
                  devcut_points: dict[str, tuple[float, float]], path, funding_rate):
    fig, ax = plt.subplots(figsize=(9, 5.2))
    for name, c in curves.items():
        ls = "--" if name == "Random" else "-"
        ax.plot(c["approval_rate"] * 100, c["profit"] / 1e6, ls, lw=2,
                color=COLORS[name], label=name)
    for name, (rate, prof) in devcut_points.items():
        if name == "Random":
            continue
        ax.plot(rate * 100, prof / 1e6, "o", ms=8, color=COLORS[name],
                markeredgecolor=SURFACE, markeredgewidth=2, zorder=5)
    ax.axhline(approve_all / 1e6, color=INK_2, lw=1, ls=":")
    ax.text(2, approve_all / 1e6 + 0.12, f"Approve everyone: ${approve_all / 1e6:.1f}M",
            color=INK_2, fontsize=10)
    ax.set_xlabel("Approval rate (safest applicants first, %)")
    ax.set_ylabel("Realized economic profit, $M")
    ax.set_title(f"2015 out-of-time book: profit by approval rate "
                 f"({funding_rate:.0%} funding cost)")
    ax.set_xlim(0, 100)
    ax.legend(loc="upper left")
    ax.annotate("dots = cutoff chosen on 2007-2014 data",
                xy=(0.99, 0.03), xycoords="axes fraction", ha="right",
                fontsize=9, color=INK_2)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def expected_vs_actual(t: pd.DataFrame, path):
    fig, ax = plt.subplots(figsize=(9, 4.6))
    x = np.arange(len(t))
    w = 0.38
    ax.bar(x - w / 2 - 0.01, t["predicted_pd"] * 100, w, color="#a9c8ee",
           label="Predicted (champion)")
    ax.bar(x + w / 2 + 0.01, t["actual_default"] * 100, w, color=COLORS["Champion scorecard"],
           label="Actual 2015")
    for i, r in t.iterrows():
        ax.text(i + w / 2, r["actual_default"] * 100 + 0.8, f"+{r['gap_pts']:.1f}",
                ha="center", fontsize=9, color=INK_2)
    ax.set_xticks(x, t["grade"])
    ax.set_xlabel("Lending Club grade")
    ax.set_ylabel("Default rate, %")
    ax.set_title("Predicted vs actual default rate, 2015 vintage (gap in points)")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def information_value(iv: pd.DataFrame, selected: list[str], path):
    iv = iv.sort_values("iv")
    fig, ax = plt.subplots(figsize=(8, 6.5))
    colors = [COLORS["Champion scorecard"] if f in selected else "#c9c7c1" for f in iv["feature"]]
    ax.barh(iv["feature"], iv["iv"], color=colors, height=0.7)
    ax.axvline(0.02, color=INK_2, lw=1, ls=":")
    ax.text(0.021, -0.6, "IV = 0.02 threshold", fontsize=9, color=INK_2)
    ax.set_xlabel("Information Value (development sample)")
    ax.set_title("Feature screening: Information Value (blue = in final scorecard)")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
