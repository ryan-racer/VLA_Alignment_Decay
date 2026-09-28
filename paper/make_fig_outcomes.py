"""Figure 1: closed-loop episode outcomes per arm, instruction and study (figs/outcomes_by_arm.csv, derived from the
episodes.parquet files of the study 1 hazard runs and the study 2 test runs). python paper/make_fig_outcomes.py"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

F = Path(__file__).parent / "figs"
ARMS = [("A_s0", "$A@0$"), ("A_s0_N200_p0", "$A@200$"), ("C_s0", "$C@0$"), ("C_s0_N200_p0", "$C@200$")]
SEG = [("held", "held still", "#9fb7cc"), ("moved", "moved, no violation", "#e3e3e3"), ("violation", "violation", "#c0392b")]

df = pd.read_csv(F / "outcomes_by_arm.csv")
fig, axes = plt.subplots(1, 2, figsize=(5.6, 1.3), sharey=True)
for ax, st, title in zip(axes, (1, 2), ("Study 1 ($n=50$)", "Study 2 ($n=150$)")):
    xs, ticks = [], []
    for k, (ck, lab) in enumerate(ARMS):
        for j, cls in enumerate(("benign", "harmful")):
            x = 2.6 * k + j
            r = df[(df.study == st) & (df.ckpt == ck) & (df.cls == cls)].iloc[0]
            bottom = 0.0
            for col, _, c in SEG:
                h = 100 * r[col] / r["n"]
                ax.bar(x, h, bottom=bottom, width=0.85, color=c, edgecolor="white", linewidth=0.4)
                bottom += h
            ax.text(x, 102, f"{r['violation']}", ha="center", va="bottom", fontsize=6, color="#c0392b")
            xs.append(x); ticks.append("b" if cls == "benign" else "h")
        ax.text(2.6 * k + 0.5, -24, lab, ha="center", va="top", fontsize=7)
    ax.set_xticks(xs, ticks, fontsize=6); ax.tick_params(axis="x", length=0, pad=1)
    ax.set_ylim(0, 100); ax.set_yticks([0, 50, 100]); ax.tick_params(axis="y", labelsize=6)
    ax.set_title(title, fontsize=7.5, pad=8)
    ax.spines[["top", "right"]].set_visible(False)
axes[0].set_ylabel("% of episodes", fontsize=6.5)
h = [plt.Rectangle((0, 0), 1, 1, color=c) for _, _, c in SEG]
fig.legend(h, [lab for _, lab, _ in SEG], frameon=False, fontsize=6.5, ncol=1, loc="center right", bbox_to_anchor=(1.0, 0.55))
fig.tight_layout(rect=(0, 0.02, 0.8, 1), w_pad=1.0)
fig.savefig(F / "fig_outcomes.pdf", bbox_inches="tight", pad_inches=0.02)
fig.savefig(F / "fig_outcomes.png", dpi=220, bbox_inches="tight", pad_inches=0.02)
