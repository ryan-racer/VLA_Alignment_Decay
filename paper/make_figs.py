"""Figure 1 of the paper from the study 1 result tables (copied from $DATA/figures). python paper/make_figs.py"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

F = Path(__file__).parent / "figs"
ARMS = [("P", "$P$"), ("A_s0", "$A@0$"), ("A_s0_N50_p0", "$A@50$"), ("A_s0_N200_p0", "$A@200$"),
        ("C_s0", "$C@0$"), ("C_s0_N200_p0", "$C@200$")]
COL = {"harmful": "#c0392b", "benign": "#2471a3", "blank": "#7f8c8d"}


def panel(ax, df, ylabel, title):
    arms = [(k, lab) for k, lab in ARMS if k in set(df["ckpt"])]
    for i, cls in enumerate(("harmful", "benign", "blank")):
        d = df[df["cls"] == cls].set_index("ckpt")
        xs = [j + (i - 1) * 0.22 for j, (k, _) in enumerate(arms) if k in d.index]
        r = [d.loc[k, "rate"] for k, _ in arms if k in d.index]
        lo = [d.loc[k, "rate"] - d.loc[k, "lo"] for k, _ in arms if k in d.index]
        hi = [d.loc[k, "hi"] - d.loc[k, "rate"] for k, _ in arms if k in d.index]
        ax.bar(xs, r, 0.22, yerr=[lo, hi], color=COL[cls], label=cls, capsize=2, error_kw=dict(lw=0.8))
    ax.set_xticks(range(len(arms)), [lab for _, lab in arms])
    ax.set_ylim(0, 1.05); ax.set_ylabel(ylabel); ax.set_title(title, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)


ref = pd.read_csv(F / "refusal_rates.csv")
vio = pd.read_csv(F / "violation_rates.csv")
vio = vio[vio["metric"] == "violation"]
fig, (a, b) = plt.subplots(1, 2, figsize=(8.2, 2.6))
panel(a, ref, "refusal rate", "(a) offline refusal, first frame")
panel(b, vio, "violation rate", "(b) closed-loop safety violations")
h, l = a.get_legend_handles_labels()
fig.legend(h, l, frameon=False, fontsize=8, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.02))
fig.tight_layout(rect=(0, 0, 1, 0.92))
fig.savefig(F / "fig1.pdf"); fig.savefig(F / "fig1.png", dpi=200)
