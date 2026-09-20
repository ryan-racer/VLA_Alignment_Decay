"""Numbers and figures from saved Parquet. No GPU, no simulator. Tested on the Mac with synthetic records.

Inputs (any number of run dirs):
  <run>/predictions.parquet   from score.py   -> Ru / Rs / blank, per state
  <run>/episodes.parquet      from rollout.py -> contact rates, outcome taxonomy, utility

    python -m ftr.analyze --runs runs/*/score_test runs/*_hazard runs/*_uold runs/*_unew --out figures/
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# --- statistics ---------------------------------------------------------------------------------


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    """(rate, lo, hi): Wilson 95% interval for k successes out of n."""
    if n == 0:
        return (float("nan"),) * 3
    p = k / n
    d = 1 + z**2 / n
    c = (p + z**2 / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / d
    return p, c - h, c + h


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar on discordant pairs (b: 0->1, c: 1->0)."""
    n = b + c
    if n == 0:
        return 1.0
    return float(min(1.0, 2 * stats.binom.cdf(min(b, c), n, 0.5)))


def paired_bootstrap(a: np.ndarray, b: np.ndarray, n_boot: int = 10000, seed: int = 0) -> tuple[float, float, float]:
    """Percentile CI for mean(b - a) over paired units (one value per state). BCa is skipped on purpose."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = b - a
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    boots = d[idx].mean(axis=1)
    return float(d.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


# --- refusal -------------------------------------------------------------------------------------


def refusal_by_state(pred: pd.DataFrame, col: str = "refused_k1") -> pd.DataFrame:
    """Aggregate templates -> one value per (ckpt, cls, state)."""
    return pred.groupby(["ckpt", "cls", "state_id"])[col].mean().reset_index()


def refusal_rates(pred: pd.DataFrame, col: str = "refused_k1", exclude_states: set | None = None) -> pd.DataFrame:
    """Per (ckpt, cls): mean over states of the per-state refusal, with Wilson interval on n_states."""
    df = refusal_by_state(pred, col)
    if exclude_states:
        df = df[~df["state_id"].isin(exclude_states)]
    g = df.groupby(["ckpt", "cls"])[col]
    out = g.agg(rate="mean", n_states="count").reset_index()
    ci = [wilson(int(round(r * n)), int(n)) for r, n in zip(out["rate"], out["n_states"])]
    out["lo"], out["hi"] = [c[1] for c in ci], [c[2] for c in ci]
    return out


def rs_exclusions(pred_p: pd.DataFrame) -> set:
    """States where P (before alignment) already refuses under the benign instruction — over-refusal cannot be
    measured there. PLAN.md: exclude and report the count."""
    b = pred_p[pred_p["cls"] == "benign"].groupby("state_id")["refused_k1"].mean()
    return set(b[b > 0.5].index)


# --- contact -------------------------------------------------------------------------------------


def contact_by_state(ep: pd.DataFrame) -> pd.DataFrame:
    """One row per (ckpt, cls, state): 1 if any episode on that state had hand contact."""
    return ep.groupby(["ckpt", "cls", "state_id"])["contact"].max().reset_index()


def contact_rates(ep: pd.DataFrame) -> pd.DataFrame:
    """Per (ckpt, cls): fraction of states with contact, Wilson interval."""
    df = contact_by_state(ep)
    out = df.groupby(["ckpt", "cls"])["contact"].agg(k="sum", n="count").reset_index()
    ci = [wilson(int(k), int(n)) for k, n in zip(out["k"], out["n"])]
    out["rate"], out["lo"], out["hi"] = [c[0] for c in ci], [c[1] for c in ci], [c[2] for c in ci]
    return out


def outcome_table(ep: pd.DataFrame) -> pd.DataFrame:
    """Per (ckpt, cls): share of episodes in each outcome class (held/moved/contact/timeout/success)."""
    return ep.groupby(["ckpt", "cls"])["outcome"].value_counts(normalize=True).unstack(fill_value=0.0).reset_index()


def paired_contact(ep: pd.DataFrame, ckpt_a: str, ckpt_b: str, cls: str) -> dict:
    """Paired A->B contact change on common states: delta, bootstrap CI, discordant counts, McNemar p."""
    df = contact_by_state(ep[ep["cls"] == cls])
    a = df[df["ckpt"] == ckpt_a].set_index("state_id")["contact"]
    b = df[df["ckpt"] == ckpt_b].set_index("state_id")["contact"]
    common = a.index.intersection(b.index)
    a, b = a.loc[common].astype(int), b.loc[common].astype(int)
    b01, b10 = int(((a == 0) & (b == 1)).sum()), int(((a == 1) & (b == 0)).sum())
    mean, lo, hi = paired_bootstrap(a.values, b.values)
    return dict(ckpt_a=ckpt_a, ckpt_b=ckpt_b, cls=cls, n_states=len(common), rate_a=float(a.mean()), rate_b=float(b.mean()),
                delta=mean, lo=lo, hi=hi, safe_to_unsafe=b01, unsafe_to_safe=b10, mcnemar_p=mcnemar_exact(b01, b10))


# --- utility -------------------------------------------------------------------------------------


def utility(ep: pd.DataFrame) -> pd.DataFrame:
    """Per (ckpt, suite): task success rate with Wilson interval (rows with cls == 'task')."""
    out = ep.groupby(["ckpt", "suite"])["success"].agg(k="sum", n="count").reset_index()
    ci = [wilson(int(k), int(n)) for k, n in zip(out["k"], out["n"])]
    out["rate"], out["lo"], out["hi"] = [c[0] for c in ci], [c[1] for c in ci], [c[2] for c in ci]
    return out


# --- figures -------------------------------------------------------------------------------------


def _ckpt_label(ckpt: str) -> str:
    return Path(str(ckpt)).name


def fig_refusal(rates: pd.DataFrame, out: Path):
    """Fig. 1: refusal rate per checkpoint, one line per instruction class. PDF + PNG."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    for cls, g in rates.groupby("cls"):
        x = [_ckpt_label(c) for c in g["ckpt"]]
        ax.errorbar(x, g["rate"], yerr=[g["rate"] - g["lo"], g["hi"] - g["rate"]], marker="o", capsize=3, label=cls)
    ax.set_ylim(0, 1)
    ax.set_ylabel("refusal rate (±1 bin, per state)")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "fig1_refusal.pdf")
    fig.savefig(out / "fig1_refusal.png", dpi=200)
    plt.close(fig)


def fig_contact(rates: pd.DataFrame, out: Path):
    """Fig. 2: hand-contact rate per checkpoint, harmful vs benign instruction. PDF + PNG."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    for cls, g in rates.groupby("cls"):
        x = [_ckpt_label(c) for c in g["ckpt"]]
        ax.errorbar(x, g["rate"], yerr=[g["rate"] - g["lo"], g["hi"] - g["rate"]], marker="s", capsize=3, label=f"{cls} instruction")
    ax.set_ylim(0, 1)
    ax.set_ylabel("episodes with hand contact")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "fig2_contact.pdf")
    fig.savefig(out / "fig2_contact.png", dpi=200)
    plt.close(fig)


# --- entry ---------------------------------------------------------------------------------------


def load_runs(patterns: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Concatenate predictions.parquet and episodes.parquet from every run dir matching the globs."""
    preds, eps = [], []
    for pat in patterns:
        for d in glob.glob(pat):
            d = Path(d)
            if (d / "predictions.parquet").exists():
                preds.append(pd.read_parquet(d / "predictions.parquet"))
            if (d / "episodes.parquet").exists():
                eps.append(pd.read_parquet(d / "episodes.parquet"))
    pred = pd.concat(preds, ignore_index=True) if preds else pd.DataFrame()
    ep = pd.concat(eps, ignore_index=True) if eps else pd.DataFrame()
    return pred, ep


def main():
    """CLI: rates, outcome table, paired deltas, utility, figures -> --out; summary in report.json."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", default="figures")
    ap.add_argument("--p", default=None, help="ckpt name of P, for Rs exclusions")
    ap.add_argument("--pairs", nargs="*", default=[], help="ckpt_a:ckpt_b pairs for paired contact deltas")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pred, ep = load_runs(args.runs)
    report = {}
    if len(pred):
        excl = rs_exclusions(pred[pred["ckpt"] == args.p]) if args.p else set()
        rates = refusal_rates(pred, "refused_k1", exclude_states=excl)
        rates.to_csv(out / "refusal_rates.csv", index=False)
        report["rs_excluded_states"] = sorted(excl)
        report["refusal_k0_k2"] = {k: refusal_rates(pred, k).to_dict("records") for k in ("refused_k0", "refused_k2", "roboshackles_noop")}
        fig_refusal(rates, out)
    if len(ep):
        hz = ep[ep["cls"].isin(["harmful", "benign"])]
        if len(hz):
            cr = contact_rates(hz)
            cr.to_csv(out / "contact_rates.csv", index=False)
            outcome_table(hz).to_csv(out / "outcomes.csv", index=False)
            fig_contact(cr, out)
            report["paired"] = [paired_contact(hz, *p.split(":"), cls) for p in args.pairs for cls in ("harmful", "benign")]
        ut = ep[ep["cls"] == "task"]
        if len(ut):
            utility(ut).to_csv(out / "utility.csv", index=False)
    (out / "report.json").write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps(report, indent=2, default=str)[:4000])


if __name__ == "__main__":
    main()
