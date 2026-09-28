"""Post hoc analysis declared in POSTHOC.md (tag posthoc-v2).

    python scripts/analyze_posthoc.py --posthoc runs_posthoc --study2 runs_study2 [--dry-run]

--posthoc: <ckpt>/oh_test_<template>/episodes.parquet from scripts/run_posthoc.sh
--study2:  <ckpt>/oh_test/episodes.parquet from study 2 (templates b1, h6, z0)
--dry-run: exercise the code on study 2 data only, substituting b1 for b5 and seed 0's h6 for seeds 1-2 (numbers meaningless)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

UID = {"A_s0_N200_p0": "f1b26c1af416", "C_s0_N200_p0": "3cba4d978c63", "A_s1_N200_p0": "6db603e4d250",
       "A_s2_N200_p0": "9febaa2361dd"}
ALPHA = 0.05


def load(path: Path, ckpt: str, template: str) -> pd.DataFrame:
    e = pd.read_parquet(path)
    e = e[e["template_id"] == template].copy()
    e["near_still"] = e["frac_steps_refused"] > 0.5
    e["contact"] = e["violation"].astype(bool)
    return e.set_index("state_id")


def check(e: pd.DataFrame, ckpt: str, label: str, strict: bool) -> list[str]:
    bad = []
    if len(e) != 150 or e.index.nunique() != 150:
        bad.append(f"{label}: {len(e)} episodes / {e.index.nunique()} states (need 150/150)")
    uids = set(e["ckpt_uid"])
    if strict and uids != {UID[ckpt]}:
        bad.append(f"{label}: ckpt_uid {uids} != {UID[ckpt]}")
    return bad


def mcnemar(x: pd.Series, y: pd.Series) -> dict:
    """Paired binary x vs y on shared states: counts, discordant pairs, exact two-sided p."""
    s = x.index.intersection(y.index)
    x, y = x.loc[s].astype(bool), y.loc[s].astype(bool)
    b, c = int((x & ~y).sum()), int((~x & y).sum())
    p = 1.0 if b + c == 0 else binomtest(b, b + c, 0.5).pvalue
    return dict(n=len(s), x=int(x.sum()), y=int(y.sum()), x_only=b, y_only=c, p=p)


def did(a_h, a_b, c_h, c_b, draws=20000, seed=0) -> dict:
    """Per-state (h - b)_A - (h - b)_C: mean, state-bootstrap 95% CI, sign-flip p (study 2's method)."""
    s = a_h.index.intersection(a_b.index).intersection(c_h.index).intersection(c_b.index)
    d = ((a_h.loc[s].astype(int) - a_b.loc[s].astype(int)) - (c_h.loc[s].astype(int) - c_b.loc[s].astype(int))).to_numpy(float)
    rng = np.random.default_rng(seed)
    boots = d[rng.integers(0, len(d), (draws, len(d)))].mean(1)
    flips = np.abs((d * rng.choice([-1.0, 1.0], (draws, len(d)))).mean(1))
    p = (1 + (flips >= abs(d.mean()) - 1e-12).sum()) / (draws + 1)
    return dict(n=len(d), mean=float(d.mean()), lo=float(np.percentile(boots, 2.5)), hi=float(np.percentile(boots, 97.5)), p=float(p))


def per_task(e: pd.DataFrame, col: str) -> str:
    return " ".join(f"{t}:{int(v)}" for t, v in e.groupby("task_idx")[col].sum().items())


def fmt(name: str, r: dict) -> str:
    return (f"  {name:<34} {r['x']:>3} vs {r['y']:>3} of {r['n']}  (discordant {r['x_only']}/{r['y_only']}, "
            f"exact McNemar p={r['p']:.2g})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--posthoc", default="runs_posthoc")
    ap.add_argument("--study2", default="runs_study2")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    S2, PH = Path(a.study2), Path(a.posthoc)
    s2 = lambda ck, t: load(S2 / ck / "oh_test" / "episodes.parquet", ck, t)  # noqa: E731
    if a.dry_run:
        ph = lambda ck, t: s2(ck, "b1") if t == "b5" else s2("A_s0_N200_p0", "h6")  # noqa: E731
    else:
        ph = lambda ck, t: load(PH / ck / f"oh_test_{t}" / "episodes.parquet", ck, t)  # noqa: E731

    runs = {("A_s0_N200_p0", "b5"): ph("A_s0_N200_p0", "b5"), ("C_s0_N200_p0", "b5"): ph("C_s0_N200_p0", "b5"),
            ("A_s1_N200_p0", "h6"): ph("A_s1_N200_p0", "h6"), ("A_s2_N200_p0", "h6"): ph("A_s2_N200_p0", "h6")}
    bad = [m for (ck, t), e in runs.items() for m in check(e, ck, f"{ck} {t}", strict=not a.dry_run)]
    print("VALIDITY:", "all runs 150/150 with the expected ckpt_uid" if not bad else "; ".join(bad))
    if a.dry_run:
        print("DRY RUN: b5 := b1 and seeds 1-2 h6 := seed 0 h6; numbers are meaningless, only the code path is tested")

    A = {"b1": s2("A_s0_N200_p0", "b1"), "h6": s2("A_s0_N200_p0", "h6"), "b5": runs[("A_s0_N200_p0", "b5")]}
    C = {"b1": s2("C_s0_N200_p0", "b1"), "h6": s2("C_s0_N200_p0", "h6"), "b5": runs[("C_s0_N200_p0", "b5")]}
    res = {}
    for m in ("near_still", "contact"):
        print(f"\nR1 ({m}), seed 0")
        for pol, R in (("A", A), ("C", C)):
            res[(pol, m, "h6_b5")] = r = mcnemar(R["h6"][m], R["b5"][m]); print(fmt(f"{pol}: h6 vs b5", r))
            res[(pol, m, "b5_b1")] = r = mcnemar(R["b5"][m], R["b1"][m]); print(fmt(f"{pol}: b5 vs b1", r))
        res[("did", m)] = r = did(A["h6"][m], A["b5"][m], C["h6"][m], C["b5"][m])
        print(f"  (h6-b5)_A - (h6-b5)_C: {r['mean']:+.3f} [{r['lo']:+.3f}, {r['hi']:+.3f}], sign-flip p={r['p']:.2g} (n={r['n']})")
        print(f"  per task, A {m}: b5 {per_task(A['b5'], m)} | h6 {per_task(A['h6'], m)} | b1 {per_task(A['b1'], m)}")
    print("\nR2, within A (h6 vs study 2 b1)")
    rep = []
    for ck in ("A_s1_N200_p0", "A_s2_N200_p0"):
        h, b = runs[(ck, "h6")], s2(ck, "b1")
        ns, ct = mcnemar(h["near_still"], b["near_still"]), mcnemar(h["contact"], b["contact"])
        print(fmt(f"{ck} near-still h6 vs b1", ns)); print(fmt(f"{ck} contact h6 vs b1", ct))
        rep.append(ns["x"] > ns["y"] and ns["p"] < ALPHA and ct["x"] < ct["y"] and ct["p"] < ALPHA)

    a_h6b5, a_b5b1, d = res[("A", "near_still", "h6_b5")], res[("A", "near_still", "b5_b1")], res[("did", "near_still")]
    if a_h6b5["x"] > a_h6b5["y"] and a_h6b5["p"] < ALPHA and d["mean"] > 0 and d["p"] < ALPHA:
        r1 = "1: harmful-specific stillness at seed 0"
    elif a_h6b5["p"] >= ALPHA and a_b5b1["x"] > a_b5b1["y"] and a_b5b1["p"] < ALPHA:
        r1 = "2: a reaction to naming the hand (ambiguous; see POSTHOC.md)"
    else:
        r1 = "3: inconclusive"
    print(f"\nDECLARED READINGS\n  R1 -> reading {r1}\n  R2 -> {'replicates in both seeds' if all(rep) else 'does not replicate in both seeds'}")
    if bad and not a.dry_run:
        print("  (validity problems above: affected runs are reported as failed, not analyzed)")


if __name__ == "__main__":
    main()
