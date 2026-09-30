"""Per-task outcome tables for the supplement, straight from the per-episode Parquet records.

    python paper/make_supplement_tables.py --data <results folder> > paper/supp_tables.tex

Near-still = frac_steps_refused > 0.5; contact = any LIBERO-Safety violation; counts, not rates.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd

LABEL = {"P": "$P$", "A_s0": "$A$ before", "A_s0_N200_p0": "$A$ after", "A_s0_N200_e6": "$A$ 6 ep.",
         "A_s0_N200_e12": "$A$ 12 ep.", "C_s0": "$C$ before", "C_s0_N200_p0": "$C$ after",
         "C_s0_N200_e6": "$C$ 6 ep.", "C_s0_N200_e12": "$C$ 12 ep.", "A_s1": "$A$ s1 before",
         "A_s1_N200_p0": "$A$ s1 after", "A_s2": "$A$ s2 before", "A_s2_N200_p0": "$A$ s2 after"}
WORD = {"h6": "harmful", "b5": "avoid", "b1": "plain", "z0": "blank"}


def load(p: Path) -> pd.DataFrame:
    e = pd.read_parquet(p)
    e["still"] = e["frac_steps_refused"] > 0.5
    e["contact"] = e["violation"].astype(bool)
    e["success"] = e["success"].astype(bool)
    return e


def study1(d: Path) -> str:
    rows = [(c, t) for c in ("P", "A_s0", "A_s0_N200_p0", "C_s0", "C_s0_N200_p0") for t in ("h6", "b5", "z0")]
    rows += [(c, t) for c in ("A_s0_N200_e6", "A_s0_N200_e12", "C_s0_N200_e6", "C_s0_N200_e12") for t in ("h6", "b5")]
    out = [r"\begin{tabular}{llrrrrrrrr}\toprule",
           r" & & \multicolumn{3}{c}{Task 3 ($n=25$)} & \multicolumn{3}{c}{Task 4 ($n=25$)} & \multicolumn{2}{c}{Both} \\",
           r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}\cmidrule(lr){9-10}",
           r"Policy & Instr. & cont. & still & succ. & cont. & still & succ. & cont. & still \\ \midrule"]
    for c, t in rows:
        e = load(d / "runs" / c / "hazard" / "episodes.parquet")
        e = e[(e.template_id == t) & e.task_idx.isin([3, 4]) & e.state_idx.between(25, 49)]
        assert len(e) == 50 and e.state_id.nunique() == 50, (c, t, len(e))
        cells = []
        for k in (3, 4):
            s = e[e.task_idx == k]
            cells += [s.contact.sum(), s.still.sum(), s.success.sum()]
        cells += [e.contact.sum(), e.still.sum()]
        if (c, t) in (("A_s0", "h6"), ("C_s0", "h6"), ("A_s0_N200_e6", "h6")):
            out.append(r"\midrule")
        out.append(f"{LABEL[c]} & {WORD[t]} & " + " & ".join(str(int(x)) for x in cells) + r" \\")
    out.append(r"\bottomrule\end{tabular}")
    return "\n".join(out)


def study2(d: Path) -> str:
    s2, ph = d / "s2data/eps/runs_study2", d / "posthoc/mirror/runs_posthoc"
    rows = [("A_s0", "b1", s2), ("A_s0", "h6", s2), ("A_s0", "z0", s2),
            ("A_s0_N200_p0", "b1", s2), ("A_s0_N200_p0", "b5", ph), ("A_s0_N200_p0", "h6", s2), ("A_s0_N200_p0", "z0", s2),
            ("C_s0", "b1", s2), ("C_s0", "h6", s2),
            ("C_s0_N200_p0", "b1", s2), ("C_s0_N200_p0", "b5", ph), ("C_s0_N200_p0", "h6", s2),
            ("A_s1", "b1", s2), ("A_s1_N200_p0", "b1", s2), ("A_s1_N200_p0", "h6", ph),
            ("A_s2", "b1", s2), ("A_s2_N200_p0", "b1", s2), ("A_s2_N200_p0", "h6", ph)]
    out = [r"\begin{tabular}{ll" + "c" * 10 + r"rrr}\toprule",
           r"Policy & Instr. & " + " & ".join(f"T{k}{'$^h$' if k % 2 == 0 else ''}" for k in range(10))
           + r" & cont. & still & succ. \\ \midrule"]
    for c, t, root in rows:
        p = root / c / ("oh_test" if root == s2 else f"oh_test_{t}") / "episodes.parquet"
        e = load(p)
        e = e[e.template_id == t]
        assert len(e) == 150 and e.state_id.nunique() == 150, (c, t, len(e))
        g = e.groupby("task_idx")
        cells = [f"{int(g.contact.sum()[k])}/{int(g.still.sum()[k])}" for k in range(10)]
        tag = r"$^\dagger$" if root == ph else ""
        if (c, t) in (("A_s0_N200_p0", "b1"), ("C_s0", "b1"), ("C_s0_N200_p0", "b1"), ("A_s1", "b1"), ("A_s2", "b1")):
            out.append(r"\midrule")
        out.append(f"{LABEL[c]} & {WORD[t]}{tag} & " + " & ".join(cells)
                   + f" & {int(e.contact.sum())} & {int(e.still.sum())} & {int(e.success.sum())}" + r" \\")
    out.append(r"\bottomrule\end{tabular}")
    return "\n".join(out)


def study2b(d: Path) -> str:
    out = [r"\begin{tabular}{lll" + "c" * 10 + r"r}\toprule",
           r"Placement & Policy & Measure & " + " & ".join(f"T{k}" for k in range(10)) + r" & total \\ \midrule"]
    for pl in ("p1", "p2"):
        for c, m in (("C_s0_N200_p0", "contact"), ("C_s0_N200_p0", "success"), ("A_s0_N200_p0", "success"), ("A_s0_N200_p0", "contact")):
            e = load(d / "s2data/eps/runs_study2b" / c / f"object_hazard_{pl}_dev" / "episodes.parquet")
            assert len(e) == 50, (pl, c, len(e))
            g = e.groupby("task_idx")[m].sum()
            out.append(f"{pl} & {LABEL[c]} & {m} & " + " & ".join(str(int(g[k])) for k in range(10)) + f" & {int(e[m].sum())}/50" + r" \\")
        if pl == "p1":
            out.append(r"\midrule")
    out.append(r"\bottomrule\end{tabular}")
    return "\n".join(out)


def utility(d: Path) -> str:
    ck = ["P", "A_s0", "A_s0_N200_p0", "A_s0_N200_e6", "A_s0_N200_e12", "C_s0_N200_p0", "C_s0_N200_e6", "C_s0_N200_e12"]
    out = [r"\begin{tabular}{ll" + "c" * 10 + r"r}\toprule",
           r"Suite & Policy & " + " & ".join(f"T{k}" for k in range(10)) + r" & total \\ \midrule"]
    for suite, name in (("libero_spatial", "Spatial (old)"), ("libero_object", "Object (new)")):
        for c in ck:
            p = d / "s2data/eps/runs" / c / f"u_{suite}" / "episodes.parquet"
            if not p.exists():
                continue
            e = load(p)
            assert len(e) == 50, (c, suite, len(e))
            g = e.groupby("task_idx").success.sum()
            out.append(f"{name} & {LABEL[c]} & " + " & ".join(str(int(g[k])) for k in range(10)) + f" & {int(e.success.sum())}/50" + r" \\")
        if suite == "libero_spatial":
            out.append(r"\midrule")
    out.append(r"\bottomrule\end{tabular}")
    return "\n".join(out)


def refusal(d: Path) -> str:
    r = pd.read_csv(d / "s2data/figures/refusal_rates.csv")
    r = r[~r.ckpt.str.contains("@")].pivot(index="ckpt", columns="cls", values="rate")
    out = [r"\begin{tabular}{lccccccccc}\toprule",
           r" & \multicolumn{3}{c}{harmful} & \multicolumn{3}{c}{benign} & \multicolumn{3}{c}{blank} \\",
           r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}",
           r"Policy & s0 & s1 & s2 & s0 & s1 & s2 & s0 & s1 & s2 \\ \midrule",
           r"$P$ & " + " & ".join(f"{r.loc['P', k]:.2f}" if s == 0 else "" for k in ("harmful", "benign", "blank") for s in range(3)) + r" \\"]
    for arm, name in (("A_s{}", "$A$ before"), ("A_s{}_N200_p0", "$A$ after"), ("C_s{}", "$C$ before"), ("C_s{}_N200_p0", "$C$ after")):
        out.append(f"{name} & " + " & ".join(f"{r.loc[arm.format(s), k]:.2f}" for k in ("harmful", "benign", "blank") for s in range(3)) + r" \\")
    out.append(r"\bottomrule\end{tabular}")
    return "\n".join(out)


SEED0 = (("P", "$P$"), ("A_s0", "$A$ before"), ("A_s0_N50_p0", "$A$ after, $N{=}50$"), ("A_s0_N200_p0", "$A$ after"),
         ("C_s0", "$C$ before"), ("C_s0_N200_p0", "$C$ after"))
CLS = ("harmful", "benign", "blank")


def offline(d: Path) -> str:
    """Seed-0 first-frame and mid-trajectory refusal (headline criterion) and the pose-only criterion sweep."""
    f = pd.read_csv(d / "s2data/figures/refusal_by_frame.csv")
    f = f[~f.ckpt.str.contains("@")].pivot_table(index="ckpt", columns=["frame", "cls"], values="rate")
    out = [r"\begin{tabular}{lcccccc}\toprule",
           r" & \multicolumn{3}{c}{First frame} & \multicolumn{3}{c}{Mid-trajectory frame} \\",
           r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
           r"Policy & harmful & benign & blank & harmful & benign & blank \\ \midrule"]
    for ck, name in SEED0:
        out.append(f"{name} & " + " & ".join(f"{f.loc[ck, (fr, c)]:.2f}" for fr in ("first", "mid") for c in CLS) + r" \\")
    out.append(r"\bottomrule\end{tabular}")
    return "\n".join(out)


def criteria(d: Path) -> str:
    """Largest first-frame difference between criteria over all 14 checkpoints x 3 classes."""
    import json
    oc = json.load(open(d / "s2data/figures/report.json"))["other_criteria"]
    t = {k: pd.DataFrame(v).query("~ckpt.str.contains('@')", engine="python").set_index(["ckpt", "cls"]).rate for k, v in oc.items()}
    head = pd.read_csv(d / "s2data/figures/refusal_rates.csv").query("~ckpt.str.contains('@')", engine="python").set_index(["ckpt", "cls"]).rate
    pose = pd.concat([t["refused_k0"], t["refused_k1"], t["refused_k2"], t["roboshackles_noop"]], axis=1)
    spread = (pose.max(axis=1) - pose.min(axis=1)).max()
    grip = (t["refused_k1"] - head.loc[t["refused_k1"].index]).abs()
    k = grip.idxmax()
    return (f"\\newcommand{{\\suppCritSpread}}{{{spread:.2f}}}\n\\newcommand{{\\suppGripMax}}{{{grip.max():.2f}}}\n"
            f"\\newcommand{{\\suppGripWhere}}{{seed-{k[0][3]} ${k[0][0]}$ {'after' if 'N200' in k[0] else 'before'}, {k[1]}}}\n"
            f"\\newcommand{{\\suppGripN}}{{{int((grip > 0.005).sum())}}}")


def decay(d: Path) -> str:
    """First-frame refusal of the unmerged personalization adapter every 500 updates (seed 0)."""
    r = pd.read_csv(d / "s2data/figures/refusal_rates.csv")
    rows = []
    for arm in ("A", "C"):
        base = r[r.ckpt == f"{arm}_s0"].set_index("cls").rate
        rows.append((arm, 0, base))
        snap = r[r.ckpt.str.startswith(f"{arm}_s0+{arm}_s0_N200_p0@")].copy()
        snap["u"] = snap.ckpt.str.split("@").str[1].astype(int)
        for u, g in snap.groupby("u"):
            rows.append((arm, u, g.set_index("cls").rate))
        rows.append((arm, "merged", r[r.ckpt == f"{arm}_s0_N200_p0"].set_index("cls").rate))
    ups = [u for a, u, _ in rows if a == "A"]
    get = {(a, u): v for a, u, v in rows}
    out = [r"\begin{tabular}{rcccccc}\toprule",
           r" & \multicolumn{3}{c}{$A$} & \multicolumn{3}{c}{$C$} \\ \cmidrule(lr){2-4}\cmidrule(lr){5-7}",
           r"Updates & harmful & benign & blank & harmful & benign & blank \\ \midrule"]
    for u in ups:
        out.append(f"{u} & " + " & ".join(f"{get[(a, u)][c]:.2f}" for a in ("A", "C") for c in CLS) + r" \\")
    out.append(r"\bottomrule\end{tabular}")
    return "\n".join(out)


def contrast(d: Path) -> str:
    """Same-frame instruction contrast and taught-label retention on A's training frames (seed 0), as macros."""
    c = pd.read_csv(d / "s2data/figures/instruction_contrast.csv").set_index("ckpt")["mean"]
    r = pd.read_csv(d / "s2data/figures/retention.csv").set_index(["ckpt", "cls"]).match_k1
    m = {"CtrAb": c["A_s0"], "CtrAa": c["A_s0_N200_p0"], "CtrCb": c["C_s0"], "CtrCa": c["C_s0_N200_p0"], "CtrP": c["P"],
         "RetAbH": r[("A_s0", "harmful")], "RetAaH": r[("A_s0_N200_p0", "harmful")],
         "RetAbB": r[("A_s0", "benign")], "RetAaB": r[("A_s0_N200_p0", "benign")]}
    return "\n".join(f"\\newcommand{{\\supp{k}}}{{{v:.2f}}}" for k, v in m.items())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.environ.get("FTR_DATA", "ftr-review-data"), help="results folder")
    d = Path(ap.parse_args().data)
    for name, fn in (("studyOne", study1), ("studyTwo", study2), ("studyTwob", study2b), ("utility", utility),
                     ("refusal", refusal), ("offline", offline), ("decay", decay)):
        src = fn(d / "s2data/eps") if name == "studyOne" else fn(d)
        print(f"\\newcommand{{\\supp{name}}}{{%\n{src}}}\n")
    print(criteria(d) + "\n" + contrast(d))


if __name__ == "__main__":
    main()
