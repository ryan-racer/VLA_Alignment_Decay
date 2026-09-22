"""Mac-side: statistics and figures on synthetic records shaped like score.py / rollout.py output."""

import json

import numpy as np
import pandas as pd
import pytest

from ftr import analyze as A


def test_wilson_matches_known_values():
    p, lo, hi = A.wilson(5, 50)
    assert p == pytest.approx(0.10)
    assert lo == pytest.approx(0.043, abs=0.005) and hi == pytest.approx(0.214, abs=0.005)
    assert A.wilson(0, 50)[1] == pytest.approx(0.0, abs=1e-9)


def test_mcnemar():
    assert A.mcnemar_exact(0, 0) == 1.0
    assert A.mcnemar_exact(19, 1) < 0.001
    assert A.mcnemar_exact(5, 5) == 1.0


def test_paired_bootstrap_and_signflip():
    rng = np.random.default_rng(0)
    a = rng.integers(0, 2, 50)
    b = np.clip(a + (rng.random(50) < 0.4), 0, 1)
    mean, lo, hi = A.paired_bootstrap(a, b)
    assert lo <= mean <= hi and mean > 0 and lo > 0
    assert A.signflip_p(b - a) < 0.01
    assert A.signflip_p(rng.normal(0, 1, 50)) > 0.05 and A.signflip_p(np.zeros(10)) == 1.0


def _pred(ckpt, cls, rate, states=50, templates=2, seed=0, frame="first"):
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(states):
        for t in range(templates):
            r = bool(rng.random() < rate)
            rows.append(dict(ckpt=ckpt, frame=frame, state_id=f"s{s}", task_idx=3 + s % 2, cls=cls, template_id=f"{cls}{t}",
                             refused_k1=r, refused_k0=r, refused_k2=r, refused_k1_gripper=r, roboshackles_noop=r))
    return pd.DataFrame(rows)


def _episodes(ckpt, cls, rate, states=50, seed=0, suite="fshoa", obj_rate=0.0):
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(states):
        c = bool(rng.random() < rate)
        o = (not c) and bool(rng.random() < obj_rate)
        rows.append(dict(ckpt=ckpt, suite=suite, cls=cls, state_id=f"s{s}", template_id=cls[0] + "1", contact=c, violation=c or o,
                         violation_step=10 if (c or o) else None, eef_disp_max=0.03,
                         success=(not c) and rng.random() < 0.3,
                         outcome="contact" if c else "object_contact" if o else rng.choice(["held", "moved"])))
    return pd.DataFrame(rows)


def test_refusal_rates_per_state_and_frame():
    pred = pd.concat([_pred("A_s0", "harmful", 0.9), _pred("A_s0", "benign", 0.1), _pred("A_s0", "harmful", 0.2, frame="mid")])
    r = A.refusal_rates(pred)  # headline: first frames only
    a0 = r[(r.ckpt == "A_s0") & (r.cls == "harmful")].iloc[0]
    assert 0.8 < a0["rate"] < 1.0 and a0["n_states"] == 50 and a0["lo"] < a0["rate"] < a0["hi"]
    mid = A.refusal_rates(pred, frame="mid")
    assert mid.set_index("cls").loc["harmful", "rate"] < 0.4


def test_rs_exclusions_apply_to_benign_only():
    p = pd.concat([_pred("P", "benign", 0.0), _pred("P", "harmful", 0.0)])
    p.loc[(p.state_id == "s3") & (p.cls == "benign"), "refused_k1_gripper"] = True
    excl = A.rs_exclusions(p)
    assert excl == {"s3"}
    r = A.refusal_rates(p, exclude_states=excl).set_index("cls")["n_states"]
    assert r["benign"] == 49 and r["harmful"] == 50


def test_paired_refusal_detects_decay():
    pred = pd.concat([_pred("A_s0", "harmful", 0.9, seed=1), _pred("A_s0_N200_p0", "harmful", 0.3, seed=2)])
    t = A.paired_refusal(pred, "A_s0", "A_s0_N200_p0", "harmful")
    assert t["n_states"] == 50 and t["delta"] < -0.4 and t["hi"] < 0 and t["p"] < 0.001


def test_paired_episodes_did_and_figures(tmp_path):
    ep = pd.concat([_episodes("A_s0", "harmful", 0.06, seed=1), _episodes("A_s0_N200_p0", "harmful", 0.5, seed=2),
                    _episodes("C_s0", "harmful", 0.4, seed=3), _episodes("C_s0_N200_p0", "harmful", 0.45, seed=4),
                    _episodes("A_s0", "benign", 0.05, seed=5, obj_rate=0.2)])
    pc = A.paired_episodes(ep, "A_s0", "A_s0_N200_p0", "harmful", "contact")
    assert pc["n_states"] == 50 and pc["delta"] > 0 and pc["safe_to_unsafe"] > pc["unsafe_to_safe"] and 0 <= pc["p"] <= 1
    d = A.did(A.episode_by_state(ep, "violation"), "violation", "harmful", "A_s0", "A_s0_N200_p0", "C_s0", "C_s0_N200_p0", "violation")
    assert d["n_states"] == 50 and d["did"] > 0.2 and d["lo"] > 0
    vr = A.episode_rates(ep, "violation")
    ben = vr[(vr.ckpt == "A_s0") & (vr.cls == "benign")].iloc[0]
    assert ben["rate"] >= A.episode_rates(ep, "contact").query("ckpt == 'A_s0' and cls == 'benign'").iloc[0]["rate"]
    A.fig_violation(vr, tmp_path)
    A.fig_refusal(A.refusal_rates(pd.concat([_pred("P", "harmful", 0.0), _pred("A_s0", "harmful", 0.9)])), tmp_path)
    assert (tmp_path / "fig1_refusal.pdf").exists() and (tmp_path / "fig2_violation.png").exists()
    ot = A.outcome_table(ep)
    assert {"contact", "object_contact", "median_violation_step"} <= set(ot.columns)
    assert list(ot["ckpt"])[0] == "A_s0"  # ordered: A_s0 before its children and before C


def test_seed_summary():
    r = pd.DataFrame([dict(ckpt=f"A_s{k}", cls="harmful", rate=v) for k, v in enumerate((0.6, 0.8, 0.7))])
    s = A.seed_summary(r).iloc[0]
    assert s["arm"] == "A_s*" and s["n_seeds"] == 3 and s["mean"] == pytest.approx(0.7) and s["max"] == 0.8


def test_gate():
    pred = pd.concat([_pred("P", c, 0.0) for c in ("harmful", "benign", "blank")] +
                     [_pred("A_s0", "harmful", 0.9), _pred("A_s0", "benign", 0.1), _pred("A_s0", "blank", 0.1)])
    rates = A.refusal_rates(pred)
    hz = pd.concat([_episodes("A_s0", "harmful", 0.05, seed=1), _episodes("C_s0", "harmful", 0.6, seed=2)])
    g = A.gate(rates, hz, "A_s0", "C_s0", "P")
    assert g["passed"] and g["refusal_installed"] and g["closed_loop_below_control"]
    hz_bad = pd.concat([_episodes("A_s0", "harmful", 0.6, seed=1), _episodes("C_s0", "harmful", 0.6, seed=2)])
    assert not A.gate(rates, hz_bad, "A_s0", "C_s0", "P")["passed"]


def test_utility():
    ep = pd.concat([_episodes("A_s0", "task", 0.0, suite="libero_spatial"), _episodes("P", "task", 0.0, suite="libero_object")])
    u = A.utility(ep)
    assert len(u) == 2 and (u["n"] == 50).all() and list(u["ckpt"]) == ["P", "A_s0"]


def test_refusal_breakdown():
    p = pd.concat([_pred("A_s0", "harmful", 1.0), _pred("A_s0", "benign", 0.0), _pred("A_s0", "harmful", 0.0, frame="mid")])
    b = A.refusal_breakdown(p)
    assert set(b["by"]) == {"template", "task", "frame"}
    fr = b[(b["by"] == "frame") & (b["cls"] == "harmful")].set_index("key")["rate"]
    assert fr["first"] == 1.0 and fr["mid"] == 0.0


def test_ckpt_names_and_load_checks(tmp_path):
    assert A.ckpt_name("/content/ftr/hf/P") == A.ckpt_name("/home/ubuntu/ftr/hf/P/") == "P"
    assert A.ckpt_name("/x/ckpt/A_s0_N200_p0") == "A_s0_N200_p0"
    for d, uid in (("a", "u1"), ("b", "u2")):
        (tmp_path / d).mkdir()
        _pred("/x/A_s0", "harmful", 0.5).assign(ckpt_uid=uid, frame=d).to_parquet(tmp_path / d / "predictions.parquet")
    with pytest.raises(SystemExit, match="different weights"):
        A.load_runs([str(tmp_path / "*")])
    with pytest.raises(SystemExit, match="matched no run"):
        A.load_runs([str(tmp_path / "nothing*")])
    (tmp_path / "b" / "predictions.parquet").unlink()
    _pred("/y/A_s0", "harmful", 0.5).assign(ckpt_uid="u1", frame="a").to_parquet(tmp_path / "b" / "predictions.parquet")
    with pytest.raises(SystemExit, match="duplicate"):
        A.load_runs([str(tmp_path / "*")])


def test_main_end_to_end(tmp_path, monkeypatch):
    runs = tmp_path / "runs"
    for name, rate in (("P", 0.0), ("A_s0", 0.9), ("A_s0_N200_p0", 0.3), ("C_s0", 0.0), ("C_s0_N200_p0", 0.05)):
        (runs / name / "score_test").mkdir(parents=True)
        pd.concat([_pred(f"/m/{name}", c, rate if c == "harmful" else 0.05, seed=hash(name) % 97) for c in ("harmful", "benign", "blank")]) \
            .to_parquet(runs / name / "score_test" / "predictions.parquet")
        (runs / name / "hazard").mkdir()
        hr = {"P": 0.9, "A_s0": 0.05, "A_s0_N200_p0": 0.5, "C_s0": 0.5, "C_s0_N200_p0": 0.5}[name]
        pd.concat([_episodes(f"/m/{name}", "harmful", hr, seed=1), _episodes(f"/m/{name}", "benign", 0.1, seed=2)]) \
            .to_parquet(runs / name / "hazard" / "episodes.parquet")
    out = tmp_path / "fig"
    monkeypatch.setattr("sys.argv", ["analyze", "--runs", str(runs / "*" / "score_test"), str(runs / "*" / "hazard"),
                                     "--p", "/other/machine/P", "--pairs", "A_s0:A_s0_N200_p0",
                                     "--did", "A_s0:A_s0_N200_p0:C_s0:C_s0_N200_p0", "--gate", "A_s0:C_s0", "--out", str(out)])
    A.main()
    rep = json.loads((out / "report.json").read_text())
    assert rep["gate"]["passed"] and len(rep["paired"]) > 0 and len(rep["did"]) > 0
    assert {"refusal_rates.csv", "violation_rates.csv", "tests.csv", "outcomes.csv", "refusal_by_frame.csv"} <= {p.name for p in out.iterdir()}


def test_lineage_check():
    parent = _pred("A_s0", "harmful", 0.9).assign(ckpt_uid="aaa", ckpt_base_uid="base-P")
    child = _pred("A_s0_N200_p0", "harmful", 0.3).assign(ckpt_uid="bbb", ckpt_base_uid="aaa")
    A.check_lineage(pd.concat([parent, child]), pd.DataFrame(), "A_s0", "A_s0_N200_p0")  # ok
    stale = child.assign(ckpt_base_uid="zzz")  # trained from a different A_s0 than the one measured
    with pytest.raises(SystemExit, match="trained from"):
        A.check_lineage(pd.concat([parent, stale]), pd.DataFrame(), "A_s0", "A_s0_N200_p0")
