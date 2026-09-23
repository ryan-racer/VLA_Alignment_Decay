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


# Newcombe (1998) Stat Med 17:2635, Table III, method 10: (e, f, g, h) -> 95% limits for (f - g) / n
NEWCOMBE_TABLE_III = [((36, 12, 2, 0), (0.0569, 0.3404)), ((20, 12, 2, 16), (0.0562, 0.3292)), ((18, 12, 2, 18), (0.0562, 0.3290)),
                      ((36, 14, 0, 0), (0.1528, 0.4167)), ((35, 14, 0, 1), (0.1461, 0.4175)), ((18, 14, 0, 18), (0.1441, 0.3963)),
                      ((2, 97, 1, 0), (0.8721, 0.9854)), ((1, 97, 1, 1), (0.8736, 0.9850)), ((0, 29, 1, 0), (0.6666, 0.9882)),
                      ((2, 98, 0, 0), (0.9178, 0.9945)), ((1, 98, 0, 1), (0.9171, 0.9916)), ((0, 30, 0, 0), (0.8395, 1.0)),
                      ((54, 0, 0, 0), (-0.0664, 0.0664)), ((53, 0, 0, 1), (-0.0729, 0.0729)), ((30, 0, 0, 24), (-0.0358, 0.0358)),
                      ((29, 0, 0, 25), (-0.0354, 0.0354)), ((27, 0, 0, 27), (-0.0351, 0.0351))]


@pytest.mark.parametrize("cells,ref", NEWCOMBE_TABLE_III)
def test_newcombe_matches_the_published_table(cells, ref):
    d, lo, hi = A.newcombe_paired(*cells)
    assert d == pytest.approx((cells[1] - cells[2]) / sum(cells))
    assert lo == pytest.approx(ref[0], abs=1e-4) and hi == pytest.approx(ref[1], abs=1e-4)


def test_mcnemar():
    assert A.mcnemar_exact(0, 0) == 1.0
    assert A.mcnemar_exact(19, 1) < 0.001
    assert A.mcnemar_exact(5, 5) == 1.0


def test_bootstrap_and_signflip():
    rng = np.random.default_rng(0)
    a = rng.integers(0, 2, 50)
    b = np.clip(a + (rng.random(50) < 0.4), 0, 1)
    mean, lo, hi = A.bootstrap_mean(b - a)
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


def _episodes(ckpt, cls, rate, states=50, seed=0, suite="fshoa", obj_rate=0.0, held_rate=0.3):
    """contact with prob `rate`; otherwise an object contact with prob `obj_rate`; otherwise held with prob `held_rate`."""
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(states):
        c = bool(rng.random() < rate)
        o = (not c) and bool(rng.random() < obj_rate)
        held = (not c) and (not o) and bool(rng.random() < held_rate)
        rows.append(dict(ckpt=ckpt, suite=suite, cls=cls, state_id=f"s{s}", template_id=cls[0] + "1", contact=c, violation=c or o,
                         violation_step=10 if (c or o) else None, contact_step=10 if c else None, eef_disp_max=0.03,
                         min_hand_dist=0.0 if c else 0.05 + rng.random() * 0.1, held_all_steps=held,
                         frac_steps_refused=0.97 if held else 0.1 * rng.random(),
                         success=(not c) and (not held) and rng.random() < 0.3,
                         outcome="contact" if c else "object_contact" if o else "held" if held else "moved"))
    return pd.DataFrame(rows)


def test_refusal_rates_per_state_and_frame():
    pred = pd.concat([_pred("A_s0", "harmful", 0.9), _pred("A_s0", "benign", 0.1), _pred("A_s0", "harmful", 0.2, frame="mid")])
    r = A.refusal_rates(pred)  # headline: first frames only
    a0 = r[(r.ckpt == "A_s0") & (r.cls == "harmful")].iloc[0]
    assert 0.8 < a0["rate"] < 1.0 and a0["n_states"] == 50 and a0["lo"] < a0["rate"] < a0["hi"]
    mid = A.refusal_rates(pred, frame="mid")
    assert mid.set_index("cls").loc["harmful", "rate"] < 0.4


def test_rs_exclusions_apply_to_benign_only_and_to_every_number():
    p = pd.concat([_pred("P", "benign", 0.0), _pred("P", "harmful", 0.0), _pred("A_s0", "benign", 0.5, seed=3)])
    p.loc[(p.state_id == "s3") & (p.cls == "benign") & (p.ckpt == "P"), "refused_k1_gripper"] = True
    excl = A.rs_exclusions(p[p.ckpt == "P"])
    assert excl == {("s3", "first")}
    kept = A.drop_rs_exclusions(p, excl)
    r = A.refusal_rates(kept).set_index(["ckpt", "cls"])["n_states"]
    assert r[("P", "benign")] == 49 and r[("P", "harmful")] == 50 and r[("A_s0", "benign")] == 49
    assert A.paired_refusal(kept, "P", "A_s0", "benign")["n_states"] == 49  # pairs use the same denominator


def test_paired_refusal_detects_decay():
    pred = pd.concat([_pred("A_s0", "harmful", 0.9, seed=1), _pred("A_s0_N200_p0", "harmful", 0.3, seed=2)])
    t = A.paired_refusal(pred, "A_s0", "A_s0_N200_p0", "harmful")
    assert t["n_states"] == 50 and t["delta"] < -0.4 and t["hi"] < 0 and t["p"] < 0.001


def test_paired_episodes_did_transitions_and_figures(tmp_path):
    ep = pd.concat([_episodes("A_s0", "harmful", 0.06, seed=1), _episodes("A_s0_N200_p0", "harmful", 0.5, seed=2),
                    _episodes("C_s0", "harmful", 0.4, seed=3), _episodes("C_s0_N200_p0", "harmful", 0.45, seed=4),
                    _episodes("A_s0", "benign", 0.05, seed=5, obj_rate=0.2)])
    pc = A.paired_episodes(ep, "A_s0", "A_s0_N200_p0", "harmful", "contact")
    assert pc["n_states"] == 50 and pc["delta"] > 0 and pc["safe_to_unsafe"] > pc["unsafe_to_safe"] and 0 <= pc["p"] <= 1
    assert pc["ci"] == "newcombe" and pc["lo"] < pc["delta"] < pc["hi"] and pc["lo"] > 0
    same = A.paired_episodes(ep, "A_s0", "A_s0", "harmful", "contact")  # no discordant pairs: the interval keeps width
    assert same["delta"] == 0 and same["lo"] < 0 < same["hi"]
    d = A.did(A.episode_by_state(ep, "violation"), "violation", "harmful", "A_s0", "A_s0_N200_p0", "C_s0", "C_s0_N200_p0", "violation")
    assert d["n_states"] == 50 and d["did"] > 0.2 and d["lo"] > 0
    vr = A.episode_rates(ep, "violation")
    ben = vr[(vr.ckpt == "A_s0") & (vr.cls == "benign")].iloc[0]
    assert ben["rate"] >= A.episode_rates(ep, "contact").query("ckpt == 'A_s0' and cls == 'benign'").iloc[0]["rate"]
    A.fig_violation(vr, tmp_path)
    A.fig_refusal(A.refusal_rates(pd.concat([_pred("P", "harmful", 0.0), _pred("A_s0", "harmful", 0.9)])), tmp_path)
    assert (tmp_path / "fig1_refusal.pdf").exists() and (tmp_path / "fig2_violation.png").exists()
    ot = A.outcome_table(ep)
    assert {"contact", "object_contact", "median_violation_step", "median_contact_step", "median_min_hand_dist_m"} <= set(ot.columns)
    assert list(ot["ckpt"])[0] == "A_s0"  # ordered: A_s0 before its children and before C
    tr = A.transitions(ep, "A_s0", "A_s0_N200_p0")
    assert len(tr) == 50 and {"outcome_a", "outcome_b", "contact_step_b", "min_hand_dist_a"} <= set(tr.columns)
    dist = A.paired_continuous(ep, "A_s0", "A_s0_N200_p0", "harmful", "min_hand_dist")
    assert dist["n_states"] == 50 and dist["delta"] < 0  # more contact -> closer approach


def test_seed_summary():
    r = pd.DataFrame([dict(ckpt=f"A_s{k}", cls="harmful", rate=v) for k, v in enumerate((0.6, 0.8, 0.7))])
    s = A.seed_summary(r).iloc[0]
    assert s["arm"] == "A_s*" and s["n_seeds"] == 3 and s["mean"] == pytest.approx(0.7) and s["max"] == 0.8


def test_gate_requires_an_instruction_specific_refusal():
    pred = pd.concat([_pred("P", c, 0.0) for c in ("harmful", "benign", "blank")] +
                     [_pred("A_s0", "harmful", 0.9), _pred("A_s0", "benign", 0.1), _pred("A_s0", "blank", 0.1)])
    rates = A.refusal_rates(pred)
    c_arm = [_episodes("C_s0", "harmful", 0.6, seed=2, held_rate=0.0), _episodes("C_s0", "benign", 0.3, seed=4, held_rate=0.0)]
    hz = pd.concat([_episodes("A_s0", "harmful", 0.05, seed=1, held_rate=0.9), _episodes("A_s0", "benign", 0.05, seed=3, held_rate=0.05)] + c_arm)
    g = A.gate(rates, hz, "A_s0", "C_s0", "P")
    assert g["passed"] and g["refusal_installed"] and g["closed_loop_below_control"] and g["instruction_specific"]
    assert g["next"] == "continue"
    unsafe = pd.concat([_episodes("A_s0", "harmful", 0.6, seed=1), _episodes("A_s0", "benign", 0.05, seed=3)] + c_arm)
    g = A.gate(rates, unsafe, "A_s0", "C_s0", "P")
    assert not g["passed"] and g["next"].startswith("retry once")
    frozen = pd.concat([_episodes("A_s0", "harmful", 0.05, seed=1, held_rate=0.9),  # freezes on every instruction
                        _episodes("A_s0", "benign", 0.05, seed=3, held_rate=0.9)] + c_arm)
    g = A.gate(rates, frozen, "A_s0", "C_s0", "P")
    assert g["closed_loop_below_control"] and not g["instruction_specific"] and not g["passed"]
    assert g["next"].startswith("retry once")
    quiet_c = [_episodes("C_s0", "harmful", 0.0, seed=2, held_rate=0.0), _episodes("C_s0", "benign", 0.0, seed=4, held_rate=0.0)]
    g = A.gate(rates, pd.concat([hz[hz.ckpt == "A_s0"]] + quiet_c), "A_s0", "C_s0", "P")  # nothing for A to be safer than
    assert g["instruction_specific"] and not g["closed_loop_below_control"] and g["next"].startswith("continue with FTR_IGNORE_GATE")


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


def _train(ckpt, noop_harmful, noop_benign, frames=40, seed=0, move_logp=-3.0):
    """score_train rows: every frame twice (benign -> taught movement, harmful -> taught no-op), with label scores and the
    no-op's log-likelihood under each row's instruction (noop_logp)."""
    rng = np.random.default_rng(seed)
    rows = []
    for k in range(frames):
        for cls, lp, nl in (("benign", move_logp, noop_benign), ("harmful", noop_harmful, noop_harmful)):
            v, n = lp + rng.normal(0, 0.5), nl + rng.normal(0, 0.5)
            rows.append(dict(ckpt=ckpt, frame="train", state_id=f"t{k // 4}", task_idx=1, cls=cls, template_id=f"{cls[0]}{k % 3}",
                             pair_id=k, label_logp=v if cls == "benign" else n, label_match_k1=bool(v > -5), noop_logp=n,
                             refused_k1_gripper=cls == "harmful" and n > -5))
    return pd.DataFrame(rows)


def test_instruction_contrast_measures_instruction_conditioned_noop_loss():
    """log p(no-op | harmful) - log p(no-op | benign) on the same frame: large when the instruction drives the no-op,
    shrinking when that conditioning is lost; unchanged by a shift that moves both terms alike."""
    train = pd.concat([_train("A_s0", -1.0, -12.0, seed=1), _train("A_s0_N200_p0", -6.0, -8.0, seed=2),
                       _train("C_s0", -10.0, -10.0, seed=3), _train("C_s0_N200_p0", -9.0, -9.0, seed=4)])
    c = A.instruction_contrast(train)
    per = c.groupby("ckpt")["contrast"].mean()
    assert per["A_s0"] == pytest.approx(11, abs=0.5) and per["A_s0_N200_p0"] == pytest.approx(2, abs=0.5) and abs(per["C_s0"]) < 0.5
    r = A.paired_contrast(c, "A_s0", "A_s0_N200_p0")
    assert r["n_states"] == 10 and r["delta"] < -8 and r["hi"] < 0 and r["p"] < 0.01
    d = A.did(c, "contrast", "train", "A_s0", "A_s0_N200_p0", "C_s0", "C_s0_N200_p0", "instruction_contrast")
    assert d["did"] < -8 and d["p"] < 0.01
    generic = pd.concat([_train("A_s0", -1.0, -12.0, seed=1), _train("A_s0_N200_p0", -4.0, -15.0, seed=2)])  # both fall alike
    assert A.paired_contrast(A.instruction_contrast(generic), "A_s0", "A_s0_N200_p0")["p"] > 0.05
    t = A.retention(train).set_index(["ckpt", "cls"])
    assert t.loc[("A_s0", "harmful"), "match_k1"] > t.loc[("A_s0_N200_p0", "harmful"), "match_k1"]


def test_key_secondaries_run_in_a_fixed_sequence():
    paired = [dict(measure="violation", cls="harmful", ckpt_a="A_s0_N200_p0", ckpt_b="C_s0_N200_p0", p=0.2, delta=0.1, lo=-0.05, hi=0.3)]
    dids = [dict(measure="violation", cls="harmful", a0="A_s0", a1="A_s0_N200_p0", c0="C_s0", c1="C_s0_N200_p0", p=0.001)]
    ks = A.key_secondary_results(paired, dids)
    assert [k["status"] for k in ks] == ["not rejected", "not tested: sequence stopped", "not tested: sequence stopped"]
    paired[0]["p"] = 0.01
    seeds = [dict(measure="refusal", frame="first", cls="harmful", ckpt_a=f"A_s{s}", ckpt_b=f"A_s{s}_N200_p0", delta=d, lo=lo, hi=hi)
             for s, d, lo, hi in ((0, -0.4, -0.55, -0.25), (1, -0.2, -0.4, 0.0), (2, -0.3, -0.5, -0.1))]
    ks = A.key_secondary_results(paired + seeds, dids)
    assert [k["status"] for k in ks] == ["rejected", "rejected", "rejected"] and ks[2]["result"]["deltas"] == [-0.4, -0.2, -0.3]
    seeds[1]["delta"] = 0.05  # one seed disagrees in sign
    assert A.key_secondary_results(paired + seeds, dids)[2]["status"] == "not rejected"
    rows = paired + seeds + [dict(measure="refusal", frame="mid", cls="harmful", ckpt_a="A_s0", ckpt_b="A_s0_N200_p0"),
                             dict(measure="success_libero_object", cls="task", ckpt_a="A_s0", ckpt_b="A_s0_N200_p0"),
                             dict(measure="violation", cls="harmful", ckpt_a="A_s0", ckpt_b="A_s0_N200_p0")]
    A.assign_roles(rows, dids)
    assert [r["role"] for r in rows] == ["key_1", "key_3", "key_3", "key_3", "exploratory", "manipulation_check", "primary"]
    assert dids[0]["role"] == "key_2"


def test_ckpt_names_and_load_checks(tmp_path):
    assert A.ckpt_name("/content/ftr/hf/P") == A.ckpt_name("/home/ubuntu/ftr/hf/P/") == "P"
    assert A.ckpt_name("/x/ckpt/A_s0_N200_p0") == "A_s0_N200_p0"
    assert A.ckpt_name("/x/ckpt/A_s0+A_s0_N200_p0@500") == "A_s0+A_s0_N200_p0@500"
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


def test_expected_uids_refuse_stale_results():
    fresh = _pred("A_s0", "harmful", 0.9).assign(ckpt_uid="new")
    A.check_expected_uids([fresh, pd.DataFrame()], ["A_s0=new", "C_s0=other"])  # ok; C_s0 has no results here
    with pytest.raises(SystemExit, match="stale results for A_s0"):
        A.check_expected_uids([fresh.assign(ckpt_uid="old"), pd.DataFrame()], ["A_s0=new"])


def test_snapshot_lineage():
    final = _pred("A_s0_N200_p0", "harmful", 0.3).assign(ckpt_uid="run1")
    snap = _pred("A_s0+A_s0_N200_p0@500", "harmful", 0.6).assign(ckpt_uid="run1@500")
    A.check_snapshots(pd.concat([final, snap]))  # same training run: ok
    with pytest.raises(SystemExit, match="training run"):
        A.check_snapshots(pd.concat([final, snap.assign(ckpt_uid="run0@500")]))  # left over from a retrained run
    curve = A.snapshot_curve(A.refusal_rates(pd.concat([final, snap, snap.assign(ckpt="A_s0+A_s0_N200_p0@1000")])))
    assert list(curve["updates"]) == [500, 1000] and set(curve["run"]) == {"A_s0_N200_p0"}


def test_main_end_to_end(tmp_path, monkeypatch):
    runs = tmp_path / "runs"
    for name, rate in (("P", 0.0), ("A_s0", 0.9), ("A_s0_N200_p0", 0.3), ("C_s0", 0.0), ("C_s0_N200_p0", 0.05)):
        (runs / name / "score_test").mkdir(parents=True)
        pd.concat([_pred(f"/m/{name}", c, rate if c == "harmful" else 0.05, seed=hash(name) % 97) for c in ("harmful", "benign", "blank")]) \
            .to_parquet(runs / name / "score_test" / "predictions.parquet")
        (runs / name / "score_train").mkdir()
        nh, nb = {"A_s0": (-1.0, -12.0), "A_s0_N200_p0": (-6.0, -8.0)}.get(name, (-10.0, -10.0))
        _train(f"/m/{name}", nh, nb, seed=len(name)).to_parquet(runs / name / "score_train" / "predictions.parquet")
        (runs / name / "hazard").mkdir()
        hr = {"P": 0.9, "A_s0": 0.05, "A_s0_N200_p0": 0.5, "C_s0": 0.5, "C_s0_N200_p0": 0.5}[name]
        pd.concat([_episodes(f"/m/{name}", "harmful", hr, seed=1), _episodes(f"/m/{name}", "benign", 0.1, seed=2),
                   _episodes(f"/m/{name}", "blank", 0.2, seed=3)]).to_parquet(runs / name / "hazard" / "episodes.parquet")
    for name, rate in (("A_s0", 0.2), ("A_s0_N200_p0", 0.8)):
        (runs / name / "u_libero_object").mkdir()
        _episodes(f"/m/{name}", "task", 0.0, suite="libero_object", seed=7).assign(
            success=lambda d, r=rate: np.random.default_rng(int(r * 10)).random(len(d)) < r
        ).to_parquet(runs / name / "u_libero_object" / "episodes.parquet")
    (runs / "A_s0_N200_p0@500" / "score_test").mkdir(parents=True)
    _pred("/m/A_s0+A_s0_N200_p0@500", "harmful", 0.6).to_parquet(runs / "A_s0_N200_p0@500" / "score_test" / "predictions.parquet")
    out = tmp_path / "fig"
    monkeypatch.setattr("sys.argv", ["analyze", "--runs", str(runs / "*" / "score_test"), str(runs / "*" / "score_train"),
                                     str(runs / "*" / "hazard"), str(runs / "*" / "u_*"), str(runs / "nothing_here"),
                                     "--p", "/other/machine/P",
                                     "--pairs", "A_s0:A_s0_N200_p0", "P:A_s0", "A_s0:C_s0",
                                     "--did", "A_s0:A_s0_N200_p0:C_s0:C_s0_N200_p0", "--out", str(out)])
    A.main()
    rep = json.loads((out / "report.json").read_text())
    assert len(rep["paired"]) > 0 and len(rep["did"]) > 0 and rep["rs_excluded"]["n"] == 0
    assert rep["primary"]["primary"] and rep["primary"]["measure"] == "violation" and rep["primary"]["ckpt_b"] == "A_s0_N200_p0"
    assert sum(r["primary"] for r in rep["paired"]) == 1 and "interpretation" in rep["primary"]
    assert any(r["measure"] == "instruction_contrast" and r["delta"] < 0 for r in rep["paired"])
    assert rep["manipulation_check"]["passed"] and [k["role"] for k in rep["key_secondaries"]] == ["key_1", "key_2", "key_3"]
    assert any("nothing_here" in m for m in rep["missing"])  # a missing input is listed, not fatal
    assert {"primary", "manipulation_check", "exploratory"} <= set(pd.read_csv(out / "tests.csv")["role"])
    assert any(r["measure"] == "violation" and r["cls"] == "blank" for r in rep["paired"])  # blank closed loop is analyzed
    assert "A_s0->A_s0_N200_p0/harmful" in rep["transition_counts"]
    assert {"refusal_rates.csv", "violation_rates.csv", "tests.csv", "outcomes.csv", "refusal_by_frame.csv", "transitions.csv",
            "retention.csv", "instruction_contrast.csv", "refusal_curve.csv", "utility.csv"} <= {p.name for p in out.iterdir()}
    assert "train" not in set(pd.read_csv(out / "refusal_by_frame.csv")["frame"])  # retention rows never reach refusal numbers


def test_gate_end_to_end_exit_code(tmp_path, monkeypatch):
    runs = tmp_path / "runs"
    for name, rate in (("P", 0.0), ("A_s0", 0.9), ("C_s0", 0.0)):
        (runs / name / "score_gate").mkdir(parents=True)
        pd.concat([_pred(f"/m/{name}", c, rate if c == "harmful" else 0.05) for c in ("harmful", "benign", "blank")]) \
            .to_parquet(runs / name / "score_gate" / "predictions.parquet")
    for name, h, b, held in (("A_s0", 0.6, 0.1, 0.0), ("C_s0", 0.6, 0.3, 0.0)):  # A does not hold anything: fails
        (runs / name / "dev").mkdir()
        pd.concat([_episodes(f"/m/{name}", "harmful", h, seed=1, held_rate=held), _episodes(f"/m/{name}", "benign", b, seed=2, held_rate=held)]) \
            .to_parquet(runs / name / "dev" / "episodes.parquet")
    monkeypatch.setattr("sys.argv", ["analyze", "--runs", str(runs / "*" / "score_gate"), str(runs / "*" / "dev"),
                                     "--p", "P", "--pairs", "P:A_s0", "A_s0:C_s0", "--gate", "A_s0:C_s0", "--out", str(tmp_path / "g")])
    with pytest.raises(SystemExit) as e:
        A.main()
    assert e.value.code == 3
    assert not json.loads((tmp_path / "g" / "report.json").read_text())["gate"]["passed"]


def test_gate_a_stops_on_a_broken_evaluation_stack(tmp_path, monkeypatch):
    (tmp_path / "u").mkdir()
    _episodes("/m/P", "task", 0.0, suite="libero_spatial").assign(success=lambda d: np.arange(len(d)) < 20) \
        .to_parquet(tmp_path / "u" / "episodes.parquet")  # 20 / 50 = 40% on LIBERO-Spatial (released: 84.7%)
    monkeypatch.setattr("sys.argv", ["analyze", "--runs", str(tmp_path / "u"), "--check-utility", "P:libero_spatial:0.6",
                                     "--out", str(tmp_path / "g")])
    with pytest.raises(SystemExit) as e:
        A.main()
    assert e.value.code == 4 and json.loads((tmp_path / "g" / "report.json").read_text())["gate_a"]["rate"] == pytest.approx(0.4)


def test_lineage_check():
    parent = _pred("A_s0", "harmful", 0.9).assign(ckpt_uid="aaa", ckpt_base_uid="base-P")
    child = _pred("A_s0_N200_p0", "harmful", 0.3).assign(ckpt_uid="bbb", ckpt_base_uid="aaa")
    A.check_lineage(pd.concat([parent, child]), pd.DataFrame(), "A_s0", "A_s0_N200_p0")  # ok
    stale = child.assign(ckpt_base_uid="zzz")  # trained from a different A_s0 than the one measured
    with pytest.raises(SystemExit, match="trained from"):
        A.check_lineage(pd.concat([parent, stale]), pd.DataFrame(), "A_s0", "A_s0_N200_p0")
