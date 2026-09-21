"""Mac-side: statistics and figures on synthetic records shaped like score.py / rollout.py output."""

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


def test_paired_bootstrap_recovers_shift():
    rng = np.random.default_rng(0)
    a = rng.integers(0, 2, 50)
    b = np.clip(a + (rng.random(50) < 0.4), 0, 1)
    mean, lo, hi = A.paired_bootstrap(a, b)
    assert lo <= mean <= hi and mean > 0 and lo > 0


def _pred(ckpt, cls, rate, states=50, templates=2, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(states):
        for t in range(templates):
            r = bool(rng.random() < rate)
            rows.append(dict(ckpt=ckpt, state_id=f"s{s}", cls=cls, template_id=f"{cls}{t}", refused_k1=r, refused_k0=r,
                             refused_k2=r, refused_k1_gripper=r, roboshackles_noop=r))
    return pd.DataFrame(rows)


def test_refusal_rates_aggregate_per_state():
    pred = pd.concat([_pred("A0", "harmful", 0.9), _pred("A0", "benign", 0.1), _pred("A200", "harmful", 0.4)])
    r = A.refusal_rates(pred)
    assert set(r["cls"]) == {"harmful", "benign"}
    assert (r["n_states"] == 50).all()
    a0 = r[(r.ckpt == "A0") & (r.cls == "harmful")].iloc[0]
    assert 0.8 < a0["rate"] < 1.0 and a0["lo"] < a0["rate"] < a0["hi"]


def test_rs_exclusions():
    p = pd.concat([_pred("P", "benign", 0.0), _pred("P", "harmful", 0.0)])
    p.loc[(p.state_id == "s3") & (p.cls == "benign"), "refused_k1_gripper"] = True
    assert A.rs_exclusions(p) == {"s3"}


def _episodes(ckpt, cls, contact_rate, states=50, seed=0, suite="fshoa"):
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(states):
        c = bool(rng.random() < contact_rate)
        rows.append(dict(ckpt=ckpt, suite=suite, cls=cls, state_id=f"s{s}", template_id="h1", contact=c,
                         success=(not c) and rng.random() < 0.3, outcome="contact" if c else rng.choice(["held", "moved", "timeout"])))
    return pd.DataFrame(rows)


def test_paired_contact_and_figures(tmp_path):
    ep = pd.concat([_episodes("A0", "harmful", 0.06, seed=1), _episodes("A200", "harmful", 0.4, seed=2),
                    _episodes("A0", "benign", 0.05, seed=3), _episodes("A200", "benign", 0.08, seed=4)])
    pc = A.paired_contact(ep, "A0", "A200", "harmful")
    assert pc["n_states"] == 50 and pc["delta"] > 0 and pc["safe_to_unsafe"] > pc["unsafe_to_safe"]
    assert 0 <= pc["mcnemar_p"] <= 1
    cr = A.contact_rates(ep)
    A.fig_contact(cr, tmp_path)
    A.fig_refusal(A.refusal_rates(pd.concat([_pred("A0", "harmful", 0.9), _pred("A200", "harmful", 0.4)])), tmp_path)
    assert (tmp_path / "fig1_refusal.pdf").exists() and (tmp_path / "fig2_contact.png").exists()
    ot = A.outcome_table(ep)
    assert "contact" in ot.columns


def test_utility():
    ep = pd.concat([_episodes("A0", "task", 0.0, suite="libero_spatial"), _episodes("A200", "task", 0.0, suite="libero_object")])
    u = A.utility(ep)
    assert len(u) == 2 and (u["n"] == 50).all()


def test_truncate_to_horizon():
    ep = pd.DataFrame([dict(cls="blank", contact=True, contact_step=390, n_steps=390, outcome="contact", success=False),
                       dict(cls="benign", contact=True, contact_step=179, n_steps=179, outcome="contact", success=False),
                       dict(cls="benign", contact=False, contact_step=None, n_steps=520, outcome="moved", success=False),
                       dict(cls="harmful", contact=True, contact_step=27, n_steps=27, outcome="contact", success=False)])
    t = A.truncate_to_horizon(ep)
    assert list(t["contact"]) == [False, True, False, True]
    assert list(t["outcome"]) == ["moved", "contact", "moved", "contact"]
    assert list(t["n_steps"]) == [200, 179, 300, 27]
    ot = A.outcome_table(t.assign(ckpt="P"))
    assert ot.set_index("cls").loc["benign", "median_contact_step"] == 179
    assert A.truncate_to_horizon(ep.drop(columns=["contact_step"])).equals(ep.drop(columns=["contact_step"]))


def test_refusal_breakdown():
    p = pd.concat([_pred("A0", "harmful", 1.0), _pred("A0", "benign", 0.0)])
    p["task_idx"] = [3 if s.endswith("0") else 4 for s in p["state_id"]]
    b = A.refusal_breakdown(p)
    assert set(b["by"]) == {"template", "task"}
    t = b[(b["by"] == "template") & (b["cls"] == "harmful")]
    assert set(t["key"]) == {"harmful0", "harmful1"} and (t["rate"] == 1.0).all()
    assert (b[(b["by"] == "task") & (b["cls"] == "benign")]["rate"] == 0.0).all()
