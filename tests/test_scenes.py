"""Study 2 scenes (scenes/object_hazard). Text-level checks run on the Mac; the sim checks run on the pod (-m gpu)."""

import csv
import os
import re
from pathlib import Path

import numpy as np
import pytest

from ftr import envs

ROOT = envs.LOCAL_SUITES["object_hazard"]
TEST_HANDS = {"hairbrush_1_with_hand", "cap_1_with_hand"}  # FSHOA L0 tasks 3-4; training hands are never used
TRAIN_HANDS = {"bottle_of_black_pepper_3_with_hand", "bottle_of_beer_1_0_with_hand", "bottle_of_tomato_paste_2_with_hand"}
LIBERO = Path(os.environ.get("FTR_LIBERO", "~/code/LIBERO")).expanduser()


def rows():
    with open(ROOT / "tasks.csv") as f:
        return list(csv.DictReader(f))


def test_ten_tasks_one_test_hand_each():
    rs = rows()
    assert [int(r["task_idx"]) for r in rs] == list(range(10))
    for r in rs:
        text = (ROOT / r["bddl"]).read_text()
        hands = re.findall(r"^\s+(\w+_with_hand_1) - (\w+_with_hand)$", text, re.M)
        assert len(hands) == 1 and hands[0][1] == r["hand"] and r["hand"] in TEST_HANDS, (r["bddl"], hands)
        assert not any(h in text for h in TRAIN_HANDS)
        h = hands[0][0]
        assert f"(On {h} floor_obs_init_region)" in text
        assert f"(CheckRobotContact {h})" in text
        assert "(:motion_travel_dist 0)" in text  # envs.restore_mocap_targets supports static hands only
        assert text.count("(") == text.count(")"), r["bddl"]


def test_language_matches_libero_object():
    # {task} must be the personalization instruction verbatim
    for r in rows():
        text = (ROOT / r["bddl"]).read_text()
        assert f"(:language {r['language']})" in text


@pytest.mark.skipif(not (LIBERO / "libero/libero/bddl_files/libero_object").is_dir(), reason="needs upstream LIBERO (FTR_LIBERO)")
def test_generator_reproduces_committed_scenes(tmp_path):
    import subprocess
    import sys

    subprocess.run([sys.executable, "scripts/make_object_hazard.py", "--libero", str(LIBERO), "--out", str(tmp_path)],
                   check=True, cwd=ROOT.parent.parent)
    for p in ROOT.glob("*.bddl"):
        assert (tmp_path / p.name).read_text() == p.read_text(), p.name
    assert (tmp_path / "tasks.csv").read_text() == (ROOT / "tasks.csv").read_text()


@pytest.mark.skipif(not (ROOT / "init_states").is_dir(), reason="init states not sampled yet (pod)")
def test_local_suite_loads():
    s = envs.suite("object_hazard")
    assert envs.n_tasks(s) == 10
    shapes = set()
    for t in range(10):
        bddl, states, language = envs.task_bddl_and_states("object_hazard", t)
        assert Path(bddl).is_file() and states.dtype == np.float64 and states.ndim == 2 and len(states) >= 20
        assert len(np.unique(states, axis=0)) == len(states), f"task {t}: duplicate init states"
        shapes.add(states.shape[1])
    assert envs.MAX_STEPS["object_hazard"] == envs.MAX_STEPS["libero_object"] == 280


@pytest.mark.gpu
def test_states_restore_and_settle_without_contact():
    """Every committed state: reset_to reproduces it and nothing touches the hand before the policy acts."""
    for t in range(10):
        bddl, states, _ = envs.task_bddl_and_states("object_hazard", t)
        env = envs.make_env(bddl)
        try:
            for st in states:
                envs.reset_to(env, st)
                assert not any(envs.eval_costs(env).values()), t
                a = envs.hand_body_pos(env).copy()
                envs.reset_to(env, st)
                assert np.allclose(a, envs.hand_body_pos(env), atol=1e-6), t
        finally:
            env.close()
