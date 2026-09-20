"""Day-0 environment checks. Run on the pod after scripts/setup_pod.sh:

    PYTHONPATH=/workspace/repo:/workspace/openvla:/workspace/LIBERO-Safety pytest tests/test_env.py -v

Every assertion here corresponds to a gotcha in IMPLEMENTATION.md that would otherwise
cost a day. Keep it boring.
"""

import json
import os
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.gpu  # whole module is pod-only

P_DIR = Path(os.environ.get("FTR_P_DIR", "/workspace/hf/P"))
FSHOA_L0_TASK = 0  # any FSHOA L0 task; only used for the determinism check


# --- versions and pins ---------------------------------------------------------------------


def test_python_310():
    import sys

    assert sys.version_info[:2] == (3, 10), sys.version


def test_torch_and_flash_attn():
    import torch

    assert torch.__version__.startswith("2.2."), torch.__version__
    assert torch.cuda.is_available()
    import flash_attn  # noqa: F401  # eval loader hardcodes attn_implementation="flash_attention_2"


def test_transformers_peft_pins():
    import peft
    import transformers

    assert transformers.__version__ == "4.40.1", transformers.__version__
    assert peft.__version__ == "0.11.1", peft.__version__


def test_mujoco_237():
    # mujoco 3.x makes objects slide after set_init_state (LIBERO issue #141)
    import mujoco

    assert mujoco.__version__ == "2.3.7", mujoco.__version__


def test_numpy_pin():
    assert np.__version__ == "1.26.4", np.__version__


def test_robosuite_is_pip_copy_with_stock_controller():
    """The LIBERO-Safety fork vendors a robosuite whose osc_pose.json has output_max ±2 / kp 750.
    OpenVLA's ±0.94 outputs would become ~2 m steps. We must be on pip robosuite==1.4.1."""
    import robosuite

    assert robosuite.__version__ == "1.4.1", robosuite.__version__
    cfg = Path(robosuite.__file__).parent / "controllers" / "config" / "osc_pose.json"
    c = json.loads(cfg.read_text())
    assert c["output_max"][0] == pytest.approx(0.05), c["output_max"]
    assert c["kp"] == 150, c["kp"]
    assert "third_party" not in str(cfg), cfg


# --- rendering -----------------------------------------------------------------------------


def test_egl_env_vars():
    assert os.environ.get("MUJOCO_GL") == "egl", os.environ.get("MUJOCO_GL")
    assert os.environ.get("PYOPENGL_PLATFORM") == "egl"


def test_libero_config_exists():
    # missing ~/.libero/config.yaml makes the first import call input() and hang headless jobs
    assert (Path.home() / ".libero" / "config.yaml").exists()


def _fshoa_env_and_states():
    from libero.libero import benchmark
    from libero.libero.envs import OffScreenRenderEnv

    suite = benchmark.get_benchmark("obstacle_avoidance_human")()
    level = 0
    bddl = suite.get_task_bddl_file_path(level, FSHOA_L0_TASK)
    states = suite.get_task_init_states(level, FSHOA_L0_TASK)
    states = np.asarray(states)  # fork stores torch tensors, shape (50, nq+nv+1)
    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256)
    return env, states


def _reset_to(env, state):
    from ftr import envs

    return envs.reset_to(env, state)


@pytest.mark.gpu
def test_fshoa_renders_one_frame():
    env, states = _fshoa_env_and_states()
    try:
        obs = _reset_to(env, states[0])
        img = obs["agentview_image"]
        assert img.shape == (256, 256, 3), img.shape
        assert img.std() > 5, "blank frame: EGL is not rendering"
    finally:
        env.close()


@pytest.mark.gpu
def test_same_state_restores_identically():
    """Paired A/B rollouts are only paired if restoring the same state twice gives the same scene."""
    env, states = _fshoa_env_and_states()
    try:
        from ftr import envs

        _reset_to(env, states[3])
        a = env.sim.get_state().flatten().copy()
        ha = envs.hand_body_pos(env).copy()
        _reset_to(env, states[3])
        b = env.sim.get_state().flatten().copy()
        hb = envs.hand_body_pos(env).copy()
        # first element is time; compare qpos/qvel, and the hand (mocap-welded, placed by the seeded sampler)
        assert np.allclose(a[1:], b[1:], atol=1e-4), np.abs(a[1:] - b[1:]).max()
        assert np.allclose(ha, hb, atol=1e-4), (ha, hb)
    finally:
        env.close()


@pytest.mark.gpu
def test_with_hand_asset_loads():
    """The hand is baked into the MJCF via the fork's asset monkeypatch (lives in `libero`, not robosuite)."""
    env, states = _fshoa_env_and_states()
    try:
        from ftr import envs

        _reset_to(env, states[0])
        assert "with_hand" in envs.hand_object_name(env)
        assert envs.hand_body_pos(env).shape == (3,)
    finally:
        env.close()


# --- checkpoint ----------------------------------------------------------------------------


def test_p_norm_stats_has_exactly_one_key():
    """Released checkpoint keys its stats as `libero_spatial` (not `_no_noops`). The merged models we
    produce must keep exactly one key or predict_action asserts."""
    cfg = json.loads((P_DIR / "config.json").read_text())
    keys = list(cfg["norm_stats"].keys())
    assert keys == ["libero_spatial"], keys
    q01 = cfg["norm_stats"]["libero_spatial"]["action"]["q01"]
    q99 = cfg["norm_stats"]["libero_spatial"]["action"]["q99"]
    assert len(q01) == 7 and len(q99) == 7
    assert cfg["norm_stats"]["libero_spatial"]["action"]["mask"] == [True] * 6 + [False]
