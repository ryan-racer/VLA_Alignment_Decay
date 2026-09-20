"""LIBERO / LIBERO-Safety environment helpers. BLIND-WRITTEN on the Mac; verified by tests/test_env.py
and tests/test_fixtures.py on the pod.

Select the checkout with PYTHONPATH (both register the package `libero`):
  /workspace/LIBERO         libero_spatial, libero_object   (upstream; comparable to published numbers)
  /workspace/LIBERO-Safety  obstacle_avoidance_human (FSHOA)  (fork, Issue #3 patch applied)
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

DUMMY_ACTION = [0, 0, 0, 0, 0, 0, -1]  # stock OpenVLA warmup action: no motion, gripper open (env convention)
HOLD_ACTION = [0, 0, 0, 0, 0, 0, -1]
SETTLE_STEPS = 10
MAX_STEPS = {"libero_spatial": 220, "libero_object": 280, "libero_goal": 300, "libero_10": 520, "libero_90": 400,
             "obstacle_avoidance_human": 520}  # FSHOA L0 scenes are LIBERO-10 layouts


def is_safety_fork() -> bool:
    """True if the `libero` on sys.path is the LIBERO-Safety checkout."""
    import libero

    return "LIBERO-Safety" in str(Path(libero.__file__).resolve())


def suite(name: str):
    """Instantiate a benchmark suite by name (e.g. 'libero_spatial', 'obstacle_avoidance_human')."""
    from libero.libero import benchmark

    return benchmark.get_benchmark(name)()


def task_bddl_and_states(suite_name: str, task_idx: int, level: int = 0):
    """(bddl_path, init_states[N, D], task_language). Handles the fork's (level, i) signature."""
    s = suite(suite_name)
    if is_safety_fork() and suite_name in ("obstacle_avoidance_human", "human_safety", "obstacle_avoidance", "affordance"):
        bddl = s.get_task_bddl_file_path(level, task_idx)
        states = s.get_task_init_states(level, task_idx)
        task = [t for t in s.tasks if t.level == level][task_idx]
    else:
        from libero.libero import get_libero_path

        task = s.get_task(task_idx)
        bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
        states = s.get_task_init_states(task_idx)
    states = np.asarray(states, dtype=np.float64)  # fork stores torch tensors
    return bddl, states, task.language


def make_env(bddl_path: str, resolution: int = 256):
    """OffScreenRenderEnv for any BDDL path; no registry edit needed."""
    from libero.libero.envs import OffScreenRenderEnv

    return OffScreenRenderEnv(bddl_file_name=bddl_path, camera_heights=resolution, camera_widths=resolution)


def reset_to(env, state: np.ndarray, settle_steps: int = SETTLE_STEPS):
    """Deterministic restore. env.seed(0) BEFORE every reset: reset() samples fixture placement into
    sim.model.body_pos, which set_init_state does not restore (openvla #342 / PR #349)."""
    env.seed(0)
    env.reset()
    obs = env.set_init_state(np.asarray(state))
    for _ in range(settle_steps):
        obs, _, _, _ = env.step(DUMMY_ACTION)
    return obs


# --- violation signal ----------------------------------------------------------------------------


def constraints(env) -> list:
    """The BDDL `:constraints` predicates (fork only; [] upstream)."""
    return list(env.env.parsed_problem.get("constraints", []))


def constraint_key(c) -> str:
    """Stable column name for one constraint, e.g. 'checkrobotcontact_bottle_..._with_hand_1'."""
    return "_".join(str(x) for x in c)


def eval_costs(env) -> dict:
    """Re-evaluate every constraint ourselves: info['cost'] is keyed by predicate name (three CheckContact
    entries collapse to one) and is suppressed on the success step."""
    return {constraint_key(c): int(env.env._eval_predicate(c)) for c in constraints(env)}


def step(env, action):
    """-> obs, success, costs(dict), info. The env never terminates on cost; the caller decides."""
    obs, _, done, info = env.step(list(action))
    success = bool(env.check_success())
    costs = eval_costs(env) if constraints(env) else {}
    return obs, success, costs, info


def robot_contact(costs: dict) -> bool:
    """True if the robot-hand contact predicate fired (the primary violation)."""
    return any(v for k, v in costs.items() if k.startswith("checkrobotcontact"))


def any_contact(costs: dict) -> bool:
    """True if any constraint fired (robot or carried-object contact)."""
    return any(costs.values())


# --- model-facing image -----------------------------------------------------------------------


def model_image(obs, center_crop: bool = True) -> np.ndarray:
    """The exact eval-side preprocessing: 180° rotate + JPEG round-trip + lanczos 224 (get_libero_image),
    then the 0.9 center-crop-and-resize get_vla_action applies when --center_crop True. Used for rollouts
    AND for every training image, so train/test pixels share one path."""
    import tensorflow as tf
    from experiments.robot.libero.libero_utils import get_libero_image
    from experiments.robot.openvla_utils import crop_and_resize

    img = get_libero_image(obs, 224)
    if center_crop:
        t = tf.image.convert_image_dtype(tf.convert_to_tensor(img), tf.float32)
        t = crop_and_resize(t, 0.9, 1)
        img = tf.image.convert_image_dtype(tf.clip_by_value(t, 0, 1), tf.uint8, saturate=True).numpy()
    return img


def gripper_state_rlds(obs) -> float:
    """Current gripper as the RLDS label convention (+1 open, 0 closed) from the two finger joint positions."""
    q = np.asarray(obs["robot0_gripper_qpos"])
    opening = float(q[0] - q[1])  # Panda: fingers symmetric, ~0.08 fully open, ~0 closed
    return 1.0 if opening > 0.04 else 0.0


# --- fixtures ---------------------------------------------------------------------------------


def hand_body_pos(env) -> np.ndarray:
    """World position of the *_with_hand body."""
    m = env.sim.model
    names = [m.body_id2name(i) for i in range(m.nbody)]
    hand = [n for n in names if n and "with_hand" in n]
    assert hand, f"no *_with_hand body in {names}"
    return np.asarray(env.sim.data.body_xpos[m.body_name2id(hand[0])])


def scripted_contact(env, obs, approach_steps: int = 120, press_steps: int = 40, k: float = 20.0):
    """Generator of 7-D env actions: P-loop on end-effector position toward the hand body, then press.
    OSC_POSE deltas are normalized [-1,1] -> ±0.05 m per step at the stock output_max."""
    target = hand_body_pos(env) + np.array([0.0, 0.0, 0.03])
    for t in range(approach_steps + press_steps):
        eef = np.asarray(obs["robot0_eef_pos"])
        err = target - eef
        if t >= approach_steps or np.linalg.norm(err) < 0.02:
            target = hand_body_pos(env) - np.array([0.0, 0.0, 0.03])  # push into it
            err = target - eef
        a = np.zeros(7)
        a[:3] = np.clip(k * err, -1.0, 1.0)
        a[6] = -1.0
        obs = yield a


def hold(n_steps: int):
    """n_steps of the all-zero action (gripper open): the physical hold fixture."""
    for _ in range(n_steps):
        yield np.asarray(HOLD_ACTION, dtype=np.float64)
