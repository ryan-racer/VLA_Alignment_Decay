"""Pod: the violation signal is real. Scripted contact fires the patched CheckRobotContact; a hold does not;
a timeout is neither. Run with PYTHONPATH pointing at LIBERO-Safety."""

import numpy as np
import pytest

from ftr import envs

pytestmark = pytest.mark.gpu

SUITE, TASK = "obstacle_avoidance_human", 2  # put_both_moka_pots_on_the_stove (train task)


@pytest.fixture(scope="module")
def env_and_state():
    bddl, states, _ = envs.task_bddl_and_states(SUITE, TASK, 0)
    env = envs.make_env(bddl)
    yield env, states[0]
    env.close()


def _run(env, state, actions, max_steps=600):
    obs = envs.reset_to(env, state)
    first_contact, n = None, 0
    gen = actions(env, obs) if callable(actions) else iter(actions)
    a = next(gen)
    while True:
        obs, success, costs, _ = envs.step(env, a)
        n += 1
        if envs.robot_contact(costs) and first_contact is None:
            first_contact = n
            break
        if n >= max_steps:
            break
        try:
            a = gen.send(obs) if callable(actions) else next(gen)
        except StopIteration:
            break
    return first_contact, n


def test_constraints_are_parsed(env_and_state):
    env, state = env_and_state
    envs.reset_to(env, state)
    keys = list(envs.eval_costs(env))
    assert any(k.startswith("checkrobotcontact") for k in keys), keys
    assert sum(k.startswith("checkcontact_") for k in keys) >= 2, "per-constraint keys must not collapse"


def test_hold_never_contacts(env_and_state):
    env, state = env_and_state
    first, n = _run(env, state, list(envs.hold(envs.MAX_STEPS[SUITE])))
    assert first is None and n == envs.MAX_STEPS[SUITE]


def test_scripted_contact_fires(env_and_state):
    env, state = env_and_state
    first, n = _run(env, state, envs.scripted_contact)
    assert first is not None, "Issue #3 patch not applied or hand geoms not matched"


def test_hand_pos_is_reachable(env_and_state):
    env, state = env_and_state
    obs = envs.reset_to(env, state)
    d = np.linalg.norm(envs.hand_body_pos(env) - np.asarray(obs["robot0_eef_pos"]))
    assert 0.05 < d < 1.0, d
