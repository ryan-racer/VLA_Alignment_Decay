"""Sample init states for the study 2 scenes (needs the LIBERO-Safety fork + assets on PYTHONPATH; CPU is enough).

    python scripts/sample_object_hazard_states.py --n 20 --out scenes/object_hazard/init_states
    python scripts/sample_object_hazard_states.py --n 20 --suite object_hazard_p1   # study 2b candidate

Per task: seed s = 0, 1, 2, ... -> env.seed(s); reset (the fork's placement sampler places every object and the
hand's mocap target) -> record the flattened sim state, as LIBERO's own init files are made. A state is kept only if,
after envs.reset_to (the exact restore every rollout uses), no constraint fires during the settle steps and the hand
sits inside its BDDL region (obs_init_region, read from the task's BDDL). Seeds are consumed in order and the rejected
ones are printed, so the set is reproducible.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from ftr import envs
from scripts.make_object_hazard import floor_regions


def sample_task(task_idx: int, n: int, max_seeds: int, tol: float = 0.01, suite: str = "object_hazard"):
    bddl = str(envs.LOCAL_SUITES[suite] / envs.LocalSuite(suite).tasks[task_idx]["bddl"])
    region = floor_regions(open(bddl).read())["obs_init_region"]
    env = envs.make_env(bddl)  # soft reset still resamples every placement (see make_env)
    kept, rejected = [], []
    for seed in range(max_seeds):
        if len(kept) == n:
            break
        env.seed(seed)
        env.env.reset()
        state = np.asarray(env.get_sim_state(), dtype=np.float64).copy()
        envs.reset_to(env, state)
        fired = [k for k, v in envs.eval_costs(env).items() if v]
        xy = envs.hand_body_pos(env)[:2]
        inside = region[0] - tol <= xy[0] <= region[2] + tol and region[1] - tol <= xy[1] <= region[3] + tol
        if fired or not inside:
            rejected.append((seed, fired, [round(float(v), 3) for v in xy]))
            continue
        kept.append(state)
    env.close()
    assert len(kept) == n, f"task {task_idx}: only {len(kept)} valid states in {max_seeds} seeds; rejected {rejected}"
    return np.stack(kept), rejected


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--max-seeds", type=int, default=200)
    ap.add_argument("--tasks", nargs="+", default=["all"])
    ap.add_argument("--suite", default="object_hazard", choices=sorted(envs.LOCAL_SUITES))
    ap.add_argument("--out", default=None, help="default: scenes/<suite>/init_states")
    args = ap.parse_args()
    s = envs.LocalSuite(args.suite)
    out = Path(args.out) if args.out else envs.LOCAL_SUITES[args.suite] / "init_states"
    out.mkdir(parents=True, exist_ok=True)
    tasks = range(s.n_tasks) if args.tasks == ["all"] else [int(t) for t in args.tasks]
    for t in tasks:
        states, rejected = sample_task(t, args.n, args.max_seeds, suite=args.suite)
        stem = Path(s.tasks[t]["bddl"]).stem
        np.save(out / f"{stem}.npy", states)
        print(f"task {t} {stem}: {len(states)} states {states.shape}, rejected seeds {rejected}", flush=True)


if __name__ == "__main__":
    main()
