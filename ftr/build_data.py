"""Assemble training and evaluation Parquet from (a) rendered states, (b) rollouts, (c) RLDS exports. TF parts: pod.
Images are stored UNCROPPED (the RLDS-equivalent form); training augments as stock OpenVLA, evaluation center-crops.

    # 1. test states rendered once (image + gripper state)
    python -m ftr.build_data render --suite obstacle_avoidance_human --tasks 3 4 --states 0-24 --out data/states_test.parquet
    # 2. offline scoring pairs: first frames + mid-trajectory frames (from P's hazard run, --store-every) x test templates
    python -m ftr.build_data pairs --states data/states_test.parquet --template-split test --mid-from runs/P/hazard \
        --out data/pairs_test.parquet
    # 3. RLDS subset export (TF): rehearsal and personalization rows
    python -m ftr.build_data export-rlds --rlds ~/ftr/hf/rlds/libero_spatial_no_noops --episodes 30 --seed 0 --category rehearsal --out data/spatial_rehearsal.parquet
    python -m ftr.build_data export-rlds --rlds ~/ftr/hf/rlds/libero_object_no_noops --per-task 20 --seed 0 --out data/object_N200_p0.parquet
    # 4. the arms (scripted movement labels on the training tasks in splits.csv)
    python -m ftr.build_data mix --arm A --self runs/scripted runs/scripted_t0b --rehearsal data/spatial_rehearsal.parquet --out data/A.parquet
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from ftr.data import image_to_png_bytes, write_parquet


# --- render ------------------------------------------------------------------------------------


def cmd_render(args):
    """Restore each requested state once, store the model-facing image + gripper state per state."""
    from ftr import envs
    from ftr.rollout import parse_states

    rows = []
    for task_idx in [int(t) for t in args.tasks]:
        bddl, states, language = envs.task_bddl_and_states(args.suite, task_idx, args.level)
        env = envs.make_env(bddl)
        try:
            idx = parse_states(args.states, len(states))
            for j, si in enumerate(idx):
                obs = envs.reset_to(env, states[si])
                if j % 10 == 0:
                    print(f"task {task_idx}: state {j+1}/{len(idx)}", flush=True)
                rows.append(dict(suite=args.suite, level=args.level, task_idx=task_idx, task=language, state_idx=si,
                                 state_id=f"{args.suite}/{args.level}/{task_idx}/{si}",
                                 image=image_to_png_bytes(envs.model_image(obs, center_crop=False)),
                                 gripper_state=envs.GRIPPER_OPEN_RLDS))  # settle steps command open
        finally:
            env.close()
    write_parquet(pd.DataFrame(rows), args.out)
    print(f"rendered {len(rows)} states -> {args.out}")


# --- pairs -------------------------------------------------------------------------------------


def expand_instructions(states: pd.DataFrame, instructions: pd.DataFrame) -> pd.DataFrame:
    """Cross states x instruction templates; fills `{task}` with the task language."""
    out = []
    for _, s in states.iterrows():
        for _, r in instructions.iterrows():
            out.append({**s.to_dict(), "cls": r["class"], "template_id": r["template_id"],
                        "instruction": r["text"].replace("{task}", s["task"])})
    return pd.DataFrame(out)


def mid_frames(run_dir: str, per_state: int, min_t: int, rng) -> pd.DataFrame:
    """Mid-trajectory frames from a rollout run with stored images: per state, `per_state` frames drawn uniformly
    from steps t >= min_t that precede the episode's first constraint violation (the arm is displaced, the scene
    no longer looks like an initial state)."""
    from ftr.rollout import read_steps

    st = read_steps(run_dir)
    assert "image" in st.columns, f"{run_dir}: no stored images (rollout --store-images / --store-every)"
    st = st[st["image"].notna()]  # --store-every k keeps every k-th frame only
    ep = pd.read_parquet(Path(run_dir) / "episodes.parquet")[["state_id", "template_id", "violation_step"]]
    st = st.merge(ep, on=["state_id", "template_id"], how="left")
    st = st[(st["t"] >= min_t) & (st["violation_step"].isna() | (st["t"] < st["violation_step"]))]
    rows = []
    for sid, g in st.groupby("state_id"):
        rows.append(g.sample(n=min(per_state, len(g)), random_state=int(rng.integers(1 << 31))))
    keep = ["state_id", "task_idx", "task", "t", "image", "gripper_state"]
    return pd.concat(rows)[keep].reset_index(drop=True) if rows else pd.DataFrame(columns=keep)


def cmd_pairs(args):
    """Offline scoring set: frames x (harmful + benign + blank) templates -> pairs.parquet with a `frame` column.
    'first' = the rendered initial state; 'mid' = mid-trajectory frames (--mid-from). The 2x2 frame x instruction
    table separates refusal keyed on the instruction from refusal keyed on how the scene looks."""
    states = pd.read_parquet(args.states).assign(frame="first")
    ins = pd.read_csv(args.instructions, keep_default_na=False)
    if args.template_split:
        ins = ins[ins["split"].isin(args.template_split) | (ins["class"] == "blank")]
    frames = [states]
    if args.mid_from:
        mid = mid_frames(args.mid_from, args.mid_per_state, args.min_t, np.random.default_rng(args.seed)).assign(frame="mid")
        assert set(mid["state_id"]) <= set(states["state_id"]), "mid frames from states outside --states"
        frames.append(mid)
    df = expand_instructions(pd.concat(frames, ignore_index=True), ins)
    write_parquet(df, args.out)
    print(f"{len(df)} pairs ({df.groupby(['frame', 'cls']).size().to_dict()}) -> {args.out}")


# --- RLDS export (pod: TensorFlow) --------------------------------------------------------------


def cmd_export_rlds(args):
    """Read N trajectories from a modified_libero_rlds builder dir and write rows in our contract.
    RLDS gripper is raw robosuite (-1 open / +1 close); apply libero_dataset_transform's clip + invert -> +1 open.
    RLDS images are 256x256, already rotated 180° by the builder: resized to 224 exactly as OpenVLA's RLDS pipeline
    does (decode_and_resize -> dlimp resize_image). Stored uncropped; training augments, evaluation center-crops."""
    import dlimp as dl
    import tensorflow_datasets as tfds

    root = Path(args.rlds)
    version = sorted(p for p in root.iterdir() if p.is_dir())[-1] if (root / "1.0.0").exists() is False else root / "1.0.0"
    builder = tfds.builder_from_directory(str(version))
    ds = builder.as_dataset(split="train", shuffle_files=False)
    rng = np.random.default_rng(args.seed)
    per_task: dict[str, int] = {}
    rows, kept = [], 0
    episodes = list(ds.take(args.scan))
    rng.shuffle(episodes)
    for ep_i, ep in enumerate(episodes):
        lang = next(iter(ep["steps"].take(1)))["language_instruction"].numpy().decode().lower()
        if args.per_task is not None:
            if per_task.get(lang, 0) >= args.per_task:
                continue
            per_task[lang] = per_task.get(lang, 0) + 1
        elif kept >= args.episodes:
            break
        kept += 1
        steps = list(ep["steps"])  # decode only the episodes we keep
        for t, st in enumerate(steps):
            img = dl.transforms.resize_image(st["observation"]["image"], size=(224, 224)).numpy()
            a = st["action"].numpy().astype(np.float64)
            a[6] = 1.0 - float(np.clip(a[6], 0.0, 1.0))
            rows.append(dict(image=image_to_png_bytes(img), instruction=lang, action=a.tolist(),
                             category=args.category, source_task=lang, source_episode=ep_i, source_t=t))
    df = pd.DataFrame(rows)
    write_parquet(df, args.out)
    print(f"{kept} episodes, {len(df)} transitions, tasks={df['source_task'].nunique()} -> {args.out}")


# --- mix ---------------------------------------------------------------------------------------


def noop_rows(move: pd.DataFrame, instructions: pd.DataFrame, rng) -> pd.DataFrame:
    """Counterfactual pairs: every movement frame again, now with a harmful TRAIN template and the no-op label
    (zero motion, gripper held). Same image, opposite label: only the instruction can explain the difference."""
    harmful = instructions[(instructions["class"] == "harmful") & (instructions["split"] == "train")]
    tmpl = harmful.sample(n=len(move), replace=True, random_state=int(rng.integers(1 << 31)))
    df = move.copy()
    df["instruction"] = [t.replace("{task}", task) for t, task in zip(tmpl["text"], df["task"])]
    df["template_id"] = tmpl["template_id"].values
    df["action"] = df["gripper_state"].apply(lambda g: [0, 0, 0, 0, 0, 0, float(g)])
    df["category"] = "noop"
    return df


def load_self_rollouts(dirs, read_steps) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Steps and episodes from several run dirs; on a duplicate (state, template) key the LATER dir wins and the
    earlier dir's episode AND its steps are dropped (a re-run's clean episode must not inherit the old run's costs)."""
    eps = [pd.read_parquet(Path(d) / "episodes.parquet") for d in dirs]
    stp = [read_steps(d) for d in dirs]
    for i in range(len(dirs) - 1):
        later = set().union(*[set(zip(e["state_id"], e["template_id"])) for e in eps[i + 1:]])
        eps[i] = eps[i][[(a, b) not in later for a, b in zip(eps[i]["state_id"], eps[i]["template_id"])]]
        stp[i] = stp[i][[(a, b) not in later for a, b in zip(stp[i]["state_id"], stp[i]["template_id"])]]
    return pd.concat(stp, ignore_index=True), pd.concat(eps, ignore_index=True)


def move_rows(steps: pd.DataFrame, episodes: pd.DataFrame, instructions: pd.DataFrame, train_states: set,
              n_target: int, per_episode: int, rng, exclude: set | None = None) -> pd.DataFrame:
    """Violation-free self-rollout steps under benign TRAIN templates on TRAIN states, <= per_episode rows per
    episode (always the first frame, the rest uniform over the trajectory), successes preferred. Image and gripper
    are the pre-step frame the action was taken from."""
    benign_train = instructions[(instructions["class"] == "benign") & (instructions["split"] == "train")]
    train_ids = set(benign_train["template_id"]) | {"scripted"}
    # violation-free means NO constraint fired at any step: robot-hand contact (episodes.contact) AND carried-object
    # contact (CheckContact), which the rollout records per step in cost_* columns but does not terminate on
    cost_cols = [c for c in steps.columns if c.startswith("cost_")]
    fired = steps.groupby(["state_id", "template_id"])[cost_cols].max().max(axis=1) if cost_cols else pd.Series(dtype=float)
    dirty = set(fired[fired > 0].index)
    episodes = episodes[[(a, b) not in dirty for a, b in zip(episodes["state_id"], episodes["template_id"])]]
    ok = episodes[(~episodes["contact"]) & (episodes["cls"] == "benign") & episodes["template_id"].isin(train_ids)]
    print(f"move_rows: {len(ok)} violation-free episodes ({len(dirty)} excluded for any constraint firing)")
    bad = set(ok["state_id"]) - train_states
    assert not bad, f"self-rollouts on non-train states: {sorted(bad)[:5]}"
    assert "image" in steps.columns, "steps have no images: rerun rollout with --store-images"
    ok = ok.sort_values("success", ascending=False, kind="stable")
    exclude = exclude or set()
    picked = []
    for _, e in ok.iterrows():
        s = steps[(steps["state_id"] == e["state_id"]) & (steps["template_id"] == e["template_id"])]
        s = s[~s["refused_k1"]]  # a movement label must move
        s = s[~s.apply(lambda r: (r["state_id"], r["template_id"], int(r["t"])) in exclude, axis=1)] if len(s) else s
        if len(s) == 0:
            continue
        first = s[s["t"] == s["t"].min()]  # the initial scene: paired no-op rows need it (the offline test frames are initial)
        rest = s.drop(first.index)
        picked.append(pd.concat([first, rest.sample(n=min(per_episode - 1, len(rest)), random_state=int(rng.integers(1 << 31)))]))
        if sum(len(p) for p in picked) >= n_target:
            break
    df = pd.concat(picked).head(n_target) if picked else pd.DataFrame(columns=steps.columns)
    df = df.copy()
    df["action"] = df["action_model"]
    df["category"] = "move"
    df["src_template_id"] = df["template_id"]  # the step's key in `steps`, before scripted rows are relabelled
    # scripted rows carry the bare task language; assign a benign train template per row so the movement half
    # sees the same instruction distribution (incl. the hard negative) as the no-op half sees harmful templates
    if len(df) and (df["template_id"] == "scripted").any():
        m = df["template_id"] == "scripted"
        tmpl = benign_train.sample(n=int(m.sum()), replace=True, random_state=int(rng.integers(1 << 31)))
        df.loc[m, "instruction"] = [t.replace("{task}", task) for t, task in zip(tmpl["text"], df.loc[m, "instruction"])]
        df.loc[m, "template_id"] = tmpl["template_id"].values
    return df[["image", "instruction", "action", "category", "state_id", "template_id", "src_template_id", "t", "task", "gripper_state"]]


def cmd_mix(args):
    """Bake one training Parquet per arm. A: move + the same frames as no-op pairs + rehearsal. C: the same move and
    rehearsal rows, with the no-op slots filled by additional (disjoint) movement and rehearsal rows. Same row count,
    same update count."""
    from ftr.rollout import read_steps

    rng = np.random.default_rng(args.seed)
    ins = pd.read_csv(args.instructions, keep_default_na=False)
    from ftr.rollout import parse_states

    splits = pd.read_csv(args.splits)
    train = splits[splits["split"] == "train"]
    train_states = {f"{r.suite}/{r.level}/{r.task_idx}/{si}" for r in train.itertuples()
                    for si in parse_states(args.train_states, 10**6)}
    # one or more scripted/self-rollout run dirs (e.g. a task-0 re-run with a higher path); later dirs win on duplicate keys
    dirs = [d for d in args.self if (Path(d) / "episodes.parquet").exists()]
    print(f"self-rollout dirs: {dirs}" + (f" (missing: {[d for d in args.self if d not in dirs]})" if len(dirs) < len(args.self) else ""))
    steps, episodes = load_self_rollouts(dirs, read_steps)
    rehearsal = pd.read_parquet(args.rehearsal)[["image", "instruction", "action", "category"]].copy()
    rehearsal["category"] = "rehearsal"
    move = move_rows(steps, episodes, ins, train_states, args.n_move, args.per_episode, rng)
    reh = rehearsal.sample(n=min(args.n_rehearsal, len(rehearsal)), random_state=args.seed)
    n_pairs = len(move)  # no-op rows pair every movement frame; C replaces exactly that many rows
    if n_pairs < args.n_move:
        print(f"WARNING: {n_pairs} movement frames available, {args.n_move} wanted (few violation-free episodes)")
    if args.arm == "A":
        parts = [noop_rows(move, ins, rng), move, reh]
    else:  # C
        used = set(zip(move["state_id"], move["src_template_id"], move["t"].astype(int)))
        extra_move = move_rows(steps, episodes, ins, train_states, n_pairs // 2, args.per_episode,
                               np.random.default_rng(args.seed + 1), exclude=used)
        n_extra_reh = n_pairs - len(extra_move)
        extra_reh = rehearsal.drop(reh.index).sample(n=min(n_extra_reh, len(rehearsal) - len(reh)), random_state=args.seed + 1)
        parts = [move, reh, extra_move, extra_reh]
    cols = ["image", "instruction", "action", "category"]
    df = pd.concat([p[cols] for p in parts], ignore_index=True).sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    write_parquet(df, args.out)
    print(f"arm {args.arm}: {df['category'].value_counts().to_dict()} -> {args.out}")


def main():
    """Subcommands: render | pairs | export-rlds | mix."""
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("render"); r.add_argument("--suite", required=True); r.add_argument("--level", type=int, default=0)
    r.add_argument("--tasks", nargs="+", required=True); r.add_argument("--states", default="all"); r.add_argument("--out", required=True)
    p = sub.add_parser("pairs"); p.add_argument("--states", required=True); p.add_argument("--instructions", default="manifests/instructions.csv")
    p.add_argument("--template-split", nargs="*", default=None); p.add_argument("--out", required=True)
    p.add_argument("--mid-from", default=None, help="rollout dir with stored images: add mid-trajectory frames")
    p.add_argument("--mid-per-state", type=int, default=1); p.add_argument("--min-t", type=int, default=20); p.add_argument("--seed", type=int, default=0)
    e = sub.add_parser("export-rlds"); e.add_argument("--rlds", required=True); e.add_argument("--episodes", type=int, default=30)
    e.add_argument("--per-task", type=int, default=None); e.add_argument("--scan", type=int, default=5000); e.add_argument("--seed", type=int, default=0)
    e.add_argument("--category", default="personalize"); e.add_argument("--out", required=True)
    m = sub.add_parser("mix"); m.add_argument("--arm", choices=["A", "C"], required=True); m.add_argument("--splits", default="manifests/splits.csv")
    m.add_argument("--instructions", default="manifests/instructions.csv"); m.add_argument("--self", nargs="+", required=True, help="scripted/self-rollout run dir(s); later dirs override earlier on the same (state, template)")
    m.add_argument("--rehearsal", required=True)
    m.add_argument("--n-move", type=int, default=300, help="movement frames; A pairs each with a no-op row")
    m.add_argument("--n-rehearsal", type=int, default=300); m.add_argument("--train-states", default="0-29", help="per training task; 30-36 are dev")
    m.add_argument("--per-episode", type=int, default=5); m.add_argument("--seed", type=int, default=0); m.add_argument("--out", required=True)
    args = ap.parse_args()
    {"render": cmd_render, "pairs": cmd_pairs, "export-rlds": cmd_export_rlds, "mix": cmd_mix}[args.cmd](args)


if __name__ == "__main__":
    main()
