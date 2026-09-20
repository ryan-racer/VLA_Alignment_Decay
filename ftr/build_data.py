"""Assemble training and evaluation Parquet from (a) rendered states, (b) P self-rollouts, (c) RLDS exports.
BLIND-WRITTEN except the mixing logic; TF-dependent parts run on the pod.

    # 1. held-out and training states rendered once (images + gripper state), all instruction variants
    python -m ftr.build_data render --suite obstacle_avoidance_human --tasks 0 1 2 --states 0-29 --out data/states_train.parquet
    python -m ftr.build_data render --suite obstacle_avoidance_human --tasks 3 4 --states 0-24 --out data/states_test.parquet
    # 2. pairs for offline scoring: test states x (harmful test + benign test + blank)
    python -m ftr.build_data pairs --states data/states_test.parquet --instructions manifests/instructions.csv \
        --template-split test --out data/pairs_test.parquet
    # 3. RLDS subset export (TF): rehearsal and personalization rows
    python -m ftr.build_data export-rlds --rlds /workspace/hf/rlds/libero_spatial_no_noops --episodes 30 --seed 0 --out data/spatial_rehearsal.parquet
    python -m ftr.build_data export-rlds --rlds /workspace/hf/rlds/libero_object_no_noops --per-task 20 --seed 0 --out data/object_N200.parquet
    # 4. the arms
    python -m ftr.build_data mix --arm A --states data/states_train.parquet --self runs/P_self/steps.parquet \
        --self-episodes runs/P_self/episodes.parquet --rehearsal data/spatial_rehearsal.parquet --out data/A.parquet
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from ftr.data import image_to_png_bytes


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
            for si in parse_states(args.states, len(states)):
                obs = envs.reset_to(env, states[si])
                rows.append(dict(suite=args.suite, level=args.level, task_idx=task_idx, task=language, state_idx=si,
                                 state_id=f"{args.suite}/{args.level}/{task_idx}/{si}",
                                 image=image_to_png_bytes(envs.model_image(obs)),
                                 gripper_state=envs.gripper_state_rlds(obs)))
        finally:
            env.close()
    pd.DataFrame(rows).to_parquet(args.out)
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


def cmd_pairs(args):
    """Offline scoring set: rendered states x (harmful + benign + blank) templates -> pairs.parquet."""
    states = pd.read_parquet(args.states)
    ins = pd.read_csv(args.instructions, keep_default_na=False)
    if args.template_split:
        ins = ins[ins["split"].isin(args.template_split) | (ins["class"] == "blank")]
    df = expand_instructions(states, ins)
    df.to_parquet(args.out)
    print(f"{len(df)} pairs ({df['cls'].value_counts().to_dict()}) -> {args.out}")


# --- RLDS export (pod: TensorFlow) --------------------------------------------------------------


def cmd_export_rlds(args):
    """Read N trajectories from a modified_libero_rlds builder dir and write rows in our contract.
    RLDS gripper is raw robosuite (-1 open / +1 close); apply libero_dataset_transform's clip + invert -> +1 open.
    RLDS images are 256x256 already rotated 180° by the builder: resize to 224 with the eval lanczos path + 0.9 crop."""
    import tensorflow as tf
    import tensorflow_datasets as tfds
    from experiments.robot.libero.libero_utils import resize_image
    from experiments.robot.openvla_utils import crop_and_resize

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
        steps = list(ep["steps"])
        lang = steps[0]["language_instruction"].numpy().decode().lower()
        if args.per_task is not None:
            if per_task.get(lang, 0) >= args.per_task:
                continue
            per_task[lang] = per_task.get(lang, 0) + 1
        elif kept >= args.episodes:
            break
        kept += 1
        for t, st in enumerate(steps):
            img = st["observation"]["image"].numpy()
            img = resize_image(img, (224, 224))
            x = tf.image.convert_image_dtype(tf.convert_to_tensor(img), tf.float32)
            img = tf.image.convert_image_dtype(tf.clip_by_value(crop_and_resize(x, 0.9, 1), 0, 1), tf.uint8, saturate=True).numpy()
            a = st["action"].numpy().astype(np.float64)
            a[6] = 1.0 - float(np.clip(a[6], 0.0, 1.0))
            rows.append(dict(image=image_to_png_bytes(img), instruction=lang, action=a.tolist(),
                             category=args.category, source_task=lang, source_episode=ep_i, source_t=t))
    df = pd.DataFrame(rows)
    df.to_parquet(args.out)
    print(f"{kept} episodes, {len(df)} transitions, tasks={df['source_task'].nunique()} -> {args.out}")


# --- mix ---------------------------------------------------------------------------------------


def noop_rows(states: pd.DataFrame, instructions: pd.DataFrame, n_target: int, rng) -> pd.DataFrame:
    """Harmful train templates on training states, action = zeros + current gripper (the refusal label)."""
    harmful = instructions[(instructions["class"] == "harmful") & (instructions["split"] == "train")]
    df = expand_instructions(states, harmful)
    df["action"] = df["gripper_state"].apply(lambda g: [0, 0, 0, 0, 0, 0, float(g)])
    df["category"] = "noop"
    df = df.sample(n=min(n_target, len(df)), random_state=int(rng.integers(1 << 31)))
    return df[["image", "instruction", "action", "category", "state_id", "template_id"]]


def move_rows(steps: pd.DataFrame, episodes: pd.DataFrame, n_target: int, per_episode: int, rng, states: pd.DataFrame | None = None) -> pd.DataFrame:
    """Violation-free P self-rollout steps under benign train templates, ≤ per_episode states per episode,
    successes preferred. Image = the frame the action was taken from (re-rendered in steps.parquet by rollout)."""
    ok = episodes[(~episodes["contact"]) & (episodes["cls"] == "benign")]
    ok = ok.sort_values("success", ascending=False)
    key = ["state_id", "template_id"]
    picked = []
    for _, e in ok.iterrows():
        s = steps[(steps["state_id"] == e["state_id"]) & (steps["template_id"] == e["template_id"])]
        s = s[~s["refused_k1"]]  # a movement label must move
        if len(s) == 0:
            continue
        s = s.sample(n=min(per_episode, len(s)), random_state=int(rng.integers(1 << 31)))
        picked.append(s)
        if sum(len(p) for p in picked) >= n_target:
            break
    df = pd.concat(picked).head(n_target) if picked else pd.DataFrame()
    df = df.rename(columns={"image": "image"})
    df["action"] = df["action_model"]
    df["category"] = "move"
    return df[["image", "instruction", "action", "category", "state_id", "template_id"]]


def cmd_mix(args):
    """Bake one training Parquet per arm. A: noop+move+rehearsal. C: same move/rehearsal rows, noop slots -> more movement."""
    rng = np.random.default_rng(args.seed)
    states = pd.read_parquet(args.states)
    ins = pd.read_csv(args.instructions, keep_default_na=False)
    steps = pd.read_parquet(args.self)
    episodes = pd.read_parquet(args.self_episodes)
    rehearsal = pd.read_parquet(args.rehearsal)[["image", "instruction", "action", "category"]]
    rehearsal["category"] = "rehearsal"
    move = move_rows(steps, episodes, args.n_move, args.per_episode, rng)
    reh = rehearsal.sample(n=min(args.n_rehearsal, len(rehearsal)), random_state=args.seed)
    if args.arm == "A":
        noop = noop_rows(states, ins, args.n_noop, rng)
        parts = [noop, move, reh]
    else:  # C: same movement rows, no-op slots filled with more movement + rehearsal
        extra_move = move_rows(steps, episodes, args.n_noop // 2, args.per_episode, np.random.default_rng(args.seed + 1))
        extra_reh = rehearsal.drop(reh.index).sample(n=min(args.n_noop - len(extra_move), len(rehearsal) - len(reh)), random_state=args.seed + 1)
        parts = [move, reh, extra_move, extra_reh]
    df = pd.concat(parts, ignore_index=True).sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    df.to_parquet(args.out)
    print(f"arm {args.arm}: {df['category'].value_counts().to_dict()} -> {args.out}")


def main():
    """Subcommands: render | pairs | export-rlds | mix."""
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("render"); r.add_argument("--suite", required=True); r.add_argument("--level", type=int, default=0)
    r.add_argument("--tasks", nargs="+", required=True); r.add_argument("--states", default="all"); r.add_argument("--out", required=True)
    p = sub.add_parser("pairs"); p.add_argument("--states", required=True); p.add_argument("--instructions", default="manifests/instructions.csv")
    p.add_argument("--template-split", nargs="*", default=None); p.add_argument("--out", required=True)
    e = sub.add_parser("export-rlds"); e.add_argument("--rlds", required=True); e.add_argument("--episodes", type=int, default=30)
    e.add_argument("--per-task", type=int, default=None); e.add_argument("--scan", type=int, default=5000); e.add_argument("--seed", type=int, default=0)
    e.add_argument("--category", default="personalize"); e.add_argument("--out", required=True)
    m = sub.add_parser("mix"); m.add_argument("--arm", choices=["A", "C"], required=True); m.add_argument("--states", required=True)
    m.add_argument("--instructions", default="manifests/instructions.csv"); m.add_argument("--self", required=True)
    m.add_argument("--self-episodes", required=True); m.add_argument("--rehearsal", required=True)
    m.add_argument("--n-noop", type=int, default=300); m.add_argument("--n-move", type=int, default=300); m.add_argument("--n-rehearsal", type=int, default=300)
    m.add_argument("--per-episode", type=int, default=5); m.add_argument("--seed", type=int, default=0); m.add_argument("--out", required=True)
    args = ap.parse_args()
    {"render": cmd_render, "pairs": cmd_pairs, "export-rlds": cmd_export_rlds, "mix": cmd_mix}[args.cmd](args)


if __name__ == "__main__":
    main()
