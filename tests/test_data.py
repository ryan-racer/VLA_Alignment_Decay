"""Runs on the Mac (CPU processor, no weights). Checks the label contract that the pod parity test then
compares against the real RLDSBatchTransform."""

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from ftr import _openvla
from ftr.codec import Codec
from ftr.data import IGNORE_INDEX, ParquetTransitions, Processor, build_example, image_to_png_bytes


@pytest.fixture(scope="module")
def proc():
    return Processor()


@pytest.fixture(scope="module")
def codec():
    return Codec()


def _img(seed=0):
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 255, (224, 224, 3), dtype=np.uint8))


def test_example_shapes_and_mask(proc, codec):
    action = np.array([0.01, -0.02, 0.0, 0.1, 0.0, 0.0, 1.0])
    ex = build_example(proc.tokenizer, proc.image_transform, codec, _img(), "put both moka pots on the stove", action)
    assert ex["pixel_values"].shape == (6, 224, 224)  # fused DINO+SigLIP: two 3-channel stacks
    supervised = (ex["labels"] != IGNORE_INDEX).sum().item()
    assert supervised == 8  # 7 action tokens + </s>
    assert ex["labels"][-1].item() == proc.tokenizer.eos_token_id
    assert ex["input_ids"][0].item() == proc.tokenizer.bos_token_id


def test_prompt_text_is_the_stock_template(proc, codec):
    ex = build_example(proc.tokenizer, proc.image_transform, codec, _img(), "Put BOTH moka pots on the stove", np.zeros(7))
    text = proc.tokenizer.decode(ex["input_ids"], skip_special_tokens=False)
    assert "In: What action should the robot take to put both moka pots on the stove?\nOut:" in text
    assert text.endswith("</s>")


def test_action_tokens_are_the_codec_tokens(proc, codec):
    action = np.array([0.01, -0.02, 0.0, 0.1, 0.0, 0.0, 1.0])
    ex = build_example(proc.tokenizer, proc.image_transform, codec, _img(), "x", action)
    assert ex["labels"][-8:-1].tolist() == codec.to_token_ids(codec.normalize(action)).tolist()


def test_noop_row_encodes_zero_bins(proc, codec):
    ex = build_example(proc.tokenizer, proc.image_transform, codec, _img(), "x", np.array([0, 0, 0, 0, 0, 0, 1.0]))
    ids = ex["labels"][-8:-1].numpy()
    assert codec.token_ids_to_center_idx(ids)[:6].tolist() == codec.zero_bins().tolist()
    assert codec.refused(ids, k=0)


def test_parquet_dataset_and_collator(tmp_path, proc, codec):
    rows = [
        dict(image=image_to_png_bytes(np.asarray(_img(i))), instruction=f"task {i}", action=np.random.rand(7).tolist(), category="move")
        for i in range(3)
    ]
    rows[1]["action"] = [0, 0, 0, 0, 0, 0, 0.0]
    p = tmp_path / "train.parquet"
    pd.DataFrame(rows).to_parquet(p)
    ds = ParquetTransitions(p, proc.tokenizer, proc.image_transform, codec)
    assert len(ds) == 3
    assert list(ds.dataset_statistics) == ["libero_spatial"]
    collate = _openvla.collator_cls()(proc.tokenizer.model_max_length, proc.tokenizer.pad_token_id, padding_side="right")
    batch = collate([ds[0], ds[1], ds[2]])
    assert batch["pixel_values"].shape == (3, 6, 224, 224)
    assert batch["input_ids"].shape == batch["labels"].shape == batch["attention_mask"].shape
    # padded positions are ignored in the loss and masked in attention
    pad = batch["input_ids"] == proc.tokenizer.pad_token_id
    assert torch.all(batch["labels"][pad] == IGNORE_INDEX)
    assert torch.all(~batch["attention_mask"][pad])


def test_load_self_rollouts_later_dir_wins(tmp_path):
    from pathlib import Path

    from ftr.build_data import load_self_rollouts

    def mk(d, rows):
        d.mkdir()
        pd.DataFrame([dict(state_id=s, template_id="scripted", contact=c) for s, c, _ in rows]).to_parquet(d / "episodes.parquet")
        pd.DataFrame([dict(state_id=s, template_id="scripted", t=0, cost_x=x) for s, _, x in rows]).to_parquet(d / "steps_t0.parquet")
    mk(tmp_path / "first", [("t0/0", True, 1.0), ("t1/0", False, 0.0)])
    mk(tmp_path / "rerun", [("t0/0", False, 0.0)])
    read = lambda d: pd.read_parquet(Path(d) / "steps_t0.parquet")
    steps, eps = load_self_rollouts([tmp_path / "first", tmp_path / "rerun"], read)
    assert len(eps) == 2 and not eps.set_index("state_id").loc["t0/0", "contact"]
    assert steps.groupby("state_id")["cost_x"].max().to_dict() == {"t0/0": 0.0, "t1/0": 0.0}


def test_mix_extra_move_rows_are_disjoint():
    """C's extra movement rows must not reuse the shared movement rows, even though scripted rows are relabelled."""
    from ftr.build_data import move_rows

    ins = pd.DataFrame([dict(template_id="b1", **{"class": "benign"}, split="train", text="{task}"),
                        dict(template_id="b2", **{"class": "benign"}, split="train", text="{task} carefully")])
    states = [f"t0/{i}" for i in range(4)]
    eps = pd.DataFrame([dict(state_id=s, template_id="scripted", cls="benign", contact=False, success=True) for s in states])
    steps = pd.DataFrame([dict(state_id=s, template_id="scripted", t=t, image=b"x", instruction="pick it", task="pick it",
                               gripper_state=1.0, action_model=[0.1] * 7, refused_k1=False, cost_x=0.0)
                          for s in states for t in range(10)])
    rng = np.random.default_rng(0)
    move = move_rows(steps, eps, ins, set(states), 12, 3, rng)
    assert (move["template_id"] != "scripted").all() and (move["src_template_id"] == "scripted").all()
    used = set(zip(move["state_id"], move["src_template_id"], move["t"].astype(int)))
    extra = move_rows(steps, eps, ins, set(states), 12, 3, np.random.default_rng(1), exclude=used)
    assert len(extra) and not (set(zip(extra["state_id"], extra["src_template_id"], extra["t"].astype(int))) & used)


def test_noop_rows_pair_every_movement_frame():
    """Counterfactual pairs: each movement frame reappears with a harmful template and the no-op label; first
    frames are always among the movement frames."""
    from ftr.build_data import move_rows, noop_rows

    ins = pd.read_csv("manifests/instructions.csv", keep_default_na=False)
    states = [f"t0/{i}" for i in range(4)]
    eps = pd.DataFrame([dict(state_id=s, template_id="scripted", cls="benign", contact=False, success=True) for s in states])
    steps = pd.DataFrame([dict(state_id=s, template_id="scripted", t=t, image=f"{s}-{t}".encode(), instruction="put the bowl away",
                               task="put the bowl away", gripper_state=0.0 if t > 5 else 1.0, action_model=[0.2] * 6 + [1.0],
                               refused_k1=False, cost_x=0.0) for s in states for t in range(12)])
    move = move_rows(steps, eps, ins, set(states), 20, 5, np.random.default_rng(0))
    assert (move.groupby("state_id")["t"].min() == 0).all()  # the initial frame of every episode
    noop = noop_rows(move, ins, np.random.default_rng(0))
    assert list(noop["image"]) == list(move["image"])  # same frames, opposite label
    assert set(noop["template_id"]) <= {"h1", "h2", "h3", "h4"} and (noop["category"] == "noop").all()
    assert all(a[:6] == [0] * 6 and a[6] == g for a, g in zip(noop["action"], noop["gripper_state"]))
    with_task = noop[noop["template_id"].isin(["h3", "h4"])]
    assert len(with_task) and with_task["instruction"].str.contains("put the bowl away").all()
    assert not noop["instruction"].str.contains("{task}", regex=False).any()


def test_mid_frames_only_stored_images_before_the_violation(tmp_path):
    """--store-every 10 leaves image=None on 9 of 10 steps; mid frames must come from stored frames with
    t >= min_t that precede the episode's first violation."""
    from ftr.build_data import mid_frames

    rows = [dict(state_id="fshoa/0/3/0", template_id="b5", task_idx=3, task="x", t=t, gripper_state=1.0,
                 image=(b"png%d" % t) if t % 10 == 0 else None) for t in range(60)]
    pd.DataFrame(rows).to_parquet(tmp_path / "steps_t3_s0_b5.parquet")
    pd.DataFrame([dict(state_id="fshoa/0/3/0", template_id="b5", violation_step=45)]).to_parquet(tmp_path / "episodes.parquet")
    for seed in range(20):
        m = mid_frames(str(tmp_path), 1, 20, np.random.default_rng(seed))
        assert len(m) == 1 and m["image"].notna().all() and m.iloc[0]["t"] in (20, 30, 40)
