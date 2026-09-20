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
