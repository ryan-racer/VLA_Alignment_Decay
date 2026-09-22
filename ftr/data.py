"""Training rows -> OpenVLA examples. Mirrors RLDSBatchTransform / DummyDataset (openvla @ c8f03f4,
prismatic/vla/datasets/datasets.py L38-67, L180-232) exactly; the parity test on the pod proves it.

Row contract (one Parquet per arm, written by build_data.py):
  image        PNG bytes, 224x224, the STORED form (envs.model_image(center_crop=False): 180° rotate, JPEG round-trip,
               lanczos resize) = what OpenVLA's RLDS pipeline holds before augmentation. Training applies the stock
               random crop + colour jitter (stock_augment); evaluation applies the stock 0.9 center crop.
  instruction  str (lower-cased at use)
  action       7 floats, raw simulator units on dims 0-5, gripper on dim 6 in RLDS convention: [0,1], +1 = open
  category     'noop' | 'move' | 'rehearsal' | 'personalize'  (informational; the Dataset ignores it)
  source_*     provenance columns (task, state_id, episode, t)
A no-op row is simply action = [0,0,0,0,0,0, gripper]; normalize() turns it into the encoded zero.
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from ftr import _openvla
from ftr.codec import UNNORM_KEY, Codec

IGNORE_INDEX = -100
QUESTION = "What action should the robot take to {instruction}?"


class Processor:
    """Tokenizer + image transform from P's non-weight files. On the pod the stock AutoProcessor gives the
    same two objects; this class exists so the Mac can run test_data.py without weights."""

    def __init__(self, p: Path | None = None):
        from transformers import AutoTokenizer

        from ftr.codec import p_dir

        p = Path(p) if p is not None else p_dir()
        self.tokenizer = AutoTokenizer.from_pretrained(str(p))
        self.image_processor = _openvla.image_processor_cls().from_pretrained(str(p))
        self.image_transform = self.image_processor.apply_transform


def build_example(tokenizer, image_transform, codec: Codec, image: Image.Image, instruction: str, action: np.ndarray):
    """Exactly DummyDataset.__getitem__, with our normalization in front of the action tokenizer."""
    n = codec.normalize(np.asarray(action, dtype=np.float64))
    pb = _openvla.prompt_builder_cls()("openvla")
    pb.add_turn("human", QUESTION.format(instruction=instruction.lower()))
    pb.add_turn("gpt", codec.to_token_str(n))
    input_ids = tokenizer(pb.get_prompt(), add_special_tokens=True).input_ids
    labels = list(input_ids)
    input_ids, labels = torch.tensor(input_ids), torch.tensor(labels)
    pixel_values = image_transform(image)
    labels[: -(len(n) + 1)] = IGNORE_INDEX  # supervise the 7 action tokens + </s>
    return dict(pixel_values=pixel_values, input_ids=input_ids, labels=labels)


class ParquetTransitions(Dataset):
    """Map-style Dataset over one arm's Parquet (see row contract above); yields DummyDataset-shaped dicts.
    image_fn (uint8 HxWx3 -> uint8 HxWx3) runs before the processor: finetune.py passes stock_augment."""
    def __init__(self, parquet_path: str | Path, tokenizer, image_transform, codec: Codec | None = None, image_fn=None):
        self.df = pd.read_parquet(parquet_path)
        for col in ("image", "instruction", "action"):
            assert col in self.df.columns, f"missing column {col}"
        self.tokenizer, self.image_transform, self.image_fn = tokenizer, image_transform, image_fn
        self.codec = codec or Codec()
        # finetune.py writes this next to the checkpoint; predict_action reads it back. Frozen to P's key.
        self.dataset_statistics = {
            UNNORM_KEY: {"action": {k: np.asarray(v).tolist() for k, v in self.codec.stats.items()}}
        }

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        image = Image.open(io.BytesIO(row["image"])).convert("RGB")
        if self.image_fn is not None:
            image = Image.fromarray(self.image_fn(np.asarray(image)))
        return build_example(
            self.tokenizer, self.image_transform, self.codec, image, str(row["instruction"]), np.asarray(row["action"])
        )


def image_to_png_bytes(img: np.ndarray) -> bytes:
    """uint8 HxWx3 -> PNG bytes for a Parquet cell."""
    buf = io.BytesIO()
    Image.fromarray(np.asarray(img, dtype=np.uint8)).save(buf, format="PNG")
    return buf.getvalue()


# openvla @ c8f03f4, prismatic/vla/datasets/datasets.py L122-136 (RLDSDataset, image_aug=True, the finetune.py default)
STOCK_AUGMENT_KWARGS = dict(
    random_resized_crop=dict(scale=[0.9, 0.9], ratio=[1.0, 1.0]),
    random_brightness=[0.2],
    random_contrast=[0.8, 1.2],
    random_saturation=[0.8, 1.2],
    random_hue=[0.05],
    augment_order=["random_resized_crop", "random_brightness", "random_contrast", "random_saturation", "random_hue"],
)
_TF_READY = False


def stock_augment(img: np.ndarray, seed=None) -> np.ndarray:
    """The stock training augmentation: dlimp's augment_image with RLDSDataset's kwargs (called per image, as the
    RLDS frame transform does). TF runs on CPU only, as in prismatic's RLDS module. Pod only (TF + dlimp)."""
    global _TF_READY
    import tensorflow as tf

    if not _TF_READY:
        try:
            tf.config.set_visible_devices([], "GPU")
        except RuntimeError:
            pass  # TF already initialized (e.g. prismatic's RLDS module ran the same call first)
        _TF_READY = True
    import dlimp as dl

    if seed is None:
        seed = torch.randint(0, 2**31 - 1, (2,)).numpy()  # torch RNG: seeded by finetune.py
    seed = tf.constant(np.asarray(seed), dtype=tf.int32)
    return dl.transforms.augment_image(tf.convert_to_tensor(img), **STOCK_AUGMENT_KWARGS, seed=seed).numpy()


def _done_fields(ckpt) -> dict:
    d = Path(str(ckpt)) / "DONE"
    return dict(l.split("=", 1) for l in d.read_text().splitlines() if "=" in l) if d.exists() else {}


def ckpt_uid(ckpt) -> str:
    """Identity of a checkpoint: the run_uid finetune.py writes into DONE (a re-merge of the same adapter keeps it),
    or, for a checkpoint we did not train (P, no DONE), a hash of its config.json. Stored with every result."""
    import hashlib

    f = _done_fields(ckpt)
    if f:
        assert "run_uid" in f, f"{ckpt}/DONE has no run_uid (written by older code): retrain or re-merge it"
        return f["run_uid"]
    return "base-" + hashlib.sha1((Path(str(ckpt)) / "config.json").read_bytes()).hexdigest()[:12]


def ckpt_base_uid(ckpt) -> str:
    """The uid of the checkpoint this one was fine-tuned from ('' for P). analyze checks parent/child lineage."""
    return _done_fields(ckpt).get("base_uid", "")


def write_parquet(df: pd.DataFrame, path) -> None:
    """Atomic Parquet write: a killed process never leaves a truncated file that a done-file check would accept."""
    import os

    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    df.to_parquet(tmp)
    os.replace(tmp, path)
