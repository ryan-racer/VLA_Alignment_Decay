"""Training rows -> OpenVLA examples. Mirrors RLDSBatchTransform / DummyDataset (openvla @ c8f03f4,
prismatic/vla/datasets/datasets.py L38-67, L180-232) exactly; the parity test on the pod proves it.

Row contract (one Parquet per arm, written by build_data.py):
  image        PNG bytes, 224x224, already through the EVAL preprocessing (180° rotate, lanczos resize,
               0.9 center-crop-and-resize) so train and test pixels go through the same function
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
    """Map-style Dataset over one arm's Parquet (see row contract above); yields DummyDataset-shaped dicts."""
    def __init__(self, parquet_path: str | Path, tokenizer, image_transform, codec: Codec | None = None):
        self.df = pd.read_parquet(parquet_path)
        for col in ("image", "instruction", "action"):
            assert col in self.df.columns, f"missing column {col}"
        self.tokenizer, self.image_transform = tokenizer, image_transform
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
        return build_example(
            self.tokenizer, self.image_transform, self.codec, image, str(row["instruction"]), np.asarray(row["action"])
        )


def image_to_png_bytes(img: np.ndarray) -> bytes:
    """uint8 HxWx3 -> PNG bytes for a Parquet cell."""
    buf = io.BytesIO()
    Image.fromarray(np.asarray(img, dtype=np.uint8)).save(buf, format="PNG")
    return buf.getvalue()
