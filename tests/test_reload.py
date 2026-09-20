"""Pod, after the first training run: the merged checkpoint has exactly one norm_stats key and agrees with
the unmerged adapter on fixed observations within tolerance (bf16 merge is not bit-exact).

    FTR_RUN=runs/A_s0 FTR_ADAPTER=adapter-tmp/A_s0 pytest tests/test_reload.py -m gpu
"""

import io
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

pytestmark = pytest.mark.gpu

RUN = Path(os.environ.get("FTR_RUN", "runs/A_s0"))
ADAPTER = Path(os.environ.get("FTR_ADAPTER", "adapter-tmp/A_s0"))
PAIRS = Path(os.environ.get("FTR_PAIRS", "data/pairs_test.parquet"))
N_OBS = 20


def test_merged_has_one_norm_stats_key():
    cfg = json.loads((RUN / "config.json").read_text())
    assert list(cfg["norm_stats"]) == ["libero_spatial"], list(cfg["norm_stats"])
    assert (RUN / "dataset_statistics.json").exists()
    for f in ("configuration_prismatic.py", "modeling_prismatic.py", "processing_prismatic.py"):
        assert (RUN / f).exists(), f


def test_merged_agrees_with_unmerged_adapter():
    from peft import PeftModel
    from transformers import AutoModelForVision2Seq

    from ftr.rollout import load_policy, predict

    merged, proc = load_policy(str(RUN))
    base = AutoModelForVision2Seq.from_pretrained("/workspace/hf/P", torch_dtype=torch.bfloat16, trust_remote_code=True,
                                                  attn_implementation="flash_attention_2").to("cuda:0")
    unmerged = PeftModel.from_pretrained(base, str(ADAPTER)).eval()
    unmerged.norm_stats = merged.norm_stats
    unmerged.bin_centers, unmerged.vocab_size = merged.bin_centers, merged.vocab_size
    unmerged.get_action_dim, unmerged.get_action_stats = merged.get_action_dim, merged.get_action_stats
    pairs = pd.read_parquet(PAIRS).head(N_OBS)
    agree, max_delta = 0, 0.0
    for _, r in pairs.iterrows():
        img = np.asarray(Image.open(io.BytesIO(r["image"])).convert("RGB"))
        a1, ids1 = predict(merged, proc, img, r["instruction"])
        a2, ids2 = predict(unmerged, proc, img, r["instruction"])
        agree += int(np.array_equal(ids1, ids2))
        max_delta = max(max_delta, float(np.max(np.abs(a1[:6] - a2[:6]))))
    assert agree / N_OBS >= 0.95, f"token agreement {agree}/{N_OBS}, max |Δaction| {max_delta:.4f}"
