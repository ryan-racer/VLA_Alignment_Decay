"""Pod, after the first training run: the merged checkpoint has exactly one norm_stats key and makes the same refusal
decisions as the unmerged adapter. Stock OpenVLA merges LoRA into bf16 weights, which is not bit-exact: borderline
greedy tokens can flip. What the paper measures is the refusal decision, so that is what must agree (>= 95% of all
pairs); exact token agreement and the largest action difference are reported alongside (they go in the paper).

    FTR_RUN=runs/A_s0 FTR_ADAPTER=adapter-tmp/A_s0 FTR_PAIRS=data/pairs_gate.parquet pytest tests/test_reload.py -m gpu
"""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

pytestmark = pytest.mark.gpu

RUN = Path(os.environ.get("FTR_RUN", "runs/A_s0"))
ADAPTER = Path(os.environ.get("FTR_ADAPTER", "adapter-tmp/A_s0"))
PAIRS = Path(os.environ.get("FTR_PAIRS", "data/pairs_gate.parquet"))
DECISION_FLOOR = 0.95


def test_merged_has_one_norm_stats_key():
    cfg = json.loads((RUN / "config.json").read_text())
    assert list(cfg["norm_stats"]) == ["libero_spatial"], list(cfg["norm_stats"])
    assert (RUN / "dataset_statistics.json").exists()
    for f in ("configuration_prismatic.py", "modeling_prismatic.py", "processing_prismatic.py"):
        assert (RUN / f).exists(), f


def test_merged_agrees_with_unmerged_adapter():
    from peft import PeftModel
    from transformers import AutoModelForVision2Seq

    from ftr import envs
    from ftr.codec import Codec
    from ftr.rollout import load_policy, predict

    merged, proc = load_policy(str(RUN))
    base = AutoModelForVision2Seq.from_pretrained(os.environ["FTR_P_DIR"], torch_dtype=torch.bfloat16, trust_remote_code=True,
                                                  attn_implementation="flash_attention_2").to("cuda:0")
    unmerged = PeftModel.from_pretrained(base, str(ADAPTER)).eval()
    unmerged.norm_stats = merged.norm_stats
    unmerged.bin_centers, unmerged.vocab_size = merged.bin_centers, merged.vocab_size
    unmerged.get_action_dim, unmerged.get_action_stats = merged.get_action_dim, merged.get_action_stats
    codec = Codec()
    pairs = pd.read_parquet(PAIRS)
    same_decision, same_tokens, max_delta = 0, 0, 0.0
    for _, r in pairs.iterrows():
        img = envs.policy_view(r["image"])
        a1, ids1 = predict(merged, proc, img, r["instruction"])
        a2, ids2 = predict(unmerged, proc, img, r["instruction"])
        g = int(codec.to_token_ids(codec.noop_label(r["gripper_state"]))[6])
        same_decision += int(codec.refused(ids1, 1, gripper_expected_id=g) == codec.refused(ids2, 1, gripper_expected_id=g))
        same_tokens += int(np.array_equal(ids1, ids2))
        max_delta = max(max_delta, float(np.max(np.abs(a1[:6] - a2[:6]))))
    n = len(pairs)
    msg = (f"merged vs unmerged adapter on {n} pairs: refusal decision agreement {same_decision}/{n}, "
           f"token agreement {same_tokens}/{n}, max |Δaction| {max_delta:.4f}")
    print(msg)
    (RUN / "merge_agreement.txt").write_text(msg + "\n")
    assert same_decision / n >= DECISION_FLOOR, msg
