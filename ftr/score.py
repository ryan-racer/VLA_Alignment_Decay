"""Offline generated-action scoring: pairs.parquet x checkpoint -> predictions.parquet. Model-only (no simulator),
so 2-3 copies fit on one 80 GB card. BLIND-WRITTEN; first run is Phase 2.

    python -m ftr.score --ckpt runs/A_s0 --pairs data/pairs_test.parquet --out runs/A_s0/score_test
"""

from __future__ import annotations

import argparse
import io
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from ftr.codec import Codec
from ftr.rollout import load_policy, predict


def main():
    """CLI: every pair -> one prediction row with refused_k0/1/2, gripper rule, RoboShackles flag."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    vla, processor = load_policy(args.ckpt)
    codec = Codec()
    pairs = pd.read_parquet(args.pairs)
    rows, t0 = [], time.time()
    for i, r in pairs.iterrows():
        img = np.asarray(Image.open(io.BytesIO(r["image"])).convert("RGB"))
        action, ids = predict(vla, processor, img, r["instruction"])
        g_expected = int(codec.to_token_ids(codec.noop_label(r["gripper_state"]))[6])
        rows.append(dict(
            ckpt=args.ckpt, state_id=r["state_id"], task_idx=r["task_idx"], cls=r["cls"], template_id=r["template_id"],
            instruction=r["instruction"], token_ids=ids.astype(np.int64).tolist(), action=action.tolist(),
            refused_k0=codec.refused(ids, 0), refused_k1=codec.refused(ids, 1), refused_k2=codec.refused(ids, 2),
            refused_k1_gripper=codec.refused(ids, 1, gripper_expected_id=g_expected),
            roboshackles_noop=codec.roboshackles_noop(action), motion_norm=float(np.linalg.norm(action[:3])),
        ))
        if (i + 1) % 50 == 0:
            print(f"{i+1}/{len(pairs)} ({(time.time()-t0)/(i+1):.2f} s/pred)")
    df = pd.DataFrame(rows)
    df.to_parquet(out / "predictions.parquet")
    (out / "args.json").write_text(json.dumps({**vars(args), "n": len(df)}, indent=2))
    summary = df.groupby("cls")[["refused_k0", "refused_k1", "refused_k2", "roboshackles_noop"]].mean()
    print(summary.round(3))


if __name__ == "__main__":
    main()
