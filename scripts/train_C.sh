#!/usr/bin/env bash
# Comparator C: same movement rows as A, no-op slots replaced by movement/rehearsal, same update count, from P.
set -euo pipefail
source /workspace/env.sh
SEED=${1:-0}
cd /workspace/repo
python -m ftr.finetune \
  --vla_path /workspace/hf/P \
  --data_parquet data/C.parquet \
  --run_root_dir runs --run_id C_s$SEED \
  --epochs 3 --batch_size 8 --grad_accumulation_steps 2 --learning_rate 5e-4 \
  --lora_rank 32 --seed $SEED
