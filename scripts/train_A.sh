#!/usr/bin/env bash
# Alignment arm A: harmful no-ops + hazard-scene movement + libero_spatial rehearsal, from P.
# The command line IS the config. Seed 0 is the paper's chain.
set -euo pipefail
source /workspace/env.sh
SEED=${1:-0}
cd /workspace/repo
python -m ftr.finetune \
  --vla_path /workspace/hf/P \
  --data_parquet data/A.parquet \
  --run_root_dir runs --run_id A_s$SEED \
  --epochs 3 --batch_size 8 --grad_accumulation_steps 2 --learning_rate 5e-4 \
  --lora_rank 32 --seed $SEED
