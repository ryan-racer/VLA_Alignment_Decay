#!/usr/bin/env bash
# Personalization: fresh rank-32 adapter on a merged A or C checkpoint, N libero_object demos, 3 epochs, no replay.
#   scripts/personalize.sh A_s0 200 0     ->  runs/A_s0_N200_p0   (parent, N, personalization seed)
set -euo pipefail
source /workspace/env.sh
PARENT=${1:?parent run id, e.g. A_s0}
N=${2:?N demos: 50 or 200}
PSEED=${3:-0}
cd /workspace/repo
python -m ftr.finetune \
  --vla_path runs/$PARENT \
  --data_parquet data/object_N${N}_p${PSEED}.parquet \
  --run_root_dir runs --run_id ${PARENT}_N${N}_p${PSEED} \
  --epochs 3 --batch_size 8 --grad_accumulation_steps 2 --learning_rate 5e-4 \
  --lora_rank 32 --seed $PSEED
