#!/usr/bin/env bash
# Phase 0 local environment (no GPU): enough to run tests/test_codec.py, test_data.py, test_analyze.py.
# Uses uv to get Python 3.10. openvla is installed with --no-deps so torch/tf/flash-attn never come in.
#
#   bash scripts/setup_mac.sh && source .venv/bin/activate && pytest tests -m "not gpu"
set -euo pipefail

REPO=$(cd "$(dirname "$0")/.." && pwd)
OPENVLA_DIR=${OPENVLA_DIR:-$HOME/code/openvla}
OPENVLA_SHA=c8f03f4
P_DIR=$REPO/.cache/P     # config + tokenizer + processor files only, no weights

cd "$REPO"
[ -d .venv ] || uv venv --python 3.10 .venv
source .venv/bin/activate

# CPU torch is enough for the tokenizer/processor; version-match the pod anyway.
uv pip install -q torch==2.2.2 torchvision==0.17.2 \
    transformers==4.40.1 tokenizers==0.19.1 timm==0.9.10 numpy==1.26.4 \
    draccus==0.8.0 rich jsonlines pandas scipy seaborn pyarrow pytest pillow huggingface_hub

[ -d "$OPENVLA_DIR" ] || git clone -q https://github.com/openvla/openvla.git "$OPENVLA_DIR"
git -C "$OPENVLA_DIR" checkout -q $OPENVLA_SHA
uv pip install -q --no-deps -e "$OPENVLA_DIR"

# P's non-weight files: stats live in config.json; tokenizer + processor for label construction.
if [ ! -f "$P_DIR/config.json" ]; then
    hf download openvla/openvla-7b-finetuned-libero-spatial --local-dir "$P_DIR" \
        --exclude "*.safetensors"
fi

cat > .env.mac <<EOF
export FTR_P_DIR=$P_DIR
export PYTHONPATH=$REPO:$OPENVLA_DIR
EOF
echo "ok. next:  source .venv/bin/activate && source .env.mac && pytest tests -m 'not gpu' -v"
