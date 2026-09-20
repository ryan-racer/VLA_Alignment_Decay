#!/usr/bin/env bash
# Idempotent environment setup for one RunPod A100 80GB with a network volume at /workspace.
# Base image: RunPod "pytorch 2.2.0 / py3.10 / cuda 12.1.1 devel" (confirm the exact tag on day 0).
# Everything lives on /workspace so a pod swap costs nothing. Re-running is safe.
#
#   git clone <this repo> /workspace/repo && bash /workspace/repo/scripts/setup_pod.sh
#   source /workspace/env.sh && pytest /workspace/repo/tests/test_env.py -v
set -euo pipefail

W=/workspace
REPO=$W/repo
OPENVLA_SHA=c8f03f4
LIBERO_SHA=8f1084e3132a39270c3a13ebe37270a43ece2a01
LIBERO_SAFETY_SHA=19ec8df23eedfbb9265bafd3e56495fcebfcfcd0

# ---- apt (EGL headless rendering, flash-attn build, fork extras) ----------------------------
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq cmake ninja-build ffmpeg libegl1 libgl1 libglew-dev libosmesa6-dev \
    linux-headers-generic libmagickwand-dev imagemagick libfontconfig1-dev unzip git-lfs >/dev/null

# ---- venv on the volume ----------------------------------------------------------------------
[ -d $W/venv ] || python3.10 -m venv $W/venv
source $W/venv/bin/activate
pip install -q --upgrade pip

# ---- torch 2.2.0 cu121 (README's pytorch-cuda=12.4 line conflicts with the pyproject pin) ------
python -c "import torch; assert torch.__version__.startswith('2.2.')" 2>/dev/null || \
    pip install -q torch==2.2.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/cu121

# ---- openvla @ pinned commit (brings transformers 4.40.1, peft 0.11.1, tf 2.15, draccus 0.8.0) ---
if [ ! -d $W/openvla ]; then
    git clone -q https://github.com/openvla/openvla.git $W/openvla
fi
git -C $W/openvla checkout -q $OPENVLA_SHA
pip install -q -e $W/openvla

# ---- flash-attn 2.5.5: no cu12 wheel for torch2.2/cp310, so build once (tens of minutes) -------
python -c "import flash_attn" 2>/dev/null || \
    { pip install -q packaging ninja; MAX_JOBS=4 pip install flash-attn==2.5.5 --no-build-isolation; }

# ---- LIBERO deps the OpenVLA way (never LIBERO's own requirements.txt: transformers 4.21, torch 1.11)
pip install -q -r $W/openvla/experiments/robot/libero/libero_requirements.txt

# ---- transitive pins upstream leaves loose --------------------------------------------------
pip install -q numpy==1.26.4 mujoco==2.3.7 tensorflow-metadata==1.14.0 wandb==0.17.9 \
    scipy pyarrow pandas seaborn pytest

# ---- LIBERO (utility suites) and the LIBERO-Safety fork (FSHOA scenes) at pinned SHAs ---------
# Both register the top-level package `libero`. Neither is pip-installed; select with PYTHONPATH.
[ -d $W/LIBERO ] || git clone -q https://github.com/Lifelong-Robot-Learning/LIBERO.git $W/LIBERO
git -C $W/LIBERO checkout -q $LIBERO_SHA
[ -d $W/LIBERO-Safety ] || git clone -q https://github.com/LIBERO-SAFETY/LIBERO-Safety.git $W/LIBERO-Safety
git -C $W/LIBERO-Safety checkout -q $LIBERO_SAFETY_SHA
# Issue #3: check_robot_contact never fires as shipped. Apply once.
git -C $W/LIBERO-Safety apply --check $REPO/patches/libero_safety_issue3.patch 2>/dev/null && \
    git -C $W/LIBERO-Safety apply $REPO/patches/libero_safety_issue3.patch || true
# fork extras ONLY — never its requirements.txt, never its third_party/robosuite-1.4 (osc_pose ±2 / kp 750)
pip install -q "usd-core>=25.5" wand scikit-image

# ---- ~/.libero/config.yaml: without it the first import calls input() and hangs ---------------
mkdir -p ~/.libero
if [ ! -f ~/.libero/config.yaml ]; then
cat > ~/.libero/config.yaml <<EOF
benchmark_root: $W/LIBERO/libero/libero
bddl_files: $W/LIBERO/libero/libero/bddl_files
init_states: $W/LIBERO/libero/libero/init_files
datasets: $W/data/libero
assets: $W/LIBERO/libero/libero/assets
EOF
fi

# ---- artifacts (HF cache on the volume) -----------------------------------------------------
export HF_HOME=$W/hf
pip install -q -U huggingface_hub
[ -f $W/hf/P/config.json ] || hf download openvla/openvla-7b-finetuned-libero-spatial --local-dir $W/hf/P
[ -d $W/hf/rlds/libero_spatial_no_noops ] || hf download openvla/modified_libero_rlds --repo-type dataset \
    --include "libero_spatial_no_noops/*" --include "libero_object_no_noops/*" --local-dir $W/hf/rlds
if [ ! -d $W/LIBERO-Safety/libero/libero/assets/.unpacked ]; then
    hf download LIBERO-Safety/libero_safety_assets --repo-type dataset --local-dir $W/hf/safety_assets
    # the repo ships one zip; unpack into the fork's assets dir
    find $W/hf/safety_assets -name '*.zip' -exec unzip -q -o {} -d $W/LIBERO-Safety/libero/libero/assets/ \;
    mkdir -p $W/LIBERO-Safety/libero/libero/assets/.unpacked
fi

# ---- env file to source per shell ------------------------------------------------------------
cat > $W/env.sh <<EOF
source $W/venv/bin/activate
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl NVIDIA_DRIVER_CAPABILITIES=all
export HF_HOME=$W/hf FTR_P_DIR=$W/hf/P WANDB_MODE=offline
export TOKENIZERS_PARALLELISM=false
# pick ONE libero: LIBERO for libero_spatial/libero_object, LIBERO-Safety for FSHOA
export PYTHONPATH=$REPO:$W/openvla:\${LIBERO_DIR:-$W/LIBERO-Safety}
EOF

echo "done. next:  source $W/env.sh && pytest $REPO/tests/test_env.py -v"
echo "then:        pip freeze > $REPO/requirements.txt"
