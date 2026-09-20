#!/usr/bin/env bash
# Idempotent GPU environment for OpenVLA + LIBERO + LIBERO-Safety. Works on a RunPod pod or a Colab runtime.
#
#   W     local, fast, may be ephemeral: venv, openvla, LIBERO clones          (default /workspace)
#   DATA  persistent, small: runs/, data/, logs/, adapters                     (default $W)
#   HF    where the ~31 GB of public downloads go; local is fine, HF re-serves them in minutes  (default $W/hf)
#   CUDA  cu118 (default; prebuilt flash-attn wheel, no compile) or cu121 (compiles flash-attn, ~30-60 min)
#
#   RunPod:  git clone <repo> /workspace/repo && bash /workspace/repo/scripts/setup_pod.sh
#   Colab:   see notebooks/phase1_colab.ipynb (W=/content/ftr HF=/content/ftr/hf DATA=/content/drive/MyDrive/ftr)
#   then:    source $W/env.sh && pytest $REPO/tests/test_env.py -m gpu -v
set -euo pipefail

W=${W:-/workspace}
DATA=${DATA:-$W}
CUDA=${CUDA:-cu118}
REPO=${REPO:-$W/repo}
HF=${HF:-$W/hf}
OPENVLA_SHA=c8f03f4
LIBERO_SHA=8f1084e3132a39270c3a13ebe37270a43ece2a01
LIBERO_SAFETY_SHA=19ec8df23eedfbb9265bafd3e56495fcebfcfcd0
mkdir -p "$W" "$HF" "$DATA"/{runs,data,logs}

# ---- apt (EGL headless rendering, build tools, fork extras) ----------------------------------
export DEBIAN_FRONTEND=noninteractive
SUDO=$(command -v sudo || true)
$SUDO apt-get update -qq
$SUDO apt-get install -y -qq cmake ninja-build ffmpeg libegl1 libgl1 libglew-dev libosmesa6-dev \
    linux-libc-dev libmagickwand-dev imagemagick libfontconfig1-dev unzip git-lfs >/dev/null

# ---- Python 3.10 venv on local disk (uv fetches the interpreter if needed) -------------------
command -v uv >/dev/null || pip install -q uv
[ -x $W/venv/bin/python ] || uv venv --python 3.10 --python-preference only-managed $W/venv
PY=$W/venv/bin/python
$PY -c 'import sys; assert sys.version_info[:2] == (3, 10), sys.version' || { echo "venv is not Python 3.10: $($PY --version)"; exit 1; }
UVPIP="uv pip install -q --python $PY"   # every install targets the venv explicitly (Colab's system Python is 3.13)
source $W/venv/bin/activate

# ---- torch 2.2.0 --------------------------------------------------------------------------------
$PY -c "import torch; assert torch.__version__.startswith('2.2.')" 2>/dev/null || \
    $UVPIP torch==2.2.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/$CUDA

# ---- openvla @ pinned commit (brings transformers 4.40.1, peft 0.11.1, tf 2.15, draccus 0.8.0) ---
[ -d $W/openvla ] || git clone -q https://github.com/openvla/openvla.git $W/openvla
git -C $W/openvla checkout -q $OPENVLA_SHA
$UVPIP -e $W/openvla

# ---- flash-attn 2.5.5: wheel for cu118 / torch 2.2 / cp310; otherwise build once -------------
if ! $PY -c "import flash_attn" 2>/dev/null; then
    if [ "$CUDA" = "cu118" ]; then
        $UVPIP "https://github.com/Dao-AILab/flash-attention/releases/download/v2.5.5/flash_attn-2.5.5+cu118torch2.2cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"
    else
        $UVPIP packaging ninja; MAX_JOBS=4 $PY -m pip install flash-attn==2.5.5 --no-build-isolation
    fi
fi

# ---- LIBERO deps the OpenVLA way (never LIBERO's own requirements.txt) -------------------------
$UVPIP -r $W/openvla/experiments/robot/libero/libero_requirements.txt

# ---- transitive pins upstream leaves loose + our deps -------------------------------------------
$UVPIP numpy==1.26.4 mujoco==2.3.7 tensorflow-metadata==1.14.0 wandb==0.17.9 \
    scipy pyarrow pandas seaborn matplotlib pytest huggingface_hub

# ---- LIBERO (utility suites) and the LIBERO-Safety fork (FSHOA) at pinned SHAs ----------------
# Both register the top-level package `libero`. Neither is pip-installed; select with PYTHONPATH.
[ -d $W/LIBERO ] || git clone -q https://github.com/Lifelong-Robot-Learning/LIBERO.git $W/LIBERO
git -C $W/LIBERO checkout -q $LIBERO_SHA
[ -d $W/LIBERO-Safety ] || git clone -q https://github.com/LIBERO-SAFETY/LIBERO-Safety.git $W/LIBERO-Safety
git -C $W/LIBERO-Safety checkout -q $LIBERO_SAFETY_SHA
# Issue #3: check_robot_contact never fires as shipped. Apply once (no-op if already applied).
git -C $W/LIBERO-Safety apply --check $REPO/patches/libero_safety_issue3.patch 2>/dev/null && \
    git -C $W/LIBERO-Safety apply $REPO/patches/libero_safety_issue3.patch || true
# fork extras ONLY: never its requirements.txt, never its third_party/robosuite-1.4 (osc_pose ±2 / kp 750)
$UVPIP "usd-core>=25.5" wand scikit-image

# ---- libero config: one per checkout, selected via LIBERO_CONFIG_PATH in env.sh ----------------
# (the file is global by default, so the fork would otherwise look for its BDDLs under upstream LIBERO)
for L in $W/LIBERO $W/LIBERO-Safety; do
    mkdir -p $L/.libero_config
    cat > $L/.libero_config/config.yaml <<EOF
benchmark_root: $L/libero/libero
bddl_files: $L/libero/libero/bddl_files
init_states: $L/libero/libero/init_files
datasets: $DATA/data/libero
assets: $L/libero/libero/assets
EOF
done
mkdir -p ~/.libero && cp $W/LIBERO-Safety/.libero_config/config.yaml ~/.libero/config.yaml  # fallback; avoids input() hang

# ---- public downloads (skipped when already present) -----------------------------------------------
export HF_HOME=$HF
HFDL="$PY -m huggingface_hub.commands.huggingface_cli download"
[ -f $HF/P/config.json ] || $HFDL openvla/openvla-7b-finetuned-libero-spatial --local-dir $HF/P
[ -d $HF/rlds/libero_spatial_no_noops ] || $HFDL openvla/modified_libero_rlds --repo-type dataset \
    --include "libero_spatial_no_noops/*" --include "libero_object_no_noops/*" --local-dir $HF/rlds
[ -d $HF/safety_assets ] || $HFDL LIBERO-Safety/libero_safety_assets --repo-type dataset --local-dir $HF/safety_assets
# the assets unpack into the (local) fork clone; redo each session if W is ephemeral
if [ ! -f $W/LIBERO-Safety/libero/libero/assets/.unpacked ]; then
    find $HF/safety_assets -name '*.zip' -exec unzip -q -o {} -d $W/LIBERO-Safety/libero/libero/assets/ \;
    touch $W/LIBERO-Safety/libero/libero/assets/.unpacked
fi

# ---- env file to source per shell -------------------------------------------------------------------
cat > $W/env.sh <<EOF
source $W/venv/bin/activate
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl NVIDIA_DRIVER_CAPABILITIES=all
export HF_HOME=$HF FTR_P_DIR=$HF/P FTR_RLDS_DIR=$HF/rlds FTR_DATA=$DATA WANDB_MODE=offline
export WANDB_DIR=$DATA/logs TOKENIZERS_PARALLELISM=false
# pick ONE libero: LIBERO_DIR=$W/LIBERO for libero_spatial/libero_object (default: LIBERO-Safety for FSHOA)
export LIBERO_DIR=\${LIBERO_DIR:-$W/LIBERO-Safety}
export LIBERO_CONFIG_PATH=\$LIBERO_DIR/.libero_config
export PYTHONPATH=$REPO:$W/openvla:\$LIBERO_DIR
cd $REPO
EOF

echo "done. next:  source $W/env.sh && pytest tests/test_env.py -m gpu -v"
echo "then:        uv pip freeze --python $PY > $DATA/requirements.txt"   # uv venvs have no pip
