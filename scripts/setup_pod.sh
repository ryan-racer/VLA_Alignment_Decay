#!/usr/bin/env bash
# Idempotent GPU environment for OpenVLA + LIBERO + LIBERO-Safety on any Linux + NVIDIA box (Lambda, RunPod, Colab).
# Runs as a normal user with sudo; everything lives under $HOME/ftr unless overridden.
#
#   W     venv, openvla, LIBERO clones, merged checkpoints                      (default $HOME/ftr)
#   DATA  runs/, data/, logs/, adapters, figures; point at a persistent filesystem to survive instance teardown
#         (Lambda: DATA=/lambda/nfs/<filesystem>/ftr)                            (default $W)
#   HF    where the ~31 GB of public downloads go; local is fine, HF re-serves them in minutes  (default $W/hf)
#   CUDA  cu118 (default; prebuilt flash-attn wheel, no compile) or cu121 (compiles flash-attn, ~30-60 min)
#
#   git clone https://github.com/ryan-racer/VLA_Alignment_Decay.git ~/ftr/repo && bash ~/ftr/repo/scripts/setup_pod.sh
#   then:    source ~/ftr/env.sh && pytest tests/test_env.py tests/test_fixtures.py tests/test_parity.py -m gpu -v
#   then:    scripts/run_all.sh (see its header)
set -euo pipefail

W=${W:-$HOME/ftr}
DATA=${DATA:-$W}
CUDA=${CUDA:-cu118}
REPO=${REPO:-$W/repo}
HF=${HF:-$W/hf}
OPENVLA_SHA=c8f03f4
# Hugging Face revisions pinned (the repos can change under the same name)
P_REV=962318cec55ac10993ff0f5f43eda9a270b4c873        # openvla/openvla-7b-finetuned-libero-spatial
RLDS_REV=6ce6aaaaabdbe590b1eef5cd29c0d33f14a08551     # openvla/modified_libero_rlds
ASSETS_REV=94ccca773d1bab760d1aecec4f62c8d4b8797d40   # LIBERO-Safety/libero_safety_assets
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
export PATH=$HOME/.local/bin:$PATH
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
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
# Issue #3: check_robot_contact never fires as shipped. Apply once; then REQUIRE that it is applied (a patch that
# silently fails to apply would leave the primary violation signal dead).
PATCH=$REPO/patches/libero_safety_issue3.patch
git -C $W/LIBERO-Safety apply --reverse --check $PATCH 2>/dev/null || git -C $W/LIBERO-Safety apply $PATCH
git -C $W/LIBERO-Safety apply --reverse --check $PATCH || { echo "Issue #3 patch is not applied to LIBERO-Safety"; exit 1; }
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
[ -f $HF/P/config.json ] || $HFDL openvla/openvla-7b-finetuned-libero-spatial --revision $P_REV --local-dir $HF/P
# one call per suite: repeated --include flags keep only the last one; guard on a file, not a directory
for SUITE in libero_spatial_no_noops libero_object_no_noops; do
    [ -f $HF/rlds/$SUITE/1.0.0/dataset_info.json ] || \
        $HFDL openvla/modified_libero_rlds --repo-type dataset --revision $RLDS_REV --include "$SUITE/*" --local-dir $HF/rlds
done
[ -d $HF/safety_assets ] || $HFDL LIBERO-Safety/libero_safety_assets --repo-type dataset --revision $ASSETS_REV --local-dir $HF/safety_assets
# assets.zip has a top-level `assets/` folder (902,920 entries): unzip into the fork's libero/libero/ so it lands
# at libero/libero/assets/. Redo each session if W is ephemeral. Detect by a known file, not a marker.
ASSETS=$W/LIBERO-Safety/libero/libero/assets
if [ -d $ASSETS/assets/scenes ]; then   # repair a nested layout from an earlier unzip into assets/
    mv -n $ASSETS/assets/* $ASSETS/ && rm -rf $ASSETS/assets
fi
if [ ! -f $ASSETS/scenes/libero_tabletop_base_style.xml ]; then
    unzip -q -o $HF/safety_assets/assets.zip -d $W/LIBERO-Safety/libero/libero/
fi
[ -f $ASSETS/scenes/libero_tabletop_base_style.xml ] || { echo "assets not in place: $ASSETS"; exit 1; }

# ---- env file to source per shell -------------------------------------------------------------------
cat > $W/env.sh <<EOF
export W=$W DATA=$DATA REPO=$REPO HF=$HF PATH=\$HOME/.local/bin:\$PATH
source $W/venv/bin/activate
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl NVIDIA_DRIVER_CAPABILITIES=all
export HF_HOME=$HF FTR_P_DIR=$HF/P FTR_RLDS_DIR=$HF/rlds FTR_DATA=$DATA WANDB_MODE=offline
export WANDB_DIR=$DATA/logs TOKENIZERS_PARALLELISM=false
export MPLBACKEND=Agg   # headless; the fork imports matplotlib at import time
export PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1   # progress lines reach logs immediately; segfaults print a traceback
# two rollout processes share the GPU: keep TF from grabbing all VRAM and cap the thread pools so they don't fight
export TF_FORCE_GPU_ALLOW_GROWTH=true OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 TF_NUM_INTRAOP_THREADS=2 TF_NUM_INTEROP_THREADS=1
# pick ONE libero: LIBERO_DIR=$W/LIBERO for libero_spatial/libero_object (default: LIBERO-Safety for FSHOA)
export LIBERO_DIR=\${LIBERO_DIR:-$W/LIBERO-Safety}
export LIBERO_CONFIG_PATH=\$LIBERO_DIR/.libero_config
export PYTHONPATH=$REPO:$W/openvla:\$LIBERO_DIR
cd $REPO
EOF

echo "done. next:  source $W/env.sh && pytest tests/test_env.py -m gpu -v"
echo "then:        uv pip freeze --python $PY > $DATA/requirements.txt"   # uv venvs have no pip
