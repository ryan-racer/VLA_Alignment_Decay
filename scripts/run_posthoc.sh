#!/usr/bin/env bash
# Post hoc evaluation runs (POSTHOC.md, tag posthoc-v1). Evaluation only, two rollouts at a time, resumable.
#
#   nohup bash ~/ftr/repo/scripts/run_posthoc.sh >> $DATA/posthoc.out 2>&1 &
#
# Needs merged checkpoints in $W/ckpt/{A_s0,C_s0,A_s1,A_s2}_N200_p0 (copied from the backup).
set -o pipefail
source ${W:-$HOME/ftr}/env.sh
S=$(cd "$(dirname "$0")/.." && pwd)
export PYTHONPATH=$S:$W/openvla:$LIBERO_DIR
cd $S
L=$DATA/logs/posthoc; R=$DATA/runs_posthoc; CKPTS=$W/ckpt
mkdir -p $L $R
echo "posthoc code $(git -C $S describe --tags --always --dirty) | $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
FILT='^\[|timing|passed|failed|Traceback|Error|done:'

fail() { echo "== $*" | tee -a $L/FAILED; }
stage() {
    local name=$1 done=$2; shift 2
    if [ -f "$done" ]; then echo "== $name: done"; return 0; fi
    echo "== $name: start $(date +%H:%M)"
    if "$@" && [ -f "$done" ]; then echo "== $name: ok $(date +%H:%M)"; else fail "$name: FAILED $(date +%H:%M)"; return 1; fi
}
PIDS=""
bg() { "$@" > /dev/null 2>&1 & PIDS="$PIDS $!"; }
waitbg() { [ -z "$PIDS" ] || wait $PIDS; PIDS=""; }
rollout() {
    local out=$1 log=$2 try; shift 2
    for try in 1 2; do
        python -m ftr.rollout --resume --out $out "$@" 2>&1 | tee -a $L/$log.log | { grep --line-buffered -E "$FILT"; true; } && return 0
        echo "== $log: attempt $try exited non-zero" | tee -a $L/$log.log
    done
    return 1
}
roll() {  # roll <ckpt> <class> <template>
    [ -f $CKPTS/$1/DONE ] || { fail "skipped (missing $CKPTS/$1/DONE)"; return 1; }
    stage $1_$3 $R/$1/oh_test_$3/DONE rollout $R/$1/oh_test_$3 $1_$3 --ckpt $CKPTS/$1 --suite object_hazard --tasks all \
        --states 5-19 --classes $2 --templates $3 --video 1
}

bg roll A_s0_N200_p0 benign b5      # R1
bg roll C_s0_N200_p0 benign b5
waitbg
bg roll A_s1_N200_p0 harmful h6     # R2
bg roll A_s2_N200_p0 harmful h6
waitbg
echo "== POSTHOC STAGES ATTEMPTED $(date)"
