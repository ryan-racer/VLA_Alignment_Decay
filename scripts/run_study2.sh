#!/usr/bin/env bash
# Study 2 (PREREG_STUDY2.md), unattended and resumable, on the box that holds study 1's checkpoints. Nothing is trained.
#
#   git clone --branch prereg-v2 https://github.com/ryan-racer/VLA_Alignment_Decay.git ~/ftr/study2   # the frozen plan
#   nohup bash ~/ftr/study2/scripts/run_study2.sh > $DATA/study2.out 2>&1 &
#
# Order: gates on dev states (states 0-4) -> stop if either fails (G1's one retry is a new tag, never automatic) ->
# confirmatory rollouts on test states (5-19) -> the study 2 report -> exploratory dose-response checkpoints.
# Same stage/done-file/FAILED conventions as run_all.sh; runs from its own checkout so study 1's stays untouched.
set -o pipefail
source ${W:-$HOME/ftr}/env.sh
S2=$(cd "$(dirname "$0")/.." && pwd)
export PYTHONPATH=$S2:$W/openvla:$LIBERO_DIR   # this checkout's ftr/ and scenes/, the LIBERO-Safety fork
cd $S2
L=$DATA/logs/study2; R=$DATA/runs_study2; CKPTS=$W/ckpt
mkdir -p $L $R
echo "study 2 code $(git -C $S2 describe --tags --always --dirty) | $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
while pgrep -f "[p]ython -m ftr|[r]un_all.sh|[d]ose_response.sh" >/dev/null; do echo "waiting for the GPU (study 1 jobs)... $(date +%H:%M)"; sleep 120; done
FILT='^\[|timing|passed|failed|Traceback|Error|done:'

fail() { echo "== $*" | tee -a $L/FAILED; }
stage() {
    local name=$1 done=$2; shift 2
    if [ -f "$done" ]; then echo "== $name: done"; return 0; fi
    echo "== $name: start $(date +%H:%M)"
    if "$@" && [ -f "$done" ]; then echo "== $name: ok $(date +%H:%M)"; else fail "$name: FAILED $(date +%H:%M)"; return 1; fi
}
need() { for f in "$@"; do [ -f "$f" ] || { fail "skipped (missing $f)"; return 1; }; done; }
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
uid() { sed -n 's/^run_uid=//p' $CKPTS/$1/DONE 2>/dev/null; }
expect() { for n in "$@"; do echo "$n=$(uid $n)"; done; }
OH="--suite object_hazard --tasks all"
DEV="$OH --states 0-4"
TEST="$OH --states 5-19"
# one rollout set per checkpoint and split; two run on the GPU at a time
roll() {  # roll <ckpt> <dev|test> <classes> <templates>
    need $CKPTS/$1/DONE || return 1
    local split=$2; local states=$([ $split = dev ] && echo "$DEV" || echo "$TEST")
    stage ${split}_$1 $R/$1/oh_$split/DONE rollout $R/$1/oh_$split ${split}_$1 --ckpt $CKPTS/$1 $states --classes $3 --templates $4 --video 1
}

# ---- 1. gates (dev states, b1 only) ------------------------------------------------------------------
bg roll C_s0_N200_p0 dev benign b1
bg roll A_s0 dev benign b1
waitbg
GATE_OK=$DATA/figures_study2_gate/passed
if [ ! -f $GATE_OK ]; then
    python -m ftr.analyze --prereg study2 --runs "$R/*/oh_dev" --gate2 C_s0_N200_p0:A_s0 --out $DATA/figures_study2_gate \
        > $L/gate.log 2>&1; rc=$?
    tail -25 $L/gate.log
    if [ $rc -eq 0 ]; then touch $GATE_OK; else fail "study 2 gate failed (rc $rc): see $L/gate.log; stopping"; exit 3; fi
fi

# ---- 2. confirmatory (test states) ------------------------------------------------------------------
bg roll A_s0 test "benign harmful blank" "b1 h6 z0"
bg roll A_s0_N200_p0 test "benign harmful blank" "b1 h6 z0"
waitbg
bg roll C_s0 test "benign harmful" "b1 h6"
bg roll C_s0_N200_p0 test "benign harmful" "b1 h6"
waitbg
for s in 1 2; do
    bg roll A_s$s test benign b1
    bg roll A_s${s}_N200_p0 test benign b1
    waitbg
done

# ---- 3. the study 2 report --------------------------------------------------------------------------
ALL="A_s0 A_s0_N200_p0 C_s0 C_s0_N200_p0 A_s1 A_s1_N200_p0 A_s2 A_s2_N200_p0"
echo "== analyze_study2: start $(date +%H:%M)"
python -m ftr.analyze --prereg study2 --runs "$R/*/oh_test" \
    --pairs A_s0:A_s0_N200_p0 A_s0_N200_p0:C_s0_N200_p0 C_s0:C_s0_N200_p0 A_s1:A_s1_N200_p0 A_s2:A_s2_N200_p0 \
    --did A_s0:A_s0_N200_p0:C_s0:C_s0_N200_p0 --expect-uid $(expect $ALL) --out $DATA/figures_study2 > $L/analyze.log 2>&1 \
    && echo "== analyze_study2: ok $(date +%H:%M)" || fail "analyze_study2: FAILED"
tail -40 $L/analyze.log

# ---- 4. exploratory: dose-response checkpoints (if study 1's dose run made them) ------------------------
for K in 6 12; do
    for ARM in A C; do [ -f $CKPTS/${ARM}_s0_N200_e$K/DONE ] && bg roll ${ARM}_s0_N200_e$K test "benign harmful" "b1 h6"; done
    waitbg
done
if ls $R/*_e*/oh_test/DONE >/dev/null 2>&1; then
    python -m ftr.analyze --prereg study2 --runs "$R/*/oh_test" \
        --pairs A_s0:A_s0_N200_e6 A_s0:A_s0_N200_e12 A_s0_N200_e12:C_s0_N200_e12 \
        --out $DATA/figures_study2_dose > $L/analyze_dose.log 2>&1 || fail "analyze_study2_dose: FAILED"
fi
echo "== STUDY 2 STAGES ATTEMPTED $(date)"
