#!/usr/bin/env bash
# Study 2b (PREREG_STUDY2B.md), unattended and resumable, on the box that holds study 1's checkpoints. Nothing is trained.
#
#   git clone --branch prereg-v3 <bundle> ~/ftr/study2b   # the frozen plan
#   nohup bash ~/ftr/study2b/scripts/run_study2b.sh > $DATA/study2b.out 2>&1 &
#
# Order: per candidate placement (p1, then p2) on dev states (0-4): G1 + G3 -> G2 only if both pass -> the first
# candidate passing all three is selected -> confirmatory b1 rollouts on its test states (5-19) -> the study 2b report.
# Same stage/done-file/FAILED conventions as run_study2.sh.
set -o pipefail
source ${W:-$HOME/ftr}/env.sh
S2=$(cd "$(dirname "$0")/.." && pwd)
export PYTHONPATH=$S2:$W/openvla:$LIBERO_DIR   # this checkout's ftr/ and scenes/, the LIBERO-Safety fork
cd $S2
L=$DATA/logs/study2b; R=$DATA/runs_study2b; CKPTS=$W/ckpt; G=$DATA/figures_study2b_gate
mkdir -p $L $R $G
echo "study 2b code $(git -C $S2 describe --tags --always --dirty) | $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
while pgrep -f "[p]ython -m ftr|[r]un_all.sh|[d]ose_response.sh|[r]un_study2.sh" >/dev/null; do echo "waiting for the GPU... $(date +%H:%M)"; sleep 120; done
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
roll() {  # roll <ckpt> <suite> <dev|test>: b1 only
    need $CKPTS/$1/DONE || return 1
    local states=$([ $3 = dev ] && echo "0-4" || echo "5-19")
    stage ${3}_$2_$1 $R/$1/$2_$3/DONE rollout $R/$1/$2_$3 ${3}_$2_$1 --ckpt $CKPTS/$1 --suite $2 --tasks all \
        --states $states --classes benign --templates b1 --video 1
}
gate() {  # gate <suite>: analyze --gate2b on that suite's dev runs; prints the gate; exit status 3 if any gate fails
    python -m ftr.analyze --prereg study2b --runs "$R/*/$1_dev" --gate2b C_s0_N200_p0:A_s0:A_s0_N200_p0 --out $G/$1 \
        > $L/gate_$1.log 2>&1; local rc=$?
    if [ $rc -ne 0 ] && [ $rc -ne 3 ]; then tail -20 $L/gate_$1.log; fail "gate $1: analyze crashed (rc $rc); stopping"; exit 4; fi
    python -c "import json; g = json.load(open('$G/$1/report.json'))['gate']; print('gate $1:', json.dumps(g))"
    return $rc
}
passed() { python -c "import json, sys; g = json.load(open('$G/$1/report.json'))['gate']; sys.exit(0 if all(g[k]['passed'] for k in sys.argv[1:]) else 1)" "${@:2}"; }

# ---- 1. placement gates (dev states, b1) -------------------------------------------------------------
SEL=$(cat $G/selected 2>/dev/null)
if [ -z "$SEL" ]; then
    for P in object_hazard_p1 object_hazard_p2; do
        bg roll C_s0_N200_p0 $P dev
        bg roll A_s0_N200_p0 $P dev
        waitbg
        gate $P
        if ! passed $P g1 g3; then echo "== $P: G1/G3 failed, next candidate"; continue; fi
        roll A_s0 $P dev
        if gate $P; then SEL=$P; echo $P > $G/selected; break; fi
        echo "== $P: G2 failed, next candidate"
    done
fi
if [ -z "$SEL" ]; then fail "study 2b gate failed: no placement passed; stopping"; echo "== STUDY 2B STAGES ATTEMPTED $(date)"; exit 3; fi
echo "== selected placement: $SEL"

# ---- 2. confirmatory (test states, b1), three rollouts at a time -------------------------------------
for WAVE in "A_s0 A_s0_N200_p0 C_s0" "C_s0_N200_p0 A_s1 A_s1_N200_p0" "A_s2 A_s2_N200_p0"; do
    for CK in $WAVE; do bg roll $CK $SEL test; done
    waitbg
done

# ---- 3. the study 2b report --------------------------------------------------------------------------
ALL="A_s0 A_s0_N200_p0 C_s0 C_s0_N200_p0 A_s1 A_s1_N200_p0 A_s2 A_s2_N200_p0"
echo "== analyze_study2b: start $(date +%H:%M)"
python -m ftr.analyze --prereg study2b --runs "$R/*/${SEL}_test" \
    --pairs A_s0:A_s0_N200_p0 A_s0_N200_p0:C_s0_N200_p0 C_s0:C_s0_N200_p0 A_s1:A_s1_N200_p0 A_s2:A_s2_N200_p0 \
    --did A_s0:A_s0_N200_p0:C_s0:C_s0_N200_p0 --expect-uid $(expect $ALL) --out $DATA/figures_study2b > $L/analyze.log 2>&1 \
    && echo "== analyze_study2b: ok $(date +%H:%M)" || fail "analyze_study2b: FAILED"
tail -40 $L/analyze.log
echo "== STUDY 2B STAGES ATTEMPTED $(date)"
