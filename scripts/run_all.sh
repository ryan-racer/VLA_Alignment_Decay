#!/usr/bin/env bash
# The whole experiment, unattended and resumable, on one GPU box (Lambda, RunPod, any Linux + NVIDIA).
#
#   git clone https://github.com/ryan-racer/VLA_Alignment_Decay.git ~/ftr/repo
#   bash ~/ftr/repo/scripts/setup_pod.sh                       # once, ~20 min (downloads ~31 GB)
#   source ~/ftr/env.sh && pytest tests/test_env.py tests/test_fixtures.py tests/test_parity.py -m gpu
#   export GH_TOKEN=...                                        # optional: push logs + figures to GitHub every 10 min
#   nohup bash ~/ftr/repo/scripts/run_all.sh > ~/ftr/run_all.out 2>&1 &
#   tail -f ~/ftr/run_all.out
#
# Every stage is skipped if its done-file exists, so re-running continues where it stopped. Rollouts resume
# per episode and refuse to mix checkpoints. A failed stage, and every stage skipped because an input is missing,
# is listed in $L/FAILED. The code that runs is the checkout at launch: log syncing uses its own clone.
# Gate B (safeguard installed) stops the run before personalization; FTR_IGNORE_GATE=1 continues anyway.
set -o pipefail
source ${W:-$HOME/ftr}/env.sh
L=$DATA/logs/run; R=$DATA/runs; D=$DATA/data; CKPTS=$W/ckpt
SEEDS=${SEEDS:-"0 1 2"}   # alignment seeds: all get offline measures; closed loop on seed 0
mkdir -p $L $R $D $CKPTS
echo "code $(git -C $REPO rev-parse --short HEAD) | disk: $(df -h $W | tail -1 | awk '{print $4" free of "$2}') | $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
FILT='^\[|timing|passed|failed|Traceback|Error|wrote|violation-free|move_rows|arm [AC]:|updates total|done:|pairs \('
while pgrep -f "[p]ython -m ftr" >/dev/null; do echo "waiting for running ftr processes... $(date +%H:%M)"; sleep 60; done

# ---- log + figure sync to GitHub from a separate clone (reset to origin each time: it only ever carries logs) --
sync_logs() {
    [ -n "$GH_TOKEN" ] || return 0
    local S=$W/logsync
    {   [ -d $S/.git ] || git clone -q https://$GH_TOKEN@github.com/ryan-racer/VLA_Alignment_Decay.git $S
        git -C $S fetch -q origin && git -C $S reset -q --hard origin/main
        mkdir -p $S/logs/run && cp $L/*.log $S/logs/run/ && { [ ! -f $L/FAILED ] || cp $L/FAILED $S/logs/run/; }
        for f in figures figures_gate; do [ ! -d $DATA/$f ] || { mkdir -p $S/$f && cp -r $DATA/$f/. $S/$f/; }; done
        git -C $S -c user.email=rq11@rice.edu -c user.name="Ryan Quinlivan" add -A logs figures figures_gate
        git -C $S -c user.email=rq11@rice.edu -c user.name="Ryan Quinlivan" commit -qm "run sync $(date +%H:%M)"
        git -C $S push -q origin HEAD:main; } >/dev/null 2>&1   # a rejected push is simply retried next time
    return 0
}
# the loop owns its sleep: on TERM it kills the sleep and exits (sleep & wait is interruptible), so nothing is orphaned
( trap 'kill $SLEEP_PID 2>/dev/null; exit 0' TERM; while true; do sleep 600 & SLEEP_PID=$!; wait $SLEEP_PID; sync_logs; done ) &
SYNC_PID=$!
trap 'kill $SYNC_PID 2>/dev/null; wait $SYNC_PID 2>/dev/null; sync_logs' EXIT

fail() { echo "== $*" | tee -a $L/FAILED; }
stage() {  # stage <name> <done-file> <cmd...> : run unless done; a nonzero exit OR a missing done-file is a failure
    local name=$1 done=$2; shift 2
    if [ -f "$done" ]; then echo "== $name: done"; return 0; fi
    echo "== $name: start $(date +%H:%M)"
    if "$@" && [ -f "$done" ]; then echo "== $name: ok $(date +%H:%M)"; else fail "$name: FAILED $(date +%H:%M)"; return 1; fi
}
need() { for f in "$@"; do [ -f "$f" ] || { fail "skipped (missing $f)"; return 1; }; done; }
filt() { grep --line-buffered -E "$1"; true; }
PIDS=""  # parallel stages: never a bare `wait`, which would also wait for the sync loop
bg() { "$@" > /dev/null 2>&1 & PIDS="$PIDS $!"; }
waitbg() { [ -z "$PIDS" ] || wait $PIDS; PIDS=""; }
status() { for d in "$@"; do [ -f $R/$d/DONE ] && echo "== $d: ok" || echo "== $d: FAILED (see FAILED)"; done; }

rollout() {  # rollout <out-dir> <log-name> <args...>
    local out=$1 log=$2; shift 2
    python -m ftr.rollout --resume --out $out "$@" 2>&1 | tee -a $L/$log.log | filt "$FILT"
}
train() {  # train <run_id> <vla_path> <parquet> <seed> : if its adapter exists, rebuild the merged checkpoint from it
    # (deterministic, same run_uid) instead of retraining (not bit-identical: flash-attn backward)
    local extra=""; [ -f $R/adapters/$1/provenance.json ] && { echo "   re-merging saved adapter $1"; extra="--merge_only true"; }
    python -m ftr.finetune --vla_path $2 --data_parquet $3 --run_root_dir $CKPTS --run_id $1 --adapter_tmp_dir $R/adapters $extra \
        --epochs 3 --batch_size 8 --grad_accumulation_steps 2 --learning_rate 5e-4 --lora_rank 32 --seed $4 \
        2>&1 | tee -a $L/train_$1.log | filt "rows|Saving|done|Traceback|Error|re-merg"
}
score() { python -m ftr.score --ckpt $2 --pairs $D/pairs_test.parquet --out $R/$1/score_test 2>&1 | tee $L/score_$1.log | tail -6; }
analyze() {  # analyze <out-dir> <log> <args...> : the exit status is analyze's own
    local out=$1 log=$2; shift 2
    python -m ftr.analyze --out $out "$@" > $L/$log.log 2>&1; local rc=$?
    cat $out/refusal_rates.csv $out/violation_rates.csv 2>/dev/null | tee -a $L/$log.log; tail -3 $L/$log.log; return $rc
}
FSHOA="--suite obstacle_avoidance_human"
HAZ="$FSHOA --tasks 3 4 --states 0-24 --classes harmful benign --templates h5 b5 --video 2"

# ================================ A. data ==========================================================
stage render_test $D/states_test.parquet python -m ftr.build_data render $FSHOA --tasks 3 4 --states 0-24 --out $D/states_test.parquet
# scripted movement labels (no model; task 0 needs a higher, no-place path to clear the hand) and P's hazard baseline
# (frames kept every 10 steps: mid-trajectory test frames), all three in parallel
bg stage scripted     $R/scripted/DONE     rollout $R/scripted     a4_scripted     --scripted $FSHOA --tasks 1 2 --states 0-29 --video 2
bg stage scripted_t0b $R/scripted_t0b/DONE rollout $R/scripted_t0b a4b_scripted_t0 --scripted $FSHOA --tasks 0 --states 0-29 --no-place --clearance 0.45 --video 2
bg stage P_hazard     $R/P/hazard/DONE     rollout $R/P/hazard     a5_P_hazard     --ckpt $FTR_P_DIR $FSHOA --tasks 3 4 --states 0-24 \
    --classes harmful benign blank --templates h5 b5 z0 --video 2 --store-every 10
waitbg
status scripted scripted_t0b P/hazard
# offline test set: initial frames + one mid-trajectory frame per state, x the held-out templates + blank
need $D/states_test.parquet $R/P/hazard/DONE && stage pairs $D/pairs_test.parquet python -m ftr.build_data pairs \
    --states $D/states_test.parquet --template-split test --mid-from $R/P/hazard --out $D/pairs_test.parquet
need $D/pairs_test.parquet && stage P_score $R/P/score_test/predictions.parquet score P $FTR_P_DIR
EXPORT="python -m ftr.build_data export-rlds --seed 0"
stage export_rehearsal $D/spatial_rehearsal.parquet $EXPORT --rlds $FTR_RLDS_DIR/libero_spatial_no_noops --episodes 30 --category rehearsal --out $D/spatial_rehearsal.parquet
stage export_N200 $D/object_N200_p0.parquet $EXPORT --rlds $FTR_RLDS_DIR/libero_object_no_noops --per-task 20 --out $D/object_N200_p0.parquet
stage export_N50  $D/object_N50_p0.parquet  $EXPORT --rlds $FTR_RLDS_DIR/libero_object_no_noops --per-task 5 --out $D/object_N50_p0.parquet
mix() { python -m ftr.build_data mix --arm $1 --self $R/scripted $R/scripted_t0b --rehearsal $D/spatial_rehearsal.parquet \
    --out $D/$1.parquet 2>&1 | tee -a $L/a8_mix.log | filt "move_rows|arm|Error|Traceback"; }
if need $D/spatial_rehearsal.parquet $R/scripted/DONE $R/scripted_t0b/DONE; then
    stage mix_A $D/A.parquet mix A; stage mix_C $D/C.parquet mix C
fi
sync_logs

# ================================ B. align + gate (seed 0), then the other seeds =====================
align() {  # align <seed> : train A and C from P, score both offline
    local S=$1
    need $D/A.parquet && stage train_A_s$S $CKPTS/A_s$S/DONE train A_s$S $FTR_P_DIR $D/A.parquet $S
    need $D/C.parquet && stage train_C_s$S $CKPTS/C_s$S/DONE train C_s$S $FTR_P_DIR $D/C.parquet $S
    for CK in A_s$S C_s$S; do need $CKPTS/$CK/DONE $D/pairs_test.parquet && stage score_$CK $R/$CK/score_test/predictions.parquet score $CK $CKPTS/$CK; done
}
align 0
reload() { FTR_RUN=$CKPTS/A_s0 FTR_ADAPTER=$R/adapters/A_s0 FTR_PAIRS=$D/pairs_test.parquet pytest tests/test_reload.py -m gpu -v -s \
    2>&1 | tee $L/b2_reload.log | tail -3; grep -q '2 passed' $L/b2_reload.log && touch $L/b2_reload.ok; }
need $CKPTS/A_s0/DONE $D/pairs_test.parquet && stage reload_A $L/b2_reload.ok reload
# closed loop, seed 0: dev (training scenes, held-out states) then the hazard matrix, two processes at a time
for CK in A_s0 C_s0; do need $CKPTS/$CK/DONE && bg stage dev_$CK $R/$CK/dev/DONE rollout $R/$CK/dev b4_dev_$CK --ckpt $CKPTS/$CK $FSHOA --tasks 0 1 2 --states 30-36 --classes harmful benign --templates h5 b5; done
waitbg
for CK in A_s0 C_s0; do need $CKPTS/$CK/DONE && bg stage hazard_$CK $R/$CK/hazard/DONE rollout $R/$CK/hazard b6_hazard_$CK --ckpt $CKPTS/$CK $HAZ; done
waitbg
status A_s0/dev C_s0/dev A_s0/hazard C_s0/hazard
# Gate B on TEST scenes; the dev states get their own table (training-task scenes, never pooled with test)
need $R/P/score_test/predictions.parquet $R/A_s0/score_test/predictions.parquet $R/C_s0/score_test/predictions.parquet \
     $R/P/hazard/DONE $R/A_s0/hazard/DONE $R/C_s0/hazard/DONE &&
    stage gate $DATA/figures_gate/report.json analyze $DATA/figures_gate b5_gate \
        --runs "$R/P/score_test" "$R/A_s0/score_test" "$R/C_s0/score_test" "$R/P/hazard" "$R/A_s0/hazard" "$R/C_s0/hazard" \
        --p P --pairs P:A_s0 A_s0:C_s0 --gate A_s0:C_s0 --targets $D/A.parquet $D/C.parquet
need $R/A_s0/dev/DONE $R/C_s0/dev/DONE &&
    stage gate_dev $DATA/figures_gate/dev/report.json analyze $DATA/figures_gate/dev b5_gate_dev --runs "$R/A_s0/dev" "$R/C_s0/dev" --pairs A_s0:C_s0
sync_logs
if ! python -c "import json,sys; sys.exit(0 if json.load(open('$DATA/figures_gate/report.json'))['gate']['passed'] else 3)" 2>/dev/null; then
    fail "Gate B not passed (or not evaluated): see $L/b5_gate.log"
    [ "$FTR_IGNORE_GATE" = 1 ] || { echo "== stopping before the other seeds and personalization (FTR_IGNORE_GATE=1 continues)"; exit 3; }
fi
for S in $SEEDS; do [ "$S" = 0 ] || align $S; done

# ================================ C. personalize + measure =========================================
for S in $SEEDS; do
    need $CKPTS/A_s$S/DONE $D/object_N200_p0.parquet && stage personalize_A${S}_200 $CKPTS/A_s${S}_N200_p0/DONE train A_s${S}_N200_p0 $CKPTS/A_s$S $D/object_N200_p0.parquet $S
    need $CKPTS/C_s$S/DONE $D/object_N200_p0.parquet && stage personalize_C${S}_200 $CKPTS/C_s${S}_N200_p0/DONE train C_s${S}_N200_p0 $CKPTS/C_s$S $D/object_N200_p0.parquet $S
done
need $CKPTS/A_s0/DONE $D/object_N50_p0.parquet && stage personalize_A0_50 $CKPTS/A_s0_N50_p0/DONE train A_s0_N50_p0 $CKPTS/A_s0 $D/object_N50_p0.parquet 0
sync_logs
for CK in A_s0_N50_p0 $(for S in $SEEDS; do echo A_s${S}_N200_p0 C_s${S}_N200_p0; done); do
    need $CKPTS/$CK/DONE && stage score_$CK $R/$CK/score_test/predictions.parquet score $CK $CKPTS/$CK
done
for CK in A_s0_N200_p0 C_s0_N200_p0; do need $CKPTS/$CK/DONE && bg stage hazard_$CK $R/$CK/hazard/DONE rollout $R/$CK/hazard c3_hazard_$CK --ckpt $CKPTS/$CK $HAZ; done
waitbg
status A_s0_N200_p0/hazard C_s0_N200_p0/hazard
sync_logs
# utility on the upstream LIBERO checkout: spatial + object in parallel, per checkpoint
export LIBERO_DIR=$W/LIBERO LIBERO_CONFIG_PATH=$W/LIBERO/.libero_config PYTHONPATH=$REPO:$W/openvla:$W/LIBERO
for CK in P A_s0 A_s0_N200_p0; do
    CKPT=$([ "$CK" = P ] && echo $FTR_P_DIR || echo $CKPTS/$CK)
    [ "$CK" = P ] || need $CKPTS/$CK/DONE || continue
    for SU in libero_spatial libero_object; do
        bg stage util_${CK}_$SU $R/$CK/u_$SU/DONE rollout $R/$CK/u_$SU c4_util_${CK}_$SU --ckpt $CKPT --suite $SU --tasks all --states 0-4 --task-instruction
    done
    waitbg
    status $CK/u_libero_spatial $CK/u_libero_object
done
export LIBERO_DIR=$W/LIBERO-Safety LIBERO_CONFIG_PATH=$W/LIBERO-Safety/.libero_config PYTHONPATH=$REPO:$W/openvla:$W/LIBERO-Safety
# paired / control-adjusted tests only for seeds whose four offline scores all exist (one failed seed must not block the rest)
sc() { [ -f $R/$1/score_test/predictions.parquet ]; }
PAIRS="A_s0_N200_p0:C_s0_N200_p0"; DID=""
sc A_s0_N50_p0 && PAIRS="$PAIRS A_s0:A_s0_N50_p0"
for S in $SEEDS; do
    if sc A_s$S && sc A_s${S}_N200_p0 && sc C_s$S && sc C_s${S}_N200_p0; then
        PAIRS="$PAIRS A_s$S:A_s${S}_N200_p0 C_s$S:C_s${S}_N200_p0"; DID="$DID A_s$S:A_s${S}_N200_p0:C_s$S:C_s${S}_N200_p0"
    else fail "seed $S: offline scores incomplete, left out of the paired tests"; fi
done
need $R/A_s0_N200_p0/hazard/DONE $R/C_s0_N200_p0/hazard/DONE &&
    stage analyze $DATA/figures/report.json analyze $DATA/figures c5_analyze \
        --runs "$R/*/score_test" "$R/*/hazard" "$R/*/u_*" --p P --pairs $PAIRS --did $DID \
        --targets $D/A.parquet $D/C.parquet $D/object_N50_p0.parquet $D/object_N200_p0.parquet
echo "== ALL STAGES ATTEMPTED $(date)"; [ ! -f $L/FAILED ] || { echo "== failures:"; cat $L/FAILED; }
