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
# per episode. A failed stage, and every stage skipped because an input is missing, is listed in $L/FAILED.
# The code that runs is the checkout at launch: log syncing uses its own clone and never touches $REPO.
set -o pipefail
source ${W:-$HOME/ftr}/env.sh
L=$DATA/logs/run; R=$DATA/runs; D=$DATA/data; CKPTS=$W/ckpt
mkdir -p $L $R $D $CKPTS
echo "code $(git -C $REPO rev-parse --short HEAD) | disk: $(df -h $W | tail -1 | awk '{print $4" free of "$2}') | $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
FILT='^\[|timing|passed|failed|Traceback|Error|wrote|violation-free|move_rows|arm [AC]:|updates total|done:'
while pgrep -f "[p]ython -m ftr" >/dev/null; do echo "waiting for running ftr processes... $(date +%H:%M)"; sleep 60; done

# ---- log + figure sync to GitHub from a separate clone (reset to origin each time: it only ever carries logs) --
sync_logs() {
    [ -n "$GH_TOKEN" ] || return 0
    local S=$W/logsync
    {   [ -d $S/.git ] || git clone -q https://$GH_TOKEN@github.com/ryan-racer/VLA_Alignment_Decay.git $S
        git -C $S fetch -q origin && git -C $S reset -q --hard origin/main
        mkdir -p $S/logs/run && cp $L/*.log $S/logs/run/ && { [ ! -f $L/FAILED ] || cp $L/FAILED $S/logs/run/; }
        [ ! -d $DATA/figures ] || { mkdir -p $S/figures && cp -r $DATA/figures/. $S/figures/; }
        git -C $S -c user.email=rq11@rice.edu -c user.name="Ryan Quinlivan" add -A logs figures
        git -C $S -c user.email=rq11@rice.edu -c user.name="Ryan Quinlivan" commit -qm "run sync $(date +%H:%M)"
        git -C $S push -q origin HEAD:main; } >/dev/null 2>&1   # a rejected push is simply retried next time
    return 0
}
( while true; do sleep 600; sync_logs; done ) &
SYNC_PID=$!
trap 'pkill -P $SYNC_PID 2>/dev/null; kill $SYNC_PID 2>/dev/null; sync_logs' EXIT  # -P: also the loop's sleep

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

rollout() {  # rollout <out-dir> <log-name> <args...>
    local out=$1 log=$2; shift 2
    python -m ftr.rollout --resume --out $out "$@" 2>&1 | tee -a $L/$log.log | filt "$FILT"
}
train() {  # train <run_id> <vla_path> <parquet> : if its adapter exists, rebuild the merged checkpoint from it
    # (deterministic) instead of retraining (not bit-identical: flash-attn backward)
    local extra=""; [ -f $R/adapters/$1/adapter_config.json ] && { echo "   re-merging saved adapter $1"; extra="--merge_only true"; }
    python -m ftr.finetune --vla_path $2 --data_parquet $3 --run_root_dir $CKPTS --run_id $1 --adapter_tmp_dir $R/adapters $extra \
        --epochs 3 --batch_size 8 --grad_accumulation_steps 2 --learning_rate 5e-4 --lora_rank 32 --seed 0 \
        2>&1 | tee -a $L/train_$1.log | filt "rows|Saving|done|Traceback|Error|re-merg"
}
score() { python -m ftr.score --ckpt $2 --pairs $D/pairs_test.parquet --out $R/$1/score_test 2>&1 | tee $L/score_$1.log | tail -6; }
HAZ="--suite obstacle_avoidance_human --tasks 3 4 --states 0-24 --classes harmful benign --templates h5 b5 --video 2"
FSHOA="--suite obstacle_avoidance_human"

# ================================ A. data ==========================================================
stage render_train $D/states_train.parquet python -m ftr.build_data render $FSHOA --tasks 0 1 2 --states 0-29 --out $D/states_train.parquet
stage render_test  $D/states_test.parquet  python -m ftr.build_data render $FSHOA --tasks 3 4 --states 0-24 --out $D/states_test.parquet
need $D/states_test.parquet && stage pairs $D/pairs_test.parquet \
    python -m ftr.build_data pairs --states $D/states_test.parquet --template-split test --out $D/pairs_test.parquet
# scripted movement labels (no model; task 0 needs a higher, no-place path to clear the hand) and P's hazard baseline,
# all three in parallel; then the RLDS exports (CPU/TF) while P is scored
bg stage scripted     $R/scripted/DONE     rollout $R/scripted     a4_scripted    --scripted $FSHOA --tasks 1 2 --states 0-29 --video 2
bg stage scripted_t0b $R/scripted_t0b/DONE rollout $R/scripted_t0b a4b_scripted_t0 --scripted $FSHOA --tasks 0 --states 0-29 --no-place --clearance 0.45 --video 2
bg stage P_hazard     $R/P/hazard/DONE     rollout $R/P/hazard     a5_P_hazard    --ckpt $FTR_P_DIR $FSHOA --tasks 3 4 --states 0-24 --classes harmful benign blank --templates h5 b5 z0 --video 2
waitbg
for s in scripted scripted_t0b P/hazard; do [ -f $R/$s/DONE ] && echo "== $s: ok" || echo "== $s: FAILED"; done
need $D/pairs_test.parquet && stage P_score $R/P/score_test/predictions.parquet score P $FTR_P_DIR
EXPORT="python -m ftr.build_data export-rlds --seed 0"
stage export_rehearsal $D/spatial_rehearsal.parquet $EXPORT --rlds $FTR_RLDS_DIR/libero_spatial_no_noops --episodes 30 --category rehearsal --out $D/spatial_rehearsal.parquet
stage export_N200 $D/object_N200_p0.parquet $EXPORT --rlds $FTR_RLDS_DIR/libero_object_no_noops --per-task 20 --out $D/object_N200_p0.parquet
stage export_N50  $D/object_N50_p0.parquet  $EXPORT --rlds $FTR_RLDS_DIR/libero_object_no_noops --per-task 5 --out $D/object_N50_p0.parquet
mix() { python -m ftr.build_data mix --arm $1 --states $D/states_train.parquet --self $R/scripted $R/scripted_t0b \
    --rehearsal $D/spatial_rehearsal.parquet --out $D/$1.parquet 2>&1 | tee -a $L/a8_mix.log | filt "move_rows|arm|Error|Traceback"; }
if need $D/states_train.parquet $D/spatial_rehearsal.parquet $R/scripted/DONE $R/scripted_t0b/DONE; then
    stage mix_A $D/A.parquet mix A; stage mix_C $D/C.parquet mix C
fi
sync_logs

# ================================ B. align + gate ===================================================
need $D/A.parquet && stage train_A $CKPTS/A_s0/DONE train A_s0 $FTR_P_DIR $D/A.parquet
need $D/C.parquet && stage train_C $CKPTS/C_s0/DONE train C_s0 $FTR_P_DIR $D/C.parquet
reload() { FTR_RUN=$CKPTS/A_s0 FTR_ADAPTER=$R/adapters/A_s0 FTR_PAIRS=$D/pairs_test.parquet pytest tests/test_reload.py -m gpu -v \
    2>&1 | tee $L/b2_reload.log | tail -3; grep -q '2 passed' $L/b2_reload.log && touch $L/b2_reload.ok; }
need $CKPTS/A_s0/DONE && stage reload_A $L/b2_reload.ok reload
for CK in A_s0 C_s0; do need $CKPTS/$CK/DONE && stage score_$CK $R/$CK/score_test/predictions.parquet score $CK $CKPTS/$CK; done
# dev (training-task scenes, held-out states) and the baseline hazard matrix: four rollouts, two at a time
for CK in A_s0 C_s0; do need $CKPTS/$CK/DONE && bg stage dev_$CK $R/$CK/dev/DONE rollout $R/$CK/dev b4_dev_$CK --ckpt $CKPTS/$CK $FSHOA --tasks 0 1 2 --states 30-36 --classes harmful benign --templates h5 b5; done
waitbg
for CK in A_s0 C_s0; do need $CKPTS/$CK/DONE && bg stage hazard_$CK $R/$CK/hazard/DONE rollout $R/$CK/hazard b6_hazard_$CK --ckpt $CKPTS/$CK $HAZ; done
waitbg
for CK in A_s0 C_s0; do for s in dev hazard; do [ -f $R/$CK/$s/DONE ] && echo "== ${s}_$CK: ok" || echo "== ${s}_$CK: FAILED"; done; done
analyze() {  # analyze <out-dir> <log> <args...> : the exit status is analyze's own
    local out=$1 log=$2; shift 2
    python -m ftr.analyze --out $out "$@" > $L/$log.log 2>&1; local rc=$?
    cat $out/refusal_rates.csv $out/contact_rates.csv 2>/dev/null | tee -a $L/$log.log; tail -3 $L/$log.log; return $rc
}
# gate on TEST scenes only; the dev states get their own table (training-task scenes, never pooled with test)
need $R/P/score_test/predictions.parquet $R/A_s0/hazard/DONE $R/C_s0/hazard/DONE &&
    stage gate $DATA/figures_gate/report.json analyze $DATA/figures_gate b5_gate \
    --runs $R/P/score_test $R/A_s0/score_test $R/C_s0/score_test $R/P/hazard $R/A_s0/hazard $R/C_s0/hazard \
    --p P --pairs A_s0:C_s0 --targets $D/A.parquet $D/C.parquet
need $R/A_s0/dev/DONE $R/C_s0/dev/DONE &&
    stage gate_dev $DATA/figures_gate/dev/report.json analyze $DATA/figures_gate/dev b5_gate_dev \
    --runs $R/A_s0/dev $R/C_s0/dev --pairs A_s0:C_s0
sync_logs

# ================================ C. personalize + measure =========================================
need $CKPTS/A_s0/DONE $D/object_N200_p0.parquet && stage personalize_A200 $CKPTS/A_s0_N200_p0/DONE train A_s0_N200_p0 $CKPTS/A_s0 $D/object_N200_p0.parquet
need $CKPTS/C_s0/DONE $D/object_N200_p0.parquet && stage personalize_C200 $CKPTS/C_s0_N200_p0/DONE train C_s0_N200_p0 $CKPTS/C_s0 $D/object_N200_p0.parquet
need $CKPTS/A_s0/DONE $D/object_N50_p0.parquet  && stage personalize_A50  $CKPTS/A_s0_N50_p0/DONE  train A_s0_N50_p0  $CKPTS/A_s0 $D/object_N50_p0.parquet
sync_logs
for CK in A_s0_N50_p0 A_s0_N200_p0 C_s0_N200_p0; do need $CKPTS/$CK/DONE && stage score_$CK $R/$CK/score_test/predictions.parquet score $CK $CKPTS/$CK; done
for CK in A_s0_N200_p0 C_s0_N200_p0; do need $CKPTS/$CK/DONE && bg stage hazard_$CK $R/$CK/hazard/DONE rollout $R/$CK/hazard c3_hazard_$CK --ckpt $CKPTS/$CK $HAZ; done
waitbg
for CK in A_s0_N200_p0 C_s0_N200_p0; do [ -f $R/$CK/hazard/DONE ] && echo "== hazard_$CK: ok" || echo "== hazard_$CK: FAILED"; done
sync_logs
# utility on the upstream LIBERO checkout: spatial + object in parallel, per checkpoint
export LIBERO_DIR=$W/LIBERO LIBERO_CONFIG_PATH=$W/LIBERO/.libero_config PYTHONPATH=$REPO:$W/openvla:$W/LIBERO
for CK in P A_s0 A_s0_N200_p0; do
    CKPT=$([ "$CK" = P ] && echo $FTR_P_DIR || echo $CKPTS/$CK)
    [ "$CK" = P ] || need $CKPTS/$CK/DONE || continue
    for S in libero_spatial libero_object; do
        bg stage util_${CK}_$S $R/$CK/u_$S/DONE rollout $R/$CK/u_$S c4_util_${CK}_$S --ckpt $CKPT --suite $S --tasks all --states 0-4 --task-instruction
    done
    waitbg
    for S in libero_spatial libero_object; do [ -f $R/$CK/u_$S/DONE ] && echo "== util_${CK}_$S: ok" || echo "== util_${CK}_$S: FAILED"; done
done
export LIBERO_DIR=$W/LIBERO-Safety LIBERO_CONFIG_PATH=$W/LIBERO-Safety/.libero_config PYTHONPATH=$REPO:$W/openvla:$W/LIBERO-Safety
need $R/A_s0_N200_p0/hazard/DONE $R/C_s0_N200_p0/hazard/DONE &&
    stage analyze $DATA/figures/report.json analyze $DATA/figures c5_analyze \
    --runs $R/*/score_test $R/*/hazard $R/*/u_* \
    --p P --pairs A_s0:A_s0_N200_p0 C_s0:C_s0_N200_p0 A_s0_N200_p0:C_s0_N200_p0 \
    --targets $D/A.parquet $D/C.parquet $D/object_N50_p0.parquet $D/object_N200_p0.parquet
echo "== ALL STAGES ATTEMPTED $(date)"; [ ! -f $L/FAILED ] || { echo "== failures:"; cat $L/FAILED; }
