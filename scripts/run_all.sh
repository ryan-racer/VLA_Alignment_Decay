#!/usr/bin/env bash
# The whole remaining pipeline, unattended and resumable. Every stage is skipped if its DONE marker exists,
# rollouts resume mid-run, a failed stage is recorded in $L/FAILED and dependent stages are skipped.
# Logs (and figures at the end) are pushed to GitHub every 10 minutes by a background sync.
#
#   From the Colab terminal (survives cell interrupts; needs the notebook's env vars + GH_TOKEN exported):
#     export W=/content/ftr DATA=/content/drive/MyDrive/ftr REPO=/content/ftr/repo HF=/content/ftr/hf GH_TOKEN=...
#     git -C $REPO fetch -q && git -C $REPO reset -q --hard origin/main && bash $REPO/scripts/setup_pod.sh 2>&1 | tail -1
#     nohup bash -c 'source $W/env.sh && bash $REPO/scripts/run_all.sh' > $DATA/logs/phase2/run_all.out 2>&1 &
#     tail -f $DATA/logs/phase2/run_all.out
#   Or as a notebook cell:  %%shell ... source $W/env.sh && bash $REPO/scripts/run_all.sh
set -o pipefail
source $W/env.sh
L=$DATA/logs/phase2; mkdir -p $L; R=$DATA/runs; D=$DATA/data
# Checkpoints go to the LOCAL disk: a merged 7B checkpoint is ~15 GB and Drive is 10 GB. Results (parquet, videos,
# DONE markers) and the LoRA adapters stay on Drive. If the runtime dies, merged checkpoints are rebuilt from the
# saved adapters (deterministic, ~2 min each); only a training that was in flight is redone.
CKPTS=$W/ckpt; mkdir -p $CKPTS; echo "local disk: $(df -h $W | tail -1 | awk '{print $4" free of "$2}')"
FILT='^\[|timing|passed|failed|Traceback|Error|wrote|violation-free|move_rows|arm [AC]:|updates total|done:|exit [0-9]'
# wait for any rollout/training already running (e.g. the notebook's P baseline) rather than killing it
while pgrep -f "[p]ython -m ftr" >/dev/null; do echo "waiting for running ftr processes... $(date +%H:%M)"; sleep 60; done

# ---- background log sync to GitHub (needs GH_TOKEN in the environment) ---------------------------
sync_logs() {
    cd $REPO && mkdir -p logs/phase2 && cp $L/*.log logs/phase2/ 2>/dev/null; [ -f $L/FAILED ] && cp $L/FAILED logs/phase2/
    [ -d $DATA/figures ] && { mkdir -p figures && cp -r $DATA/figures/* figures/ 2>/dev/null; git add figures >/dev/null 2>&1; }
    git add logs/phase2 >/dev/null 2>&1 && git commit -qm "phase2 sync $(date +%H:%M)" >/dev/null 2>&1 \
        && git fetch -q && git rebase -q origin/main && git push -q https://$GH_TOKEN@github.com/ryan-racer/VLA_Alignment_Decay.git HEAD:main
    cd $REPO
}
git -C $REPO config user.email "rq11@rice.edu"; git -C $REPO config user.name "Ryan Quinlivan"
( while true; do sleep 600; sync_logs; done ) &
SYNC_PID=$!
trap 'kill $SYNC_PID 2>/dev/null; sync_logs' EXIT

stage() {  # stage <name> <done-file> <cmd...>  : run unless done; record failure
    local name=$1 done=$2; shift 2
    if [ -f "$done" ]; then echo "== $name: done"; return 0; fi
    echo "== $name: start $(date +%H:%M)"
    if "$@"; then echo "== $name: ok $(date +%H:%M)"; else echo "== $name: FAILED $(date +%H:%M)" | tee -a $L/FAILED; return 1; fi
}
need() { for f in "$@"; do [ -f "$f" ] || { echo "   skip: missing $f"; return 1; }; done; }

# parallel stages: `wait` with no pids would also wait for the sync loop above, forever
PIDS=""
bg() { "$@" > /dev/null 2>&1 & PIDS="$PIDS $!"; }
waitbg() { [ -n "$PIDS" ] && wait $PIDS; PIDS=""; }
rollout() {  # rollout <out-dir> <log-name> <args...>
    local out=$1 log=$2; shift 2
    python -m ftr.rollout --resume --out $out "$@" 2>&1 | tee $L/$log.log | grep --line-buffered -E "$FILT"
}
train() {  # train <run_id> <vla_path> <parquet> : adapters go to Drive ($R/adapters, a few hundred MB each); if one
    # already exists the merged checkpoint is rebuilt from it (deterministic) instead of retraining (not deterministic)
    local extra=""; [ -f $R/adapters/$1/adapter_config.json ] && { echo "   re-merging saved adapter $1"; extra="--merge_only true"; }
    python -m ftr.finetune --vla_path $2 --data_parquet $3 --run_root_dir $CKPTS --run_id $1 --adapter_tmp_dir $R/adapters $extra \
        --epochs 3 --batch_size 8 --grad_accumulation_steps 2 --learning_rate 5e-4 --lora_rank 32 --seed 0 \
        2>&1 | tee -a $L/train_$1.log | grep --line-buffered -E "rows|updates|Saving|done|Traceback|Error|trainable|re-merg"
}
rescue_adapters() {  # adapters trained before adapters lived on Drive: copy any local one to Drive (atomic)
    for d in $CKPTS/adapters/*/; do [ -d "$d" ] || continue; x=$(basename $d)
        [ -f $R/adapters/$x/adapter_config.json ] && continue
        mkdir -p $R/adapters && cp -r $d $R/adapters/.$x.tmp && cp $CKPTS/$x/dataset_statistics.json $R/adapters/.$x.tmp/ 2>/dev/null \
            && mv $R/adapters/.$x.tmp $R/adapters/$x && echo "   saved adapter $x to Drive" || rm -rf $R/adapters/.$x.tmp
    done
}
score() { python -m ftr.score --ckpt $CKPTS/$1 --pairs $D/pairs_test.parquet --out $R/$1/score_test 2>&1 | tee $L/score_$1.log | tail -6; }
HAZ="--suite obstacle_avoidance_human --tasks 3 4 --states 0-24 --classes harmful benign --templates h5 b5 --video 2"

# ================================ A. data ==========================================================
stage "scripted_t0b" $R/scripted_t0b/DONE rollout $R/scripted_t0b a4b_scripted_t0 --scripted --suite obstacle_avoidance_human --tasks 0 --states 0-29 --no-place --clearance 0.45 --video 2
stage "P_hazard" $R/P_hazard/DONE rollout $R/P_hazard a5_P_hazard --ckpt $FTR_P_DIR --suite obstacle_avoidance_human --tasks 3 4 --states 0-24 --classes harmful benign blank --templates h5 b5 z0 --video 2
score_P() { python -m ftr.score --ckpt $FTR_P_DIR --pairs $D/pairs_test.parquet --out $R/P_score 2>&1 | tee $L/a6_P_score.log | tail -6; }
[ -f $R/P_score/predictions.parquet ] || stage "P_score" $R/P_score/predictions.parquet score_P
stage "export_rehearsal" $D/spatial_rehearsal.parquet python -m ftr.build_data export-rlds --rlds $FTR_RLDS_DIR/libero_spatial_no_noops --episodes 30 --seed 0 --category rehearsal --out $D/spatial_rehearsal.parquet
stage "export_N200" $D/object_N200_p0.parquet python -m ftr.build_data export-rlds --rlds $FTR_RLDS_DIR/libero_object_no_noops --per-task 20 --seed 0 --out $D/object_N200_p0.parquet
stage "export_N50" $D/object_N50_p0.parquet python -m ftr.build_data export-rlds --rlds $FTR_RLDS_DIR/libero_object_no_noops --per-task 5 --seed 0 --out $D/object_N50_p0.parquet
mix() { python -m ftr.build_data mix --arm $1 --states $D/states_train.parquet --self $R/scripted $R/scripted_t0b --rehearsal $D/spatial_rehearsal.parquet --out $D/$1.parquet 2>&1 | tee -a $L/a8_mix.log; }
need $D/spatial_rehearsal.parquet $R/scripted/episodes.parquet && { stage "mix_A" $D/A.parquet mix A; stage "mix_C" $D/C.parquet mix C; }
sync_logs

# ================================ B. align + gate ===================================================
need $D/A.parquet && stage "train_A" $CKPTS/A_s0/DONE train A_s0 $FTR_P_DIR $D/A.parquet
need $D/C.parquet && stage "train_C" $CKPTS/C_s0/DONE train C_s0 $FTR_P_DIR $D/C.parquet
rescue_adapters
need $CKPTS/A_s0/DONE && stage "reload_A" $L/b2_reload.ok bash -c "FTR_RUN=$CKPTS/A_s0 FTR_ADAPTER=$R/adapters/A_s0 FTR_PAIRS=$D/pairs_test.parquet pytest tests/test_reload.py -m gpu -v 2>&1 | tee $L/b2_reload.log | tail -5; grep -q '2 passed' $L/b2_reload.log && touch $L/b2_reload.ok"
for CK in A_s0 C_s0; do need $CKPTS/$CK/DONE && stage "score_$CK" $R/$CK/score_test/predictions.parquet score $CK; done
for CK in A_s0 C_s0; do need $CKPTS/$CK/DONE && stage "dev_$CK" $R/$CK/dev/DONE rollout $R/$CK/dev b4_dev_$CK --ckpt $CKPTS/$CK --suite obstacle_avoidance_human --tasks 0 1 2 --states 30-36 --classes harmful benign --templates h5 b5; done
sync_logs
# baseline half of the hazard matrix, two processes
if need $CKPTS/A_s0/DONE $CKPTS/C_s0/DONE; then
    [ -f $R/A_s0/hazard/DONE ] || bg rollout $R/A_s0/hazard b6_hazard_A_s0 --ckpt $CKPTS/A_s0 $HAZ
    [ -f $R/C_s0/hazard/DONE ] || bg rollout $R/C_s0/hazard b6_hazard_C_s0 --ckpt $CKPTS/C_s0 $HAZ
    waitbg
    for CK in A_s0 C_s0; do [ -f $R/$CK/hazard/DONE ] && echo "== hazard_$CK: ok" || echo "== hazard_$CK: FAILED" | tee -a $L/FAILED; done
fi
gate() { python -m ftr.analyze --runs $R/P_score $R/A_s0/score_test $R/C_s0/score_test $R/A_s0/dev $R/C_s0/dev $R/A_s0/hazard $R/C_s0/hazard \
    --p $FTR_P_DIR --pairs $CKPTS/A_s0:$CKPTS/C_s0 --targets $D/A.parquet $D/C.parquet --out $DATA/figures_gate 2>&1 | tee $L/b5_gate.log | head -80
    cat $DATA/figures_gate/refusal_rates.csv $DATA/figures_gate/contact_rates.csv 2>/dev/null | tee -a $L/b5_gate.log; }
need $R/A_s0/score_test/predictions.parquet $R/C_s0/score_test/predictions.parquet && stage "gate" $L/b5_gate.log gate
sync_logs

# ================================ C. personalize + measure =========================================
need $CKPTS/A_s0/DONE $D/object_N200_p0.parquet && stage "personalize_A200" $CKPTS/A_s0_N200_p0/DONE train A_s0_N200_p0 $CKPTS/A_s0 $D/object_N200_p0.parquet
need $CKPTS/C_s0/DONE $D/object_N200_p0.parquet && stage "personalize_C200" $CKPTS/C_s0_N200_p0/DONE train C_s0_N200_p0 $CKPTS/C_s0 $D/object_N200_p0.parquet
need $CKPTS/A_s0/DONE $D/object_N50_p0.parquet  && stage "personalize_A50"  $CKPTS/A_s0_N50_p0/DONE  train A_s0_N50_p0  $CKPTS/A_s0 $D/object_N50_p0.parquet
rescue_adapters
sync_logs
for CK in A_s0_N50_p0 A_s0_N200_p0 C_s0_N200_p0; do need $CKPTS/$CK/DONE && stage "score_$CK" $R/$CK/score_test/predictions.parquet score $CK; done
if need $CKPTS/A_s0_N200_p0/DONE $CKPTS/C_s0_N200_p0/DONE; then
    [ -f $R/A_s0_N200_p0/hazard/DONE ] || bg rollout $R/A_s0_N200_p0/hazard c3_hazard_A200 --ckpt $CKPTS/A_s0_N200_p0 $HAZ
    [ -f $R/C_s0_N200_p0/hazard/DONE ] || bg rollout $R/C_s0_N200_p0/hazard c3_hazard_C200 --ckpt $CKPTS/C_s0_N200_p0 $HAZ
    waitbg
    for CK in A_s0_N200_p0 C_s0_N200_p0; do [ -f $R/$CK/hazard/DONE ] && echo "== hazard_$CK: ok" || echo "== hazard_$CK: FAILED" | tee -a $L/FAILED; done
fi
sync_logs
# utility on the upstream LIBERO checkout: two processes (spatial + object), then the second checkpoint
export LIBERO_DIR=$W/LIBERO LIBERO_CONFIG_PATH=$W/LIBERO/.libero_config PYTHONPATH=$REPO:$W/openvla:$W/LIBERO
util() { rollout $R/$1/u_$2 c4_util_${1}_$2 --ckpt $3 --suite $2 --tasks all --states 0-4 --task-instruction; }
for CK in P A_s0 A_s0_N200_p0; do
    CKPT=$([ "$CK" = P ] && echo $FTR_P_DIR || echo $CKPTS/$CK)
    [ "$CK" = P ] || need $CKPTS/$CK/DONE || continue
    [ -f $R/$CK/u_libero_spatial/DONE ] || bg util $CK libero_spatial $CKPT
    [ -f $R/$CK/u_libero_object/DONE ]  || bg util $CK libero_object  $CKPT
    waitbg
    for S in libero_spatial libero_object; do [ -f $R/$CK/u_$S/DONE ] && echo "== util_${CK}_$S: ok" || echo "== util_${CK}_$S: FAILED" | tee -a $L/FAILED; done
done
export LIBERO_DIR=$W/LIBERO-Safety LIBERO_CONFIG_PATH=$W/LIBERO-Safety/.libero_config PYTHONPATH=$REPO:$W/openvla:$W/LIBERO-Safety
final() { python -m ftr.analyze --runs $R/P_score $R/*/score_test $R/*/hazard $R/*/u_* \
    --p $FTR_P_DIR --pairs $CKPTS/A_s0:$CKPTS/A_s0_N200_p0 $CKPTS/C_s0:$CKPTS/C_s0_N200_p0 \
    --targets $D/A.parquet $D/C.parquet $D/object_N50_p0.parquet $D/object_N200_p0.parquet --out $DATA/figures 2>&1 | tee $L/c5_analyze.log | head -100; }
stage "analyze" $DATA/figures/report.json final
echo "== ALL STAGES ATTEMPTED $(date)"; [ -f $L/FAILED ] && { echo "== failures:"; cat $L/FAILED; }
