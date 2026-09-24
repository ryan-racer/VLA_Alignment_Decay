#!/usr/bin/env bash
# EXPLORATORY, added after the confirmatory data were seen (24 Sep) - not part of prereg-v1.1. See PLAN.md, Amendment 2.
#
# The confirmatory personalization is 3 epochs of N=200 (5,505 updates); it ends at ~50-57% action-token accuracy,
# far short of OpenVLA's own LIBERO fine-tunes (trained to >=95%). This asks whether the result depends on that dose:
# the same recipe (same data, flags, seed 0) continued from A_s0 / C_s0 for DOSE_EPOCHS epochs in ONE run, one adapter
# snapshot per epoch, then closed-loop hazard (harmful + benign) and utility at the depths in DOSE_DEPTHS, and offline
# scoring of every snapshot. Its first 3 epochs retrace the confirmatory run (constant LR, same seed and data order; not
# bit-identical, flash-attn), so the epoch-3 snapshot is a replication check of A_s0_N200_p0.
#
# It changes nothing in run_all.sh or ftr/, and refuses to start while run_all.sh is running (shared GPU):
#   nohup bash ~/ftr/repo/scripts/dose_response.sh > $DATA/dose.out 2>&1 &
# Resumable: every stage is skipped if its done-file exists. ~30-40 GPU-h with the defaults.
set -o pipefail
source ${W:-$HOME/ftr}/env.sh
pgrep -f "scripts/run_all.sh" > /dev/null && { echo "run_all.sh is still running: start this after it finishes"; exit 1; }
L=$DATA/logs/dose; R=$DATA/runs; D=$DATA/data; CKPTS=$W/ckpt; mkdir -p $L
EPOCHS=${DOSE_EPOCHS:-12}                 # 4x the confirmatory dose
DEPTHS=${DOSE_DEPTHS:-"6 12"}             # epochs evaluated in closed loop (0 and 3 come from the confirmatory run)
# updates per epoch, from the confirmatory run itself (3 epochs)
SPE=${DOSE_SPE:-$(( $(sed -n 's/^updates=//p' $CKPTS/A_s0_N200_p0/DONE) / 3 ))}
echo "dose_response: $EPOCHS epochs x $SPE updates, closed loop at epochs $DEPTHS | code $(git -C $REPO describe --tags --always --dirty) $(date)"

fail() { echo "== $1: FAILED $(date +%H:%M) $2" | tee -a $L/FAILED; }
stage() {  # stage <name> <done-file> <cmd...>
    local name=$1 done=$2; shift 2
    [ -f "$done" ] && { echo "== $name: done"; return 0; }
    echo "== $name: start $(date +%H:%M)"
    if "$@" && [ -f "$done" ]; then echo "== $name: ok $(date +%H:%M)"; else fail $name; return 1; fi
}
need() { for f in "$@"; do [ -f "$f" ] || { fail "skipped (missing $f)"; return 1; }; done; }
PIDS=""; bg() { "$@" > /dev/null 2>&1 & PIDS="$PIDS $!"; }; waitbg() { [ -z "$PIDS" ] || wait $PIDS; PIDS=""; }
rollout() {  # as in run_all.sh: resumed once after a crash or a hung episode
    local out=$1 log=$2 try; shift 2
    for try in 1 2; do python -m ftr.rollout --resume --out $out "$@" >> $L/$log.log 2>&1 && return 0; done; return 1
}
score() { python -m ftr.score --ckpt $2 ${4:+--adapter $4} --pairs $D/pairs_$3.parquet --out $R/$1/score_$3 > $L/score_$1_$3.log 2>&1; }
upstream() ( export LIBERO_DIR=$W/LIBERO LIBERO_CONFIG_PATH=$W/LIBERO/.libero_config PYTHONPATH=$REPO:$W/openvla:$W/LIBERO; "$@" )
util() { upstream stage util_$1_$2 $R/$1/u_$2/DONE rollout $R/$1/u_$2 util_$1_$2 --ckpt $CKPTS/$1 --suite $2 --tasks all --states 0-4 --task-instruction; }
# the confirmatory personalization's flags (run_all.sh train()), longer, with a snapshot every epoch
train_long() {
    python -m ftr.finetune --vla_path $CKPTS/$1 --data_parquet $D/object_N200_p0.parquet --run_root_dir $CKPTS \
        --run_id ${1}_N200_long --adapter_tmp_dir $R/adapters --epochs $EPOCHS --batch_size 8 --grad_accumulation_steps 2 \
        --learning_rate 5e-4 --lora_rank 32 --seed 0 --save_every $SPE >> $L/train_${1}_N200_long.log 2>&1
}
# a snapshot as a merged checkpoint <arm>_N200_e<k> (rollouts need one); finetune --merge_only reads the adapter from
# <adapter_tmp_dir>/<run_id>, so link the snapshot under that name, with the run's dataset statistics beside it
merge_depth() {
    local n=${1}_N200_e$2 snap=$R/adapters/${1}_N200_long@$(( $2 * SPE )) a=$R/adapters_dose/${1}_N200_e$2
    mkdir -p $a && ln -sf $snap/* $a/ && cp $R/adapters/${1}_N200_long/dataset_statistics.json $a/ &&
        python -m ftr.finetune --merge_only true --vla_path $CKPTS/$1 --run_root_dir $CKPTS --run_id $n \
            --adapter_tmp_dir $R/adapters_dose >> $L/merge_$n.log 2>&1
}

# 1. train: one continuous run per arm (the GPU holds one training job at a time)
for ARM in A_s0 C_s0; do need $CKPTS/$ARM/DONE $D/object_N200_p0.parquet && stage train_${ARM}_long $CKPTS/${ARM}_N200_long/DONE train_long $ARM; done
# 2. merge the closed-loop depths
for K in $DEPTHS; do for ARM in A_s0 C_s0; do
    need $R/adapters/${ARM}_N200_long@$(( K * SPE ))/adapter_config.json && stage merge_${ARM}_e$K $CKPTS/${ARM}_N200_e$K/DONE merge_depth $ARM $K
done; done
# 3. closed loop: the confirmatory test states, harmful + benign (blank dropped to halve the cost), A and C in parallel
for K in $DEPTHS; do
    for ARM in A_s0 C_s0; do N=${ARM}_N200_e$K
        need $CKPTS/$N/DONE && bg stage hazard_$N $R/$N/hazard/DONE rollout $R/$N/hazard hazard_$N --ckpt $CKPTS/$N \
            --suite obstacle_avoidance_human --tasks 3 4 --states 25-49 --classes harmful benign --templates h6 b5 --video 2
    done; waitbg
done
# 4. utility at each depth: Unew (LIBERO-Object) for both arms, Uold (LIBERO-Spatial) for A
for K in $DEPTHS; do
    for ARM in A_s0 C_s0; do N=${ARM}_N200_e$K; SUITES="libero_object"; [ $ARM = A_s0 ] && SUITES="libero_spatial libero_object"
        need $CKPTS/$N/DONE || continue; for SU in $SUITES; do bg util $N $SU; done; waitbg
    done
done
# 5. offline decay curve: every snapshot of both runs, unmerged on its parent, on the test and retention sets
for ARM in A_s0 C_s0; do
    for SN in $(ls -d $R/adapters/${ARM}_N200_long@* 2>/dev/null | sort -t@ -k2 -n); do N=$(basename $SN)
        for SET in test train; do stage score_${N}_$SET $R/$N/score_$SET/predictions.parquet score $N $CKPTS/$ARM $SET $SN; done
    done
done
# 6. exploratory report: every depth against the confirmatory endpoints
uid() { sed -n 's/^run_uid=//p' $CKPTS/$1/DONE 2>/dev/null; }
expect() { for d in $CKPTS/*/DONE; do [ -f "$d" ] || continue; local n=$(basename $(dirname $d)); echo "$n=$(uid $n)"; done; }
PAIRS=""; DID=""
for K in $DEPTHS; do
    PAIRS="$PAIRS A_s0:A_s0_N200_e$K C_s0:C_s0_N200_e$K A_s0_N200_e$K:C_s0_N200_e$K A_s0_N200_p0:A_s0_N200_e$K"
    DID="$DID A_s0:A_s0_N200_e$K:C_s0:C_s0_N200_e$K"
done
analyze_dose() {
    python -m ftr.analyze --out $DATA/figures_dose --runs "$R/*/score_test" "$R/*/score_train" "$R/*/hazard" "$R/*/u_*" \
        --p P --pairs $PAIRS --did $DID --expect-uid $(expect) > $L/analyze_dose.log 2>&1
}
stage analyze_dose $DATA/figures_dose/report.json analyze_dose
echo "== DOSE STAGES ATTEMPTED $(date)"; [ ! -f $L/FAILED ] || { echo "== failures:"; cat $L/FAILED; }
