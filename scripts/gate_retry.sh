#!/usr/bin/env bash
# Gate B's one pre-registered retry (PLAN.md): run after run_all.sh stopped at the gate with
# gate.next = "retry once ...". Keeps attempt 1 for the record ($DATA/attempt1: its data, adapters, scores, dev
# rollouts, gate report), rebuilds BOTH arms with twice the counterfactual pairs (C stays matched by mix's rule), and
# leaves everything that does not depend on the arms (scripted labels, P's baseline, exports, pairs) in place.
#
#   bash ~/ftr/repo/scripts/gate_retry.sh && nohup bash ~/ftr/repo/scripts/run_all.sh >> $DATA/run_all.out 2>&1 &
set -euo pipefail
source ${W:-$HOME/ftr}/env.sh
L=$DATA/logs/run; R=$DATA/runs; D=$DATA/data; CKPTS=$W/ckpt; OLD=$DATA/attempt1
RETRY_MIX="--n-move 600 --per-episode 10"   # 600 frames x (movement + no-op), <= 10 frames per scripted episode

[ ! -e $OLD ] || { echo "the one retry was already used ($OLD exists); PLAN.md: a second failure ends at the negative-result paper"; exit 1; }
! pgrep -f "[r]un_all.sh" >/dev/null || { echo "run_all.sh is still running"; exit 1; }
python - "$DATA/figures_gate/report.json" <<'EOF' || exit 1
import json, sys
g = json.load(open(sys.argv[1]))["gate"]
if not g["next"].startswith("retry once"):
    sys.exit(f"the gate's pre-registered next step is not a retry: {g['next']}")
EOF

mkdir -p $OLD/data $OLD/runs/adapters $OLD/logs
mv $D/A.parquet $D/C.parquet $D/pairs_train.parquet $OLD/data/
[ ! -d $R/P/score_train ] || mv $R/P/score_train $OLD/runs/P_score_train   # scored on attempt 1's pairs_train
for CK in A_s0 C_s0; do
    [ ! -d $R/$CK ] || mv $R/$CK $OLD/runs/$CK
    [ ! -d $R/adapters/$CK ] || mv $R/adapters/$CK $OLD/runs/adapters/$CK
    rm -rf $CKPTS/$CK   # merged weights (~15 GB each); attempt 1's adapter is kept above
done
mv $DATA/figures_gate $OLD/figures_gate
for f in FAILED b2_reload.ok; do [ ! -f $L/$f ] || mv $L/$f $OLD/logs/; done
echo "$RETRY_MIX" > $DATA/mix_args   # run_all.sh passes it to both mixes, also after a later resume
echo "attempt 1 kept in $OLD; arms will be rebuilt with: $RETRY_MIX"
echo "now relaunch run_all.sh (it retrains A_s0 and C_s0 and re-runs the gate)"
