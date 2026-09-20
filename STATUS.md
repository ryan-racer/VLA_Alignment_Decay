# Status

Last updated: 20 September 2026 (day 0)

**Implemented** = code exists; **verified** = has a passing test or fixture; **planned** = neither.

## Implemented and verified

Nothing yet.

## Implemented, not verified

Nothing yet.

## Planned (in `PLAN.md` day order)

- `scripts/setup_pod.sh` + `tests/test_env.py`
- `manifests/splits.csv` (tasks 1–3 train, 4–5 test), `manifests/instructions.csv`
- `ftr/envs.py` + `tests/test_fixtures.py` (Issue #3 patch validated)
- `ftr/codec.py` + `tests/test_codec.py`
- `ftr/rollout.py` — P baseline on test states (harmful / benign / blank), `libero_spatial` 20 episodes
- `ftr/build_data.py` — self-rollouts, RLDS export, baked mixes, rendered pairs
- `ftr/finetune.py` (five edits) + `tests/test_parity.py`, `tests/test_reload.py`
- `ftr/score.py`, `ftr/analyze.py`

## Decisions recorded today

- Substrate: LIBERO-Safety FSHOA L0 + LIBERO + OpenVLA harness; pip robosuite 1.4.1. RoboShackles = one-sentence criterion.
- Arms: A@0, A@200, C@0, C@200 closed-loop; A@50 offline. One alignment seed, stated in the abstract.
- Dropped after review: Rc comparator, N\*, replay, matched-update, alignment seeds 2–3, RoboShackles OOD panel, BDDL twins, Docker, YAML/provenance/CLI scaffolding.
- Two held-out layouts (tasks 4–5), 25 states each.
- Refusal target and scorer share `zero_bins()`; ±1 bin primary.
- Published LIBERO-Safety hand-suite collision rates not used as baselines.

## Open items from day 0

- [ ] Prior RoboShackles replication code (codec, evaluator, 64 tests) referenced in the proposal: located / declared absent
- [ ] GPU rented (RunPod A100 80 GB + network volume); exact base-image tag confirmed
- [ ] `setup_pod.sh` run; flash-attn compiled; `pytest tests/test_env.py` passes; `requirements.txt` written
