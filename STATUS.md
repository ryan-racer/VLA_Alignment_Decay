# Status

Last updated: 20 September 2026 (day 0)

**Implemented** = code exists; **verified** = has a passing test or fixture; **planned** = neither.

## Implemented and verified

- `scripts/setup_mac.sh` — Phase-0 venv (Python 3.10 via uv, CPU torch, transformers 4.40.1, openvla `c8f03f4` with `--no-deps`, P's non-weight files). Runs clean.
- `ftr/codec.py` + `tests/test_codec.py` — 9/9 on the Mac against P's real `libero_spatial` stats. Encoded zero = center_idx `[112, 109, 127, 129, 137, 142]`; one bin ≈ 6–7 mm translation, 0.0008–0.0015 rad rotation.
- `manifests/splits.csv` — FSHOA L0 tasks 0–2 train, 3–4 test. All five are LIBERO-10 layouts with a hand added; no overlap with `libero_object`.

## Implemented, not verified

- `scripts/setup_pod.sh` — pins openvla `c8f03f4`, LIBERO `8f1084e`, LIBERO-Safety `19ec8df`; syntax-checked only
- `patches/libero_safety_issue3.patch` — `git apply --check` passes against LIBERO-Safety `19ec8df`; behaviour untested until `test_fixtures.py`
- `tests/test_env.py` — 12 checks; never run (needs the pod)

## Planned (in `PLAN.md` day order)

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

- [ ] GPU rented (RunPod A100 80 GB + network volume); exact base-image tag confirmed
- [ ] `setup_pod.sh` run; flash-attn compiled; `pytest tests/test_env.py` passes; `requirements.txt` written
