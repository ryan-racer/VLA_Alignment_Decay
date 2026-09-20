# Status

Last updated: 20 September 2026 — **Phase 0 complete** (IMPLEMENTATION.md § Build order). Next: Phase 1 on the pod.

**Implemented** = code exists; **verified** = has a passing test or fixture; **planned** = neither.

## Implemented and verified (on the Mac, `pytest -m "not gpu"` → 21 passed)

- `scripts/setup_mac.sh` — Python 3.10 venv via uv, CPU torch, `transformers==4.40.1`, openvla `c8f03f4` with `--no-deps`, P's non-weight files.
- `ftr/_openvla.py` — loads the four leaf openvla modules by path (`import prismatic` needs TensorFlow).
- `ftr/codec.py` + `tests/test_codec.py` (9) — encoded zero = center_idx `[112, 109, 127, 129, 137, 142]` under P's real stats; one bin ≈ 6–7 mm translation, 0.8–1.5 mrad rotation; label string re-tokenizes to `29871` + 7 ids.
- `ftr/data.py` + `tests/test_data.py` (5) — stock prompt template, 8 supervised positions, no-op rows equal `zero_bins()`, collator contract.
- `ftr/analyze.py` + `tests/test_analyze.py` (7) — Wilson, exact McNemar, percentile paired bootstrap, per-state aggregation, Rs exclusions, both figures render.
- `manifests/splits.csv` — FSHOA L0 tasks 0–2 train, 3–4 test (all five are LIBERO-10 layouts + hand; disjoint from `libero_object`).

## Implemented, not verified (pod)

- `scripts/setup_pod.sh` — pins openvla `c8f03f4`, LIBERO `8f1084e`, LIBERO-Safety `19ec8df`; `bash -n` only.
- `patches/libero_safety_issue3.patch` — applies cleanly to `19ec8df`; behaviour = `tests/test_fixtures.py`.
- `ftr/envs.py` — reseed-before-reset, per-constraint costs, `model_image`, gripper rule, hold/contact fixtures. **Blind.**
- `ftr/rollout.py` — closed-loop episodes → `episodes.parquet` + `steps.parquet`, token capture, outcome taxonomy. **Blind.**
- `ftr/build_data.py` — render / pairs / export-rlds / mix. **Blind** (export-rlds needs TF).
- `ftr/score.py` — offline predictions. **Blind.**
- `ftr/finetune.py` — stock loop + the five edits, single-GPU. **Blind.**
- `scripts/train_A.sh`, `train_C.sh`, `personalize.sh` — the command lines.
- `tests/test_env.py` (12), `test_fixtures.py` (4), `test_parity.py` (2), `test_reload.py` (2) — collected, never run.
- `manifests/instructions.csv` — final: 4+2 harmful (4 verb families, unseen test verbs), 4+2 benign (all mention the hand; one hard negative), blank. Rationale in manifests/README.md.
- `paper/main.tex` — §1–2 drafted, §3–4 placeholders; not compiled (no LaTeX on the Mac).

## Phase 1 checklist (one ~2 h pod session, then stop the pod, keep the volume)

- [ ] RunPod secure A100 80 GB + 200 GB network volume at `/workspace`; note the exact base-image tag
- [ ] `git clone` → `bash scripts/setup_pod.sh` (flash-attn compile is the long step)
- [ ] `source /workspace/env.sh && pytest tests/test_env.py -m gpu -v` green → `pip freeze > requirements.txt` → commit
- [ ] `pytest tests/test_fixtures.py -m gpu -v` — contact fires, hold does not
- [ ] `pytest tests/test_parity.py -m gpu -v` — tokens + pixels match OpenVLA's own pipeline
- [ ] `python -m ftr.rollout --ckpt /workspace/hf/P --suite obstacle_avoidance_human --tasks 3 --states 0-1 --classes harmful benign blank --out runs/smoke --video 2` — an action comes out, a video exists, s/episode measured

## Decisions recorded

- Substrate: LIBERO-Safety FSHOA L0 + LIBERO + OpenVLA harness; pip robosuite 1.4.1. RoboShackles = one-sentence criterion.
- Arms: A@0, A@200, C@0, C@200 closed-loop; A@50 offline. One alignment seed, stated in the abstract.
- Dropped after review: Rc comparator, N\*, replay, matched-update, alignment seeds 2–3, RoboShackles OOD panel, BDDL twins, Docker, YAML/provenance/CLI scaffolding.
- Instructions are passed at rollout time from `manifests/instructions.csv`; the env never feeds `:language` to the model.
- Training images = eval-preprocessed images (`envs.model_image`), so train/test pixels share one path.
- Refusal target and scorer share `Codec.zero_bins()`; ±1 bin primary.
- Published LIBERO-Safety hand-suite collision rates not used as baselines.
