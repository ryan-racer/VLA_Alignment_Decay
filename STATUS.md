# Status

Last updated: 20 September 2026 — **Phase 1 complete on Colab A100-40GB** (env 12/12, fixtures 4/4, parity 2/2, smoke 10 episodes). Next: `notebooks/phase2_colab.ipynb` section A.

**Implemented** = code exists; **verified** = has a passing test or fixture; **planned** = neither.

## Implemented and verified (on the Mac, `pytest -m "not gpu"` → 21 passed)

- `scripts/setup_mac.sh` — Python 3.10 venv via uv, CPU torch, `transformers==4.40.1`, openvla `c8f03f4` with `--no-deps`, P's non-weight files.
- `ftr/_openvla.py` — loads the four leaf openvla modules by path (`import prismatic` needs TensorFlow).
- `ftr/codec.py` + `tests/test_codec.py` (9) — encoded zero = center_idx `[112, 109, 127, 129, 137, 142]` under P's real stats; one bin ≈ 6–7 mm translation, 0.8–1.5 mrad rotation; label string re-tokenizes to `29871` + 7 ids.
- `ftr/data.py` + `tests/test_data.py` (5) — stock prompt template, 8 supervised positions, no-op rows equal `zero_bins()`, collator contract.
- `ftr/analyze.py` + `tests/test_analyze.py` (7) — Wilson, exact McNemar, percentile paired bootstrap, per-state aggregation, Rs exclusions, both figures render.
- `manifests/splits.csv` — FSHOA L0 tasks 0–2 train, 3–4 test (four LIBERO-10 layouts + one bowl-to-plate, each + hand; disjoint from `libero_object`).

## Verified on the pod (Phase 1, logs in `logs/phase1/`)

- `scripts/setup_pod.sh`, `patches/libero_safety_issue3.patch`, `ftr/envs.py` (restore, costs, fixtures), `ftr/rollout.py` (model path), `ftr/codec.py`, `ftr/data.py` parity with `RLDSBatchTransform` and eval preprocessing.
- Smoke: P contacts the hand in 9/10 episodes on task 3 (harmful 25–240 steps, benign 166–194, blank 24–35); 0.52 s/step.

## Implemented, not verified (pod)

- `scripts/setup_pod.sh` — pins openvla `c8f03f4`, LIBERO `8f1084e`, LIBERO-Safety `19ec8df`; `bash -n` only.
- `patches/libero_safety_issue3.patch` — applies cleanly to `19ec8df`; behaviour = `tests/test_fixtures.py`.
- `ftr/envs.py::scripted_pickplace` + `ftr/rollout.py --scripted` — movement-label source (new after Phase 1). **Blind**; `tests/test_fixtures.py::test_scripted_pickplace_moves_without_contact`.
- `ftr/build_data.py` — render / pairs / export-rlds / mix. **Blind** (export-rlds needs TF).
- `ftr/score.py` — offline predictions. **Blind.**
- `ftr/finetune.py` — stock loop + the five edits, single-GPU. **Blind.**
- `scripts/train_A.sh`, `train_C.sh`, `personalize.sh` — the command lines.
- `tests/test_env.py` (12), `test_fixtures.py` (4), `test_parity.py` (2), `test_reload.py` (2) — collected, never run.
- `manifests/instructions.csv` — final: 4+2 harmful (4 verb families, unseen test verbs), 4+2 benign (all mention the hand; one hard negative), blank. Rationale in manifests/README.md.
- `paper/main.tex` — §1–2 drafted, §3–4 placeholders; not compiled (no LaTeX on the Mac). `refs.bib`: all 13 entries verified against arXiv + venue pages (authors, titles, venues); claim wording corrected where the source was narrower.

## Phase 1 checklist — `notebooks/phase1_colab.ipynb` runs all of this on Colab (A100); logs land in `MyDrive/ftr/logs/phase1/` and are pushed to `logs/phase1/`

- [ ] Colab A100 runtime, `GH_TOKEN` secret set, run the notebook top to bottom (setup ≈ 10 min + ~36 GB of one-time downloads to Drive)
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
