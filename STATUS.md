# Status

Last updated: 21 September 2026, 05:10 UTC — **Phase 2 running unattended on Colab** (`scripts/run_all.sh`, logs sync to `logs/phase2/`).

## 22 Sep (later) — textbook fixes from the external review, before the Lambda run

Design: counterfactual paired no-op frames (same image, harmful → stay / benign → move; first frame of every trajectory included); half the harmful templates contain the task text; offline eval on first AND mid-trajectory frames (frame × instruction table). Measurement: LIBERO-Safety's any-violation rate is the headline (robot-hand contact as breakdown); one 520-step horizon; end-effector drift logged. Training: stock OpenVLA augmentation (dlimp, RLDSDataset kwargs) + stock center crop at eval; RLDS export resized as the stock pipeline. Rigor: 3 alignment seeds offline (closed loop seed 0); paired sign-flip/McNemar tests for every change; A-vs-C difference-in-differences; Gate B automated and halting; run_uid/base_uid provenance with lineage checks; atomic outputs; pinned HF revisions; the contact patch must be applied. Reviewed twice by an independent code-review pass; 32 Mac tests pass, 27 GPU tests collected (not yet run).

## 22 Sep — moving to Lambda, full re-run from scratch

Code review found ten bugs, all fixed in `6ef2f18`. Three of them change results: C's extra movement rows duplicated the shared rows; P's hazard baseline was missing from the final analysis; the gate pooled dev and test scenes. So every Colab number below is a **pilot**. The paper's numbers come from one clean `scripts/run_all.sh` run on a Lambda GPU (see README). Colab pilot, second clean-slate A (useful as a variance estimate): A@0 refusal harmful/benign/blank 0.76/0.38/0.34 → A@200 0.25/0.69/0.20; C@200 0.07/0.39/0.12; hazard harmful contact A@0 0/50, C@0 19/50, A@200 2/28 (partial), C@200 31/39 (partial).

## Overnight 20→21 Sep: what happened and what to do in the morning

Done: states rendered (90 train / 50 test), 250 test pairs, scripted movement labels (task 0 re-run with `--no-place --clearance 0.45`: 30/30 clean; 62/90 episodes violation-free after the any-constraint filter), P baseline (hazard 150 episodes + 250 offline pairs), exports, mix (A = 300 noop / 300 move / 300 rehearsal; C = 450 move / 450 rehearsal), train A and C (171 updates, 3.5 min each), score A.

Fixed tonight (all pushed): hazard horizon never applied (suite tested against the class dict; P was rolled out at 520 steps, `analyze.truncate_to_horizon` re-scores it at 200/300); mix marked the task-0 re-run dirty via the first run's steps; checkpoints moved to the local disk (Drive is 10 GB, a merged checkpoint ~15 GB — Drive quota was hit once; Trash must be emptied).

**P baseline (50 test states, class horizons):** refusal 0.0 everywhere. Contact: harmful 47/50, benign 32/50, blank 38/50. Median time to contact: harmful ~30 steps, blank ~40, benign ~180. P moves into the hand regardless of instruction; the instruction changes *when*, not *whether*.

**A@0 offline (Gate B screens):**

| | Ru harmful | Rs benign | blank |
|---|---|---|---|
| A_s0, k=1 | 0.62 | 0.41 | 0.14 |
| k=0 | 0.50 | 0.41 | 0.12 |

Ru − Ru(P) = +62 ≥ 50 ✓. **Rs = 41% > 25% ✗** (over-refusal on held-out benign templates). Blank 14% ≪ Ru ✓. `reload_A` **FAILED**: merged vs unmerged adapter token agreement 14/20, max |Δaction| 0.495 (greedy decoding cascades; also possible bf16 merge wash-out of a 171-update LoRA delta). Pipeline continues regardless (dev, hazard A/C, gate, personalization, utility, analyze).

**10:33 UTC:** hazard matrix for A@0/C@0 complete (contact /50 — harmful: P 47, A 4, C 21; benign: P 32, A 6, C 0; A held 17 harmful + 11 benign episodes). GitHub syncs then stopped → the Colab runtime most likely disconnected before/while the gate ran. Checkpoints were local-only at that point. Fix pushed: adapters now save to Drive (`$R/adapters`, LoRA weights only) and `run_all` re-merges from a saved adapter instead of retraining; but A_s0/C_s0's adapters were never on Drive, so a re-run retrains them (not bit-identical: flash-attn backward). **Before trusting personalization results, check A#2 ≈ A#1**: score A#2 into `runs/A_s0/score_test_v2` and compare token agreement with `score_test`; if < ~95%, delete `runs/A_s0/{score_test,dev,hazard}` and `runs/C_s0/{...}` so they are re-measured on the retrained models (~6 h).

**21 Sep evening — restart from a clean slate.** The session that ran the hazard matrix died ~15:20 UTC in the final save of A200. On the fresh VM the Drive mount showed stale folders: over quota (04:38) the FUSE mount had faked deletes, so `runs/A_s0`, `runs/C_s0`, `runs/adapters/*` on the server were the first-night leftovers, and last night's dev/hazard results + the rescued A/C adapters were hidden duplicates or never uploaded. On top of that, the "strip embedding keys" step removed `lm_head` LoRA weights (it IS a target under all-linear), so the re-merged A scored 89/70/44 instead of 62/41/14. Decision: delete A_s0/C_s0/adapters/figures_gate/b5_gate.log on drive.google.com, empty Trash, retrain A and C from scratch and redo everything after them (~18 h, ~95 units of 200). Last night's A@0/C@0 numbers above remain valid as a *pilot* (same recipe, different seed realisation) but are not the paired baseline for the paper.

Morning checklist (in order):
0. Is the Colab runtime alive? If not: re-run the cell (it pulls the fixes). Then the A#2 ≈ A#1 check above.
1. `cat $DATA/logs/phase2/FAILED`; read `b5_gate.log`, `figures_gate/refusal_rates.csv`, `contact_rates.csv`, `outcomes.csv`, `refusal_breakdown.csv` (per template / per task: lexical shortcut vs scene?).
2. Merge check: `python -m ftr.score --ckpt $FTR_P_DIR --adapter $W/ckpt/adapters/A_s0 --pairs $DATA/data/pairs_test.parquet --out $DATA/runs/A_s0/score_adapter` and compare with `runs/A_s0/score_test`. If the unmerged adapter refuses much more, the merge is washing out the update → save the adapter-applied model in fp32-then-bf16 or evaluate unmerged.
3. Decide on A: if C@0's benign refusal is ~0 and A's 41% holds up in dev/hazard rollouts, retrain A (cheap: 4 min + 20 min score/dev) with more epochs (sharper boundary) and/or more move rows before re-running the hazard matrix (5 h). PLAN's pre-decided response: one diagnosis pass, then negative-result framing if unresolved.
4. Relax `tests/test_reload.py` to a ±1-bin criterion only if step 2 shows the merge is sound.


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
