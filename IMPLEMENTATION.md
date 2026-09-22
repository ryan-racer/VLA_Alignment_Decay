# Implementation plan

Revised 20 September 2026 (after simplification review). Companion to `PLAN.md`. Line-level sources are in `docs/code_reuse_report.md`.

## Principle

Install `openvla/openvla` @ `c8f03f4` and import from it. Copy exactly one file (`finetune.py`) and change ~25 lines. Write one honest rollout loop instead of pretending the stock eval script is reusable. About 500 lines of new code, flat package, never installed, no framework.

## Build order

Everything that can be written and verified without a GPU is done first, on the Mac. The pod is rented only when the remaining work is *running* things. Each step names what "verified" means locally; blind-written code is marked and gets its test on the pod.

### Phase 0 — Mac, no GPU

| # | Step | Files | Verified locally by | Done when |
|---|---|---|---|---|
| 0.1 | Local env | `scripts/setup_mac.sh` | — | venv with torch (cpu), `transformers==4.40.1`, `numpy==1.26.4`, `pandas scipy seaborn pyarrow pytest`; `pip install --no-deps -e /path/openvla` at `c8f03f4`; `hf download` of P's `config.json`, `tokenizer*`, `preprocessor_config.json`, `*_prismatic.py` (no weights) |
| 0.2 | Split | `manifests/splits.csv` | eyeball | five FSHOA L0 task names from the fork's `vla_safety_task_map.py`; tasks 1–3 train, 4–5 test |
| 0.3 | Codec | `ftr/codec.py`, `tests/test_codec.py` | `pytest` on Mac | `zero_bins()` under P's real `libero_spatial` stats lands near 113/110/128/130/138/143; round-trip within one bin; `refused()` correct on synthetic hold/contact token rows; `roboshackles_noop()` agrees with `refused(k=1)` on the same rows |
| 0.4 | Instructions | `manifests/instructions.csv` | you edit | ≥3 train + 2 test templates per class per task; benign templates share nouns with harmful ones; blank row included |
| 0.5 | Dataset | `ftr/data.py`, `tests/test_data.py` | `pytest` on Mac (CPU processor, no weights) | one synthetic row → `input_ids`/`labels` with exactly 8 supervised positions (7 action + EOS), prompt text matches `In: What action should the robot take to {x}?\nOut: `, no-op row's label tokens equal `zero_bins()`, gripper rule applied |
| 0.6 | Training script | `ftr/finetune.py` | `diff` against stock; read | the five edits and nothing else; `python -c "import ftr.finetune"` on Mac |
| 0.7 | Env wrapper **(blind)** | `ftr/envs.py` | read only | `make_env`, `reset_to` with reseed, `step_with_costs` re-evaluating each constraint, `load_init_states`, hold/contact fixtures |
| 0.8 | Rollout loop **(blind)** | `ftr/rollout.py` | read only | CSV instructions, token capture via `codec.generate_with_tokens`, terminate-on-contact, outcome column, `action_model`/`action_env` both logged, Parquet + `args.json` |
| 0.9 | Data builder, scorer | `ftr/build_data.py`, `ftr/score.py` | RLDS export runs on Mac **if** `tensorflow==2.15` installs; else blind | export writes `(image, instruction, raw_action)` Parquet; mix baker produces per-arm Parquet with the planned row counts; `score.py` reads `pairs.parquet` and writes `predictions.parquet` |
| 0.10 | Analysis | `ftr/analyze.py`, `tests/test_analyze.py` | `pytest` on Mac with synthetic Parquet | Wilson, discordant pairs + exact McNemar, percentile paired bootstrap, per-state aggregation, outcome taxonomy; Fig. 1, Fig. 2, Table 1 render from fake records |
| 0.11 | Pipeline | `scripts/run_all.sh` | read | every command line of the experiment, in order, resumable; the exact configs |
| 0.12 | Paper §1–2 | `paper/main.tex` (from the CoRL template) | compiles | motivation and protocol drafted; figure/table placeholders; every scope cut stated |
| 0.13 | Tests scaffold | `tests/test_fixtures.py`, `tests/test_parity.py`, `tests/test_reload.py` | read | written against the interfaces above; skipped on Mac (`pytest.mark.gpu`) |

**Phase 0 done 20 Sep 2026** — 21 Mac tests pass, 20 pod tests collected. ~2,000 lines including the copied `finetune.py` and tests; the earlier "~200 lines" counted only the adapters. Every remaining task is "run it".

### Phase 1 — one GPU session (Colab A100 via `notebooks/phase1_colab.ipynb`, or a RunPod pod)

| # | Step | Done when |
|---|---|---|
| 1.1 | `setup_pod.sh` | flash-attn compiled, clones at pinned SHAs, patch applied, downloads present |
| 1.2 | `pytest tests/test_env.py` | green; `pip freeze > requirements.txt`; commit |
| 1.3 | `pytest tests/test_fixtures.py` | patched predicate fires on contact, not on hold |
| 1.4 | `pytest tests/test_parity.py` | our tokens and pixels match `RLDSBatchTransform` + eval preprocessing |
| 1.5 | `rollout.py` on one FSHOA state with P | an action comes out; a video exists; s/episode measured |

If 1.1–1.5 pass, the environment is real and day 1 of PLAN.md is done. Stop the pod; the volume keeps everything.

### Phase 2 — pod, continuous

PLAN.md days 3–9: P baseline on test states, self-rollouts, data build, A/C training, gates, hazard matrix, utility, analysis. Writing happens on the Mac in parallel from day 7.

## Environment (`scripts/setup_pod.sh`; any Linux + NVIDIA box, Lambda by default)

Parametrized by `W` (local: venv + clones), `DATA` (persistent: HF cache, runs, data, logs) and `CUDA` (`cu118` default → prebuilt flash-attn wheel, no compile; `cu121` compiles). Defaults: `W=DATA=$HOME/ftr`. On Lambda, point `DATA` at a persistent filesystem (`/lambda/nfs/<fs>/ftr`) if results must outlive the instance.

RunPod secure A100 80 GB ($1.59/h) + 200 GB network volume mounted at `/workspace`. No Docker build: start from RunPod's `pytorch 2.2.0 / py3.10 / cuda 12.1.1 devel` image (verify the exact tag on day 0) and run an idempotent `scripts/setup_pod.sh`:

```
# all on /workspace so a pod swap costs nothing
python3.10 -m venv /workspace/venv && source /workspace/venv/bin/activate
apt-get install -y cmake ninja-build ffmpeg libegl1 libgl1 libglew-dev libosmesa6-dev \
    linux-headers-generic libmagickwand-dev imagemagick libfontconfig1-dev
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl NVIDIA_DRIVER_CAPABILITIES=all HF_HOME=/workspace/hf
pip install torch==2.2.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/cu121
git clone openvla /workspace/openvla && git -C /workspace/openvla checkout c8f03f4 && pip install -e /workspace/openvla
MAX_JOBS=4 pip install flash-attn==2.5.5 --no-build-isolation        # source build, once; no cu12 wheel for this combo
pip install -r /workspace/openvla/experiments/robot/libero/libero_requirements.txt   # robosuite==1.4.1 bddl imageio[ffmpeg] ...
pip install numpy==1.26.4 mujoco==2.3.7 tensorflow-metadata==1.14.0 wandb==0.17.9 scipy pyarrow pandas seaborn pytest
git clone LIBERO /workspace/LIBERO ; git clone LIBERO-Safety /workspace/LIBERO-Safety   # pin SHAs in the script
git -C /workspace/LIBERO-Safety apply /workspace/repo/patches/libero_safety_issue3.patch
pip install "usd-core>=25.5" wand scikit-image     # fork extras ONLY; never its requirements.txt or third_party/robosuite
hf download openvla/openvla-7b-finetuned-libero-spatial --local-dir /workspace/hf/P
hf download openvla/modified_libero_rlds --repo-type dataset --include "libero_spatial_no_noops/*" --include "libero_object_no_noops/*"
hf download LIBERO-Safety/libero_safety_assets --repo-type dataset   # unzip into /workspace/LIBERO-Safety/libero/libero/assets/
printf 'benchmark_root: ...\n' > ~/.libero/config.yaml     # otherwise first import calls input() and hangs
pip freeze > /workspace/repo/requirements.txt              # after tests pass: this + the image tag is the environment record
```

Per job: `PYTHONPATH=/workspace/repo:/workspace/openvla:/workspace/LIBERO` for `libero_spatial`/`libero_object` (comparable to published numbers) or `…:/workspace/LIBERO-Safety` for FSHOA. Both register the package `libero`; never pip-install either.

## Layout

```
ftr/                 flat, on PYTHONPATH, never installed
  finetune.py        stock (~370) + ~25 changed lines, listed below
  codec.py     ~60   frozen-stats tokenizer wrap; zero_bins(); refused(ids, k); roboshackles_noop(a); generate_with_tokens()
  data.py      ~70   ParquetTransitions(Dataset) cloned from DummyDataset; labels via codec
  envs.py      ~80   make_env(bddl); reset_to() with reseed; step_with_costs(); load_init_states(); hold/contact fixtures
  rollout.py  ~100   (ckpt, task, states, instruction) → episodes.parquet; P baseline, self-rollouts, gate, hazard matrix, utility
  build_data.py ~110 RLDS export (TFDS, ~30); self-rollout filtering; category mix baked into rows; held-out states rendered into pairs.parquet; manifests
  score.py     ~50   model-only: pairs.parquet + ckpt → predictions.parquet (Ru/Rs/blank and the Gate-A probe)
  analyze.py   ~60   Ru/Rs/blank per state; contact rates; discordant pairs; percentile paired bootstrap; two figures, one table
patches/libero_safety_issue3.patch
manifests/*.csv      splits, instruction templates, pair ids — committed (.gitignore swallows Parquet)
scripts/setup_mac.sh, setup_pod.sh, run_all.sh   the exact command lines are the configs
tests/  test_codec.py test_data.py test_analyze.py        run on the Mac
        test_env.py test_fixtures.py test_parity.py test_reload.py   run on the pod (pytest.mark.gpu)
paper/               CoRL 2026 template + main.tex
requirements.txt     pip freeze on the pod, Phase 1
```

Stages have different dependency footprints (TF only in `build_data`, GPU in `finetune`/`rollout`/`score`, none in `analyze`); that is why there are six modules and not three. Run as `python -m ftr.<stage>`.

## The pieces

### `finetune.py` — one copy, five changes

1. `RLDSDataset` → `ParquetTransitions`; epoch loop over a shuffled `DataLoader(generator=g, num_workers>0)`.
2. **Seed everything** (`torch`, `numpy`, `random`, the loader generator) from `cfg.seed` and write it to `args.json`. The stock script does not seed; `init_lora_weights="gaussian"` and data order are random.
3. `clip_grad_norm_(trainable_params, 1.0)` before `optimizer.step()` (issues #299/#333).
4. Pass `--save_steps == --max_steps` so the merge runs once, at the end.
5. `merged_vla.config.norm_stats = {"libero_spatial": stats}` before `save_pretrained`; copy the three `*_prismatic.py` files into the merged dir.

Stage 2 is the same script with `--vla_path <merged_dir>`; it re-wraps with a fresh `LoraConfig`. Keep `draccus` inside this file only.

### `data.py`

Clone `DummyDataset` (`prismatic/vla/datasets/datasets.py` L180-232). Rows come from one Parquet per arm with the category mix already applied — A and C share the movement rows because they are the same rows; no samplers. The Dataset:

- Normalizes dims 0–5 with the **frozen `libero_spatial` q01/q99** from P's `config.json`; gripper stays in [0,1], +1 = open.
- Exposes `dataset_statistics = {"libero_spatial": <P's block>}` so `save_dataset_statistics` writes the file inference reads.
- Runs images through the same `get_libero_image` + 0.9 center-crop the evaluator uses (imported from `experiments.robot`).

**Gripper rule, written once:** the RLDS gripper label is derived from the *commanded* gripper (`1 - clip(a, 0, 1)` on the raw ±1 command), never from finger positions (a held object keeps the fingers apart). `rollout.py` tracks the last executed env gripper command; after `reset_to` it is open (1.0). The no-op label copies that value; `refused(…, gripper_expected_id)` requires the gripper token within ±k bins of it. Per-step rows carry the **pre-step** frame, pre-step gripper and the action taken from that frame. `rollout.py` logs both `action_model` (7-D, gripper ∈ [0,1]) and `action_env` (post-invert, ±1).

### `codec.py`

`zero_bins()` = `tokenize(normalize(zeros(6), libero_spatial))` — near 113/110/128/130/138/143, not six 128s — used by **both** the training label and the scorer, which is why they live in one file. `generate_with_tokens()` calls `vla.generate(input_ids, max_new_tokens=7, do_sample=False, **inputs)` and applies the four decode lines from `modeling_prismatic.py` L512-534; never via `predict_action(**kwargs)`.

### `envs.py`

- `make_env(bddl_path)` → `OffScreenRenderEnv(bddl_file_name=…, camera_heights=256, camera_widths=256)`.
- `reset_to(env, state)`: `env.seed(0)` **before every** `env.reset()`, then `set_init_state(np.asarray(state))`, then 10 dummy steps. Fixture poses drift 3 mm otherwise.
- `step_with_costs(env, action)`: after `env.step`, loop `env.env.parsed_problem['constraints']` and `_eval_predicate` each — `info['cost']` collapses same-named predicates and is suppressed on the success step. Terminate on contact in our loop; the env never does.
- Init states: fork's `get_task_init_states(level, i)`, `torch.load(..., weights_only=False)`, tensors of shape (50, 60).
- Fixtures: `hold` = zeros over the horizon; `contact` = P-loop on `transform_utils.get_pose_error` then ~40 press steps toward the hand.

### `rollout.py`

The stock `run_libero_eval.py` loop, rewritten because it needs: any BDDL path, reseed, instruction from CSV (`--classes`, `--template-split`, `--templates`), per-constraint costs, token capture, terminate-on-contact, and Parquet output: `episodes.parquet` (rewritten per episode) and `steps_t<task>.parquet` (per task; `--store-images` keeps the pre-step frame for self-rollouts). Outcome ∈ {contact, success, held, moved} for hazard runs, {success, timeout, contact} for utility runs. Headline refusal column is `refused_k1_gripper` (PLAN's criterion); `refused_k1` is kept alongside. Uses `get_vla`, `get_processor`, `get_libero_image`, `normalize_gripper_action`, `invert_gripper_action`, `save_rollout_video` from the installed `experiments.robot`. Always `--center_crop True`.

### `build_data.py`, `score.py`, `analyze.py`

`build_data.py` renders each held-out state once (reseed, restore, dummy steps) and stores the 224×224 image in `pairs.parquet`, so `score.py` is model-only and runs 2–3 copies on one card. `analyze.py` touches no GPU: per-state aggregation, Wilson intervals, discordant-pair counts with exact McNemar, `scipy.stats.bootstrap(paired=True, method='percentile')`, seaborn `lineplot(units=seed, estimator=None)`, PDF + PNG.

Records: `runs/<arm>_<seed>_<N>/{args.json, episodes.parquet | predictions.parquet}`; `args.json` holds the full argument set, seed and git SHA. A crashed run leaves no Parquet, which is all the atomicity needed.

## Tests (the five that matter)

| Test | Asserts |
|---|---|
| `test_env` | versions (`mujoco==2.3.7`, pip robosuite with `output_max` 0.05); one EGL frame; P `norm_stats` has one key; same state restored twice → poses equal within 1e-4 |
| `test_codec` | encoded zero round-trips within one bin per dim; `refused()` true on hold tokens, false on contact tokens |
| `test_parity` | one `modified_libero_rlds` sample through the real `RLDSBatchTransform` matches our `input_ids` **and** `labels` token-for-token, **and** our `pixel_values` equal the eval preprocessing of the same frame |
| `test_fixtures` | contact flips the patched `checkrobotcontact`; hold does not; timeout is neither |
| `test_reload` | merged model has exactly one `norm_stats` key; ≥95% token agreement with the unmerged adapter over 20 fixed observations (bf16 merge is not bit-exact) |

## Gotchas (each costs a day if missed)

| # | Trap | Fix |
|---|---|---|
| 1 | Fork's vendored robosuite: `osc_pose.json` `output_max ±2`, `kp 750` → 2 m steps; looks like "OpenVLA is incompetent on hand scenes" | Never install it; pip `robosuite==1.4.1`; assert in `test_env` |
| 2 | LIBERO-Safety HF dataset: metre-unit actions, TSA/FSHOA pooled, AV1, needs `lerobot==0.3.3` | Not used |
| 3 | Stats key is `libero_spatial`, not `_no_noops`; base `openvla-7b` has 25 OXE keys | `unnorm_key="libero_spatial"` everywhere; one-key assertion at reload |
| 4 | `finetune.py` recomputes stats from RLDS, merges every `save_steps`, keeps base `norm_stats`, does not seed | The five edits |
| 5 | `env.seed(0)` once per env → fixture poses drift between episodes | Reseed before every reset; determinism test |
| 6 | Issue #3: `checkrobotcontact` never fires (int ids vs names) | Checked-in patch + fixture assertion |
| 7 | `info['cost']` collapses same-named predicates; suppressed on `done`; env never terminates on cost | Per-constraint `_eval_predicate`; terminate in our loop |
| 8 | Encoded zero ≠ bin 128 | `zero_bins()` shared by label and scorer |
| 9 | No grad clipping at LR 5e-4 → collapse to identical tokens | `clip_grad_norm_(1.0)` |
| 10 | `--center_crop True` required at eval; training images must match | Same preprocessing function both sides; pixel parity test |
| 11 | `mujoco` 3.x makes objects slide after `set_init_state` | Pin 2.3.7 |
| 12 | Missing `~/.libero/config.yaml` → interactive `input()` hangs headless jobs | `setup_pod.sh` writes it |
| 13 | `.gitignore` ignores `*.parquet` | Manifests are CSV |
| 14 | Fork's `env_wrapper.py` imports `wand`, `skimage` at import time | `libmagickwand-dev` + `pip install wand scikit-image` |
| 15 | Hazard horizon check keyed the suite against the class dict → every episode ran the suite's 520 steps (P baseline was rolled out this way) | `rollout.py` checks `FORK_SUITES`; `analyze.truncate_to_horizon` re-scores any episode on the 200/300 class horizon, so P is comparable |
| 16 | Drive is 10 GB; a merged 7B checkpoint is ~15 GB, five are trained | `run_all.sh` writes checkpoints to the local disk (`$W/ckpt`), results to Drive; a runtime death only forces retraining |

## Not adopted, and why

- **openvla-oft** (training or eval): transformers fork, 8-step chunks, breaks on Hub base checkpoints.
- **RLDS / `tfds build`** for our data: stats can't be frozen without patching anyway; two registry edits; issues #227/#312 unanswered.
- **LeRobot converters**: key-name mismatch; nobody has run the output through `finetune.py`.
- **LIBERO-PRO / LIBERO-plus** installs: Python 3.8 / torch 1.11. The blank/paraphrase probe is an instruction string in `score.py`.
- **BDDL twins** for harmful/benign: the env never feeds `:language` to the model; a CSV of instructions on identical states is the pair and the better release artifact.
- **SafeVLA, HazardArena, SafeManip, SafeVLA-Bench, RedVLA**: different simulators or no code.
- **Docker build, YAML configs, provenance module, CLI dispatcher, `src/` layout, `uv_build`, Hydra, DVC, MLflow, tyro, typer, pydantic, pingouin (GPL), bootstrapx, atomicwrites**: scaffolding for a 6-arm, 15-job sprint; shell scripts and `args.json` do the same job.
- **sdpa instead of flash-attn**: loads, but a 74% vs 84.7% gap was attributed to it; unverified parity.
- **48 GB cards**: batch 8 × accum 2 plausible, unmeasured; not worth the debugging day.
- **Fallback `sim.data.contact` scan**: written only if the fork fails to install.

## Cost

Training ≈ 5 jobs ≈ 3 GPU-h. Rollouts ≈ 750 episodes at ~0.28 s/step ≈ 12–15 GPU-h. **≈ $40 compute, ≈ $55 with storage.** Two or three eval or score processes fit on one 80 GB card (each ~15 GB) — the only free throughput lever. Measure s/step in the first 100 updates and s/episode on day 3.
