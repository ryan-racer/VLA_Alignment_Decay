# Forgetting to Refuse

Does benign LoRA personalization erode a taught action-level safeguard in OpenVLA? A protocol-plus-pilot study for a 4-page workshop paper (target: SPAIS @ CoRL 2026, deadline 1 Oct 2026 AoE).

| File | What it is |
|---|---|
| [`PLAN.md`](PLAN.md) | **What and when.** Arms, gates, split, day-by-day, reporting checklist |
| [`IMPLEMENTATION.md`](IMPLEMENTATION.md) | **How.** Environment script, ~500 lines across six modules, five tests, gotchas, what not to adopt |
| [`STATUS.md`](STATUS.md) | Implemented / verified / planned. Updated as work lands |
| [`docs/Forgetting_to_Refuse_Project_Proposal_v1.pdf`](docs/Forgetting_to_Refuse_Project_Proposal_v1.pdf) | The scientific design this sprint executes a subset of |
| [`docs/code_reuse_report.md`](docs/code_reuse_report.md) | Line-level sources for `IMPLEMENTATION.md` |
| [`ftr/`](ftr/) | Code. Flat package on `PYTHONPATH`, never installed. Empty as of 20 Sep 2026 |

**Substrate:** LIBERO-Safety FSHOA L0 hand scenes for hazards, upstream LIBERO for utility, OpenVLA's stock harness throughout, pip `robosuite==1.4.1` (never the fork's vendored copy).

**Runtime:** Python 3.10, OpenVLA's pinned stack, set up by `scripts/setup_pod.sh` on one Linux + NVIDIA box (Lambda by default). The whole experiment is `scripts/run_all.sh`:

```
git clone https://github.com/ryan-racer/VLA_Alignment_Decay.git ~/ftr/repo
bash ~/ftr/repo/scripts/setup_pod.sh                     # ~20 min, ~31 GB of downloads
source ~/ftr/env.sh && pytest tests/test_env.py tests/test_fixtures.py tests/test_parity.py -m gpu
export GH_TOKEN=...                                      # optional: logs + figures pushed to logs/run/ every 10 min
nohup bash scripts/run_all.sh > ~/ftr/run_all.out 2>&1 &
```

Re-running `run_all.sh` resumes; failures and skipped stages are listed in `$DATA/logs/run/FAILED`.

Weights, datasets and raw run outputs are never committed (see `.gitignore`).
