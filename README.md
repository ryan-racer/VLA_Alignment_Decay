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

**Runtime:** Python 3.10, OpenVLA's pinned stack, set up by `scripts/setup_pod.sh` on a RunPod A100 network volume. `requirements.txt` is written from `pip freeze` once tests pass.

Weights, datasets and raw run outputs are never committed (see `.gitignore`).
