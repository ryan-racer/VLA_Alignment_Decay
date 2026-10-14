# Status

**Last updated: 13 October 2026. The project is closed. No further runs are planned in this repository.**

The project did not answer its question (does benign personalization erode a refusal taught to a VLA?). The reasons are in [`README.md`](README.md) under "Why it did not answer the question".

## Final state of each study

All runs used OpenVLA-7B on a single NVIDIA A100 (40 GB). Dates are 2026.

- **Frozen** is the commit date (US Central) of the commit the pre-registration tag points to.
- **Ran**: for the pilots and Study 1, whose logs are in this repository, the UTC dates of the automated log commits. For the other runs no logs are included, so this is only the window between the commit that froze the run and the commit that first reports its result.

| Study | Frozen (tag) | Ran | Final state | Branch |
|---|---|---|---|---|
| Colab pilots | none | 20 to 22 Sep | Treated as pilots only, after a code review found bugs that change results; not used for any reported number | `main` (`logs/phase1/`, `logs/phase2/`) |
| Study 1 (LIBERO-Safety hand scenes, n=50) | 22 Sep (`prereg-v1`, amended in `prereg-v1.1`) | 23 to 25 Sep | Complete. Primary null at a floor: harmful-instruction contact 0 of 50 before and 0 of 50 after personalization. The blocking gates passed; the two non-blocking over-refusal screens failed | `main` |
| Dose run (6 and 12 epochs) | Not pre-registered; exploratory (`PLAN.md` Amendment 2, committed 23 Sep after seed 0's Study 1 confirmatory rollouts had been seen) | within 23 to 26 Sep | Complete, exploratory only | `dose-response` |
| Study 2 (LIBERO-Object scenes with a hand, n=150) | 26 Sep (`prereg-v2`) | within 26 to 27 Sep | Confirmatory part complete. Primary significant (benign-instruction contact 0 to 40 of 150), but its manipulation check failed (task success 0 of 150 before and after), so it is not interpretable as erased safety. Its exploratory six-epoch runs were stopped part-way and its 12-epoch runs were not started | `study2` |
| Study 2b (hand beside the route) | 27 Sep (`prereg-v3`) | 27 Sep | Stopped at its hazard gate: comparator contact 6 of 50 and 0 of 50 against a required 20%. No confirmatory run | `study2b` |
| Post hoc runs (600 episodes) | 28 Sep (`posthoc-v2`, which replaced `posthoc-v1`; `posthoc-v1.1` changed only the setup script) | within 28 to 29 Sep | Complete, exploratory. Stopping specific to the harmful instruction at seed 0 (near-still in 148 vs 97 of 150 episodes); the declared replication criterion for seeds 1 and 2 was not met | `posthoc` |
| Draft paper | n/a | last commit 30 Sep | Four pages plus references, with a supplement. Never submitted to any venue; not peer reviewed | `paper` |

The numbers above are reported in the draft paper and its supplement on `paper` (`paper/main.tex`, `paper/supplement.tex`); Study 1's contact counts can also be checked against `figures/` on `main`. Per-episode results are not included in this repository for any run, and run logs are included for the pilots and Study 1 only.

## Never run

The original proposal (`docs/Forgetting_to_Refuse_Project_Proposal_v1.pdf`) included parts that were dropped at planning time (`PLAN.md`, "Scope decisions") and never run:

- the benign-stopping comparator;
- the safety-replay mitigation;
- the matched-update diagnostic.

Two further parts of the proposal were not carried out as proposed:

- **The demonstration-count sweep** (N = 10, 50, 200 in closed loop) was run only in part. N = 200 ran in closed loop. N = 50 was trained and scored offline only (seed 0). N = 10 was not run.
- **The D50 estimate** (the demonstration count at which refusal halves) was never computed; no code for it exists on any branch.

## Record keeping

- The commits behind `prereg-v1` and `prereg-v1.1` were on GitHub before Study 1 ran: that run's automated log commits on `main` descend directly from `prereg-v1.1`.
- `prereg-v2`, `prereg-v3`, the three `posthoc-*` tags and every branch except `main` were first pushed on 13 October 2026 (US time; 14 October UTC), after their data had been collected. Their timing rests on local commit dates and cannot be verified externally.
- The earlier status notes, up to 22 September, are the version of this file at tag `prereg-v1.1`.
