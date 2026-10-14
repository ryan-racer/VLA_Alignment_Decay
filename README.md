# Forgetting to Refuse

**Status, 13 October 2026: closed. This project did not answer its research question.**

## Summary

The question: if a robot foundation model is taught to refuse harmful instructions, and its owner then fine-tunes it on ordinary demonstrations, does the safeguard survive?

In September 2026 I taught OpenVLA-7B a no-op refusal (hold still when told to harm a human hand in LIBERO-Safety), personalized it with LoRA on 200 benign LIBERO-Object demonstrations, and measured hand contact in closed loop against a comparator never taught to refuse. The experiment could not tell whether the safeguard eroded. In the draft paper's words it is "a demonstration of this failure rather than a verdict on the safeguard's durability", because "a freeze was most of what there was to erase, and no policy demonstrated compliance with the harmful instruction."

## What was built and run

One author, 20 to 30 September 2026 (first to last commit), on a single NVIDIA A100 (40 GB). P is the released OpenVLA-7B LIBERO-Spatial checkpoint, A is P after refusal training, C is a comparator never shown a harmful instruction. "Personalization" is three epochs of LoRA fine-tuning unless noted. An episode is "near-still" if the policy output the no-op on more than half of its steps; "first-frame refusal" is the rate at which the predicted first action is the no-op. Counts are seed 0 unless noted.

| Piece | Headline result | Code |
|---|---|---|
| Harness | Closed-loop evaluator that runs identical initial states under harmful, benign and blank instructions; paired statistics with unit tests; go/no-go gates | `ftr/`, `tests/` on every branch; most complete on `posthoc` |
| Study 1, pre-registered, n=50 | A's harmful-instruction hand contact was 0 of 50 before personalization and 0 of 50 after: a null at a floor | `main` |
| Dose run, exploratory | A's harmful-instruction contact was 0, 5, 3 of 50 at 3, 6, 12 epochs of personalization; C's was 11, 9, 8. The 3-epoch values come from the Study 1 run | `dose-response` |
| Study 2, pre-registered, n=150 | A's benign-instruction contact rose from 0 to 40 of 150 (+0.27, 95% interval [0.20, 0.34]); seeds 1 and 2: 1 to 45 and 0 to 40. The manipulation check failed (task success 0 of 150 before and after), so by the pre-registered rule this is not read as erased safety | `study2` |
| Study 2b, pre-registered | The hand was moved beside the route. The comparator touched it in 6 of 50 and then 0 of 50 development episodes (two placements), against a required 20%. Stopped at that gate; no confirmatory run | `study2b` |
| Post hoc runs, exploratory | Personalized A was near-still in 148 of 150 episodes under the harmful wording, against 97 of 150 under a benign wording that also names the hand. The declared replication criterion for seeds 1 and 2 was not met | `posthoc` |

Where to check these numbers: all of them are reported in the draft paper and its supplement on the `paper` branch (`paper/main.tex`, `paper/supplement.tex`, `paper/supp_tables.tex`). Study 1's contact counts and refusal rates can also be checked against the run logs (`logs/run/`) and summary tables (`figures/`) on `main`.

**Per-episode results are not included in this repository** for any run. For the dose run, Study 2, Study 2b and the post hoc runs, the run logs are not included either; their results appear here only as reported in the draft paper and supplement, plus the small summary tables in `paper/figs/` on `paper`.

## Why it did not answer the question

1. **No compliance pathway.** P, never taught to refuse, touched the hand less often under the harmful instruction than under the benign one (27 vs 45 of 50). No policy, aligned or not, came detectably closer to the hand when told to harm it, so a low contact count under the harmful instruction says nothing about refusal.
2. **The taught safeguard was a non-selective freeze.** Before personalization, A's first-frame refusal rate was 0.89 under benign and 0.52 under blank instructions (1.00 under harmful). The two over-refusal screens that flag this were non-blocking; both failed and the run continued.
3. **The no-op metric could not tell refusal from inactivity.** After personalization C, never taught to refuse, scored 0.32 to 0.48 on first-frame "refusal" (0.00 before) and was near-still in 43 and 44 of 50 Study 1 episodes under the harmful and benign instructions (0 before).

## What is reusable

- The paired-state evaluator (`ftr/rollout.py`) and the statistics code with unit tests (`ftr/analyze.py`, `tests/test_analyze.py`).
- A repair for LIBERO-Safety's robot-contact check, which never fires as shipped (`patches/libero_safety_issue3.patch`).
- LIBERO-Object scenes with a human hand added (`scenes/`, on `study2`, `study2b` and `posthoc`).
- The pre-registration documents (`PLAN.md`, `PREREG_STUDY2.md`, `PREREG_STUDY2B.md`, `POSTHOC.md`).

## Controls proposed in the draft paper

From Table 2 of the draft. These are proposed, not validated.

- **Competence:** gate scenes on a never-aligned policy that completes the task and meets the hazard.
- **Compliance:** include a policy trained to comply, and show that the harness detects it.
- **Selectivity:** compare harmful and benign wordings of one form that name the same object, against a never-aligned comparator.
- **Durability:** report success, contact and stopping together, per instruction.

## Repo map

The branches were never merged. `main` is the Study 1 pre-registration commit (`prereg-v1.1`) followed by automated log commits from the Study 1 run, then this README update. `dose-response`, `study2` and `paper` each start from `prereg-v1.1`; `study2b` continues `study2`, and `posthoc` continues `study2b`.

| Branch | Holds |
|---|---|
| `main` | Study 1 code and plan (`PLAN.md`, `IMPLEMENTATION.md`); Study 1 run logs (`logs/run/`) and summary tables and figures (`figures/`, `figures_s0/`, `figures_gate/`, `figures_gate_a/`); pilot logs (`logs/phase1/`, `logs/phase2/`); the original proposal (`docs/`) |
| `dose-response` | The 12-epoch run script (`scripts/dose_response.sh`) and `PLAN.md` Amendment 2 |
| `study2` | Study 2 scenes (`scenes/object_hazard/`), analysis, driver (`scripts/run_study2.sh`) and `PREREG_STUDY2.md` |
| `study2b` | Everything on `study2`, plus the two alternative hand placements (`scenes/object_hazard_p1/`, `scenes/object_hazard_p2/`), `scripts/run_study2b.sh` and `PREREG_STUDY2B.md` |
| `posthoc` | Everything on `study2b`, plus `POSTHOC.md`, `scripts/run_posthoc.sh` and `scripts/analyze_posthoc.py` |
| `paper` | The draft paper, "When Stillness Looks Like Safety" (`paper/main.tex`, `paper/main.pdf`; four pages plus references), and its supplement (`paper/supplement.tex`, `paper/supplement.pdf`). The draft was never submitted to any venue. This branch has the Study 1 code only |

Pre-registration tags, with the commit date (2026, US Central) of the commit each points to: `prereg-v1` and `prereg-v1.1` (Study 1), 22 Sep; `prereg-v2` (Study 2), 26 Sep; `prereg-v3` (Study 2b), 27 Sep; `posthoc-v1`, `posthoc-v1.1` and `posthoc-v2` (post hoc runs), 28 Sep.

The commits behind `prereg-v1` and `prereg-v1.1` were on GitHub before Study 1 ran: that run's automated log commits on `main` descend directly from `prereg-v1.1`. The other five tags, and every branch except `main`, were first pushed on 13 October 2026 (US time; 14 October UTC), after their data had been collected. Their timing therefore rests on local commit dates and cannot be verified externally.

`README.md` and `STATUS.md` on the other branches are the 22 September versions and are out of date; the copies on `main` are current. The run instructions as frozen for Study 1 are in the README at tag `prereg-v1.1`.

## Status

13 October 2026: the project is closed, and its question is still open. No further work is planned in this repository. The final state of each study is in [`STATUS.md`](STATUS.md).

## How this was made

The code, plans and draft paper were written with heavy AI assistance. All 95 code and document commits made before this README update carry a `Co-Authored-By: Claude` trailer, as does the update itself. The other 416 commits are log uploads from the training machines, 414 of them periodic automated syncs.

Nothing in this repository has been peer reviewed. The "review" rounds mentioned in commit messages are not peer review.
