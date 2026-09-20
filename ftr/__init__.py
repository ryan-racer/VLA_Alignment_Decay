"""Forgetting to Refuse: safeguard persistence under benign personalization of OpenVLA.

Flat package, used via PYTHONPATH, never installed. Stages, in run order:

  codec.py       actions <-> tokens under P's frozen stats; the refusal label and its scorers   (Mac + pod)
  data.py        Parquet rows -> OpenVLA training examples (stock prompt/label contract)         (Mac + pod)
  envs.py        LIBERO / LIBERO-Safety: deterministic restore, per-constraint costs, fixtures  (pod)
  rollout.py     closed-loop episodes -> episodes.parquet / steps.parquet                        (pod)
  build_data.py  render states, pairs for scoring, RLDS export, per-arm training mixes          (pod)
  finetune.py    stock OpenVLA LoRA loop + five edits, single GPU                                (pod)
  score.py       offline generated-action predictions on pairs.parquet                          (pod)
  analyze.py     rates, paired deltas, figures from Parquet                                     (Mac + pod)
  _openvla.py    loads openvla leaf modules by path (import prismatic needs TensorFlow)

See PLAN.md (what/when) and IMPLEMENTATION.md (how).
"""
