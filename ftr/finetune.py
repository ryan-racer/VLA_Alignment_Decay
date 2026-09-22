"""
finetune.py — openvla @ c8f03f4 vla-scripts/finetune.py with five changes (see IMPLEMENTATION.md):

  1. RLDSDataset -> ftr.data.ParquetTransitions; epoch loop over a shuffled, seeded DataLoader
  2. seed everything; write args.json (with git SHA) into the run dir
  3. clip_grad_norm_(1.0) before optimizer.step()  (openvla #299 / #333)
  4. one save at the end: adapter -> merge -> save   (no per-save_steps merging)
  5. merged config.norm_stats = {libero_spatial}; dataset_statistics.json + the *_prismatic.py files copied next to it

Also: single GPU only (no DDP / dist.barrier). Everything else is the stock loop.

    python -m ftr.finetune --vla_path /workspace/hf/P --data_parquet data/A.parquet --run_root_dir runs --run_id A_s0 --epochs 3
"""

import json
import math
import os
import random
import shutil
import subprocess
import uuid
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import draccus
import numpy as np
import torch
import tqdm
from peft import LoraConfig, PeftModel, get_peft_model
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor
from transformers.modeling_outputs import CausalLMOutputWithPast

import wandb
from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
from prismatic.util.data_utils import PaddedCollatorForActionPrediction
from prismatic.vla.action_tokenizer import ActionTokenizer

from ftr.codec import UNNORM_KEY, Codec
from ftr.data import ParquetTransitions, ckpt_uid, stock_augment

os.environ["TOKENIZERS_PARALLELISM"] = "false"


@dataclass
class FinetuneConfig:
    # fmt: off
    """CLI flags (draccus). Effective batch = batch_size * grad_accumulation_steps."""
    vla_path: str = "/workspace/hf/P"                 # P, or a merged run dir for stage 2 (personalization)
    data_parquet: Path = Path("data/A.parquet")       # rows: image, instruction, action (see ftr/data.py)
    run_root_dir: Path = Path("runs")
    run_id: str = "A_s0"                              # runs/<run_id>/ gets the merged checkpoint
    adapter_tmp_dir: Path = Path("adapter-tmp")

    epochs: int = 3                                   # PLAN.md: three epochs over all retained transitions
    max_steps: Optional[int] = None                   # override for the tiny-overfit wiring test only
    batch_size: int = 8
    grad_accumulation_steps: int = 2                  # effective batch 16 (README: batch 16 ~ 72 GB on A100-80)
    learning_rate: float = 5e-4
    grad_clip: float = 1.0
    seed: int = 0
    num_workers: int = 0                              # TF augmentation in the main process (no TF in forked workers after
                                                      # CUDA init); costs ~15% wall time, not correctness
    image_aug: bool = True                            # stock finetune.py default: random crop + colour jitter (dlimp)

    lora_rank: int = 32
    lora_dropout: float = 0.0

    merge_only: bool = False                          # skip training: re-merge a saved adapter into vla_path (deterministic;
                                                      # used after a Colab runtime death took the local merged checkpoint)
    wandb_project: str = "ftr"
    wandb_entity: Optional[str] = None
    # fmt: on


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent, text=True).strip()
    except Exception:
        return "unknown"


def _seed(seed: int) -> torch.Generator:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    g = torch.Generator()
    g.manual_seed(seed)
    return g


@draccus.wrap()
def finetune(cfg: FinetuneConfig) -> None:
    """Load P (or a merged run), LoRA-wrap, train `epochs` over the Parquet, merge once, save with one norm_stats key."""
    assert torch.cuda.is_available(), "Fine-tuning assumes at least one GPU is available!"
    device_id = 0
    torch.cuda.set_device(device_id)
    g = _seed(cfg.seed)

    run_dir, adapter_dir = cfg.run_root_dir / cfg.run_id, cfg.adapter_tmp_dir / cfg.run_id
    os.makedirs(run_dir, exist_ok=True)
    base_uid = ckpt_uid(cfg.vla_path)
    if not cfg.merge_only:  # a re-merge must not overwrite the record of the training run
        (run_dir / "args.json").write_text(json.dumps({**asdict(cfg), "git_sha": _git_sha(), "base_uid": base_uid}, default=str, indent=2))

    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

    processor = AutoProcessor.from_pretrained(cfg.vla_path, trust_remote_code=True)
    if cfg.merge_only:
        assert (adapter_dir / "adapter_config.json").exists(), f"no adapter at {adapter_dir}"
        prov = json.loads((adapter_dir / "provenance.json").read_text())
        assert prov["base_uid"] == base_uid, f"adapter {cfg.run_id} was trained on {prov['base_uid']}, not {cfg.vla_path} ({base_uid})"
        stats = json.loads((adapter_dir / "dataset_statistics.json").read_text())
        _merge_and_save(cfg, processor, run_dir, adapter_dir, stats, prov)
        return
    vla = AutoModelForVision2Seq.from_pretrained(
        cfg.vla_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True
    ).to(device_id)

    lora_config = LoraConfig(
        r=cfg.lora_rank,
        lora_alpha=min(cfg.lora_rank, 16),
        lora_dropout=cfg.lora_dropout,
        target_modules="all-linear",
        init_lora_weights="gaussian",
    )
    vla = get_peft_model(vla, lora_config)
    vla.print_trainable_parameters()

    trainable_params = [p for p in vla.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=cfg.learning_rate)
    action_tokenizer = ActionTokenizer(processor.tokenizer)

    # --- change 1: our Dataset, frozen stats, epoch loop -------------------------------------
    codec = Codec()
    vla_dataset = ParquetTransitions(cfg.data_parquet, processor.tokenizer, processor.image_processor.apply_transform, codec,
                                     image_fn=stock_augment if cfg.image_aug else None)
    (run_dir / "dataset_statistics.json").write_text(json.dumps(vla_dataset.dataset_statistics, indent=2))
    collator = PaddedCollatorForActionPrediction(
        processor.tokenizer.model_max_length, processor.tokenizer.pad_token_id, padding_side="right"
    )
    dataloader = DataLoader(
        vla_dataset, batch_size=cfg.batch_size, shuffle=True, generator=g, collate_fn=collator,
        num_workers=cfg.num_workers, drop_last=False,
    )
    steps_per_epoch = math.ceil(len(dataloader) / cfg.grad_accumulation_steps)
    max_steps = cfg.max_steps if cfg.max_steps is not None else cfg.epochs * steps_per_epoch
    print(f"{len(vla_dataset)} rows, {len(dataloader)} micro-batches/epoch, {steps_per_epoch} updates/epoch, {max_steps} updates total")

    wandb.init(entity=cfg.wandb_entity, project=cfg.wandb_project, name=cfg.run_id, config=asdict(cfg))

    recent_losses = deque(maxlen=cfg.grad_accumulation_steps)
    recent_action_accuracies = deque(maxlen=cfg.grad_accumulation_steps)
    recent_l1_losses = deque(maxlen=cfg.grad_accumulation_steps)
    num_patches = vla.vision_backbone.featurizer.patch_embed.num_patches

    gradient_step_idx, micro_idx, done = 0, 0, False
    with tqdm.tqdm(total=max_steps, leave=False) as progress:
        vla.train()
        optimizer.zero_grad()
        for epoch in range(10**6):  # bounded by max_steps below
            for batch in dataloader:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    output: CausalLMOutputWithPast = vla(
                        input_ids=batch["input_ids"].to(device_id),
                        attention_mask=batch["attention_mask"].to(device_id),
                        pixel_values=batch["pixel_values"].to(torch.bfloat16).to(device_id),
                        labels=batch["labels"],
                    )
                    loss = output.loss
                (loss / cfg.grad_accumulation_steps).backward()

                action_logits = output.logits[:, num_patches:-1]
                action_preds = action_logits.argmax(dim=2)
                action_gt = batch["labels"][:, 1:].to(action_preds.device)
                mask = action_gt > action_tokenizer.action_token_begin_idx
                correct_preds = (action_preds == action_gt) & mask
                action_accuracy = correct_preds.sum().float() / mask.sum().float()
                continuous_actions_pred = torch.tensor(action_tokenizer.decode_token_ids_to_actions(action_preds[mask].cpu().numpy()))
                continuous_actions_gt = torch.tensor(action_tokenizer.decode_token_ids_to_actions(action_gt[mask].cpu().numpy()))
                action_l1_loss = torch.nn.functional.l1_loss(continuous_actions_pred, continuous_actions_gt)
                recent_losses.append(loss.item())
                recent_action_accuracies.append(action_accuracy.item())
                recent_l1_losses.append(action_l1_loss.item())

                micro_idx += 1
                if micro_idx % cfg.grad_accumulation_steps == 0:
                    # --- change 3 -----------------------------------------------------------
                    torch.nn.utils.clip_grad_norm_(trainable_params, cfg.grad_clip)
                    optimizer.step()
                    optimizer.zero_grad()
                    gradient_step_idx += 1
                    progress.update()
                    if gradient_step_idx % 10 == 0:
                        wandb.log(
                            {
                                "train_loss": sum(recent_losses) / len(recent_losses),
                                "action_accuracy": sum(recent_action_accuracies) / len(recent_action_accuracies),
                                "l1_loss": sum(recent_l1_losses) / len(recent_l1_losses),
                                "epoch": epoch,
                            },
                            step=gradient_step_idx,
                        )
                    if gradient_step_idx >= max_steps:
                        done = True
                        break
            if done:
                break

    # --- changes 4 + 5: save once -------------------------------------------------------------
    print(f"Saving adapter to {adapter_dir} and merged model to {run_dir}")
    # LoRA weights only (embeddings are not targets; without them the adapter is a few hundred MB and fits on Drive)
    vla.save_pretrained(adapter_dir, save_embedding_layers=False)
    (adapter_dir / "dataset_statistics.json").write_text(json.dumps(vla_dataset.dataset_statistics, indent=2))
    prov = dict(run_uid=uuid.uuid4().hex[:12], base_uid=base_uid, updates=gradient_step_idx, rows=len(vla_dataset), seed=cfg.seed)
    (adapter_dir / "provenance.json").write_text(json.dumps(prov, indent=2))
    _merge_and_save(cfg, processor, run_dir, adapter_dir, vla_dataset.dataset_statistics, prov)


def _merge_and_save(cfg, processor, run_dir: Path, adapter_dir: Path, stats: dict, prov: dict) -> None:
    """base(vla_path) + adapter -> merged checkpoint in run_dir with exactly one norm_stats key, the *_prismatic.py
    files beside it, and a DONE marker. Deterministic, so a lost merged checkpoint is rebuilt bit-identically."""
    processor.save_pretrained(run_dir)
    (run_dir / "dataset_statistics.json").write_text(json.dumps(stats, indent=2))
    base_vla = AutoModelForVision2Seq.from_pretrained(
        cfg.vla_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True
    )
    merged_vla = PeftModel.from_pretrained(base_vla, adapter_dir).merge_and_unload()
    merged_vla.config.norm_stats = stats  # exactly one key: libero_spatial
    merged_vla.save_pretrained(run_dir)
    src = Path(OpenVLAForActionPrediction.__module__.replace(".", "/")).parent  # prismatic/extern/hf
    root = next(p for p in map(Path, os.sys.path) if (Path(p or ".") / "prismatic").exists())
    for f in ("configuration_prismatic.py", "modeling_prismatic.py", "processing_prismatic.py"):
        shutil.copy(root / src / f, run_dir / f)
    # run_uid identifies the trained weights (a re-merge of the same adapter is bit-identical, so it keeps the uid)
    (run_dir / "DONE").write_text("".join(f"{k}={v}\n" for k, v in prov.items()) + f"key={UNNORM_KEY}\n")
    print(f"done: {prov['updates']} updates (run_uid {prov['run_uid']})")


if __name__ == "__main__":
    finetune()
