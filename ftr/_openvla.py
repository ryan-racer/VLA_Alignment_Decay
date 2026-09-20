"""Load leaf modules from the openvla checkout by file path.

`import prismatic` executes prismatic/__init__.py, which pulls in the RLDS/TensorFlow stack. The
modules we need (action tokenizer, prompt builder, collator, HF image processor) have no
`prismatic` imports of their own, so we load them by path. Same code on the Mac (no TF) and the pod.
"""

from __future__ import annotations

import importlib.util
import sys
from functools import lru_cache
from pathlib import Path


def openvla_root() -> Path:
    """First sys.path entry that contains the openvla checkout."""
    for root in sys.path:
        p = Path(root or ".")
        if (p / "prismatic" / "vla" / "action_tokenizer.py").exists():
            return p
    raise ImportError("openvla checkout not on sys.path (need <openvla>/prismatic/vla/action_tokenizer.py)")


@lru_cache(maxsize=None)
def load(relpath: str):
    """Import `<openvla>/<relpath>` by file path, bypassing prismatic/__init__.py. Cached."""
    path = openvla_root() / relpath
    name = "ftr_openvla_" + relpath.replace("/", "_").removesuffix(".py")
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def action_tokenizer_cls():
    """prismatic.vla.action_tokenizer.ActionTokenizer, loaded by path."""
    return load("prismatic/vla/action_tokenizer.py").ActionTokenizer


def prompt_builder_cls():
    """PurePromptBuilder: wraps turns as `In: ...\nOut: ` / `...</s>`."""
    return load("prismatic/models/backbones/llm/prompting/base_prompter.py").PurePromptBuilder


def collator_cls():
    """PaddedCollatorForActionPrediction: right-pads ids, -100 on labels, builds attention_mask."""
    return load("prismatic/util/data_utils.py").PaddedCollatorForActionPrediction


def image_processor_cls():
    """PrismaticImageProcessor: resize-naive 224 -> tensor -> normalize, two stacks (DINO+SigLIP)."""
    return load("prismatic/extern/hf/processing_prismatic.py").PrismaticImageProcessor
