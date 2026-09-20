"""Pod (needs TF + one RLDS shard): our Dataset produces the same tokens, labels and pixels as OpenVLA's own
RLDSBatchTransform + eval preprocessing. This is the evidence that the training objective is OpenVLA's."""

import os
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

pytestmark = pytest.mark.gpu

RLDS = Path(os.environ.get("FTR_RLDS_DIR", "/workspace/hf/rlds")) / "libero_spatial_no_noops" / "1.0.0"


@pytest.fixture(scope="module")
def sample():
    import tensorflow_datasets as tfds

    ds = tfds.builder_from_directory(str(RLDS)).as_dataset(split="train", shuffle_files=False)
    ep = next(iter(ds.take(1)))
    st = next(iter(ep["steps"]))
    return dict(image=st["observation"]["image"].numpy(), action=st["action"].numpy(),
                lang=st["language_instruction"].numpy().decode())


def test_tokens_and_labels_match_rlds_batch_transform(sample):
    from transformers import AutoProcessor

    from prismatic.models.backbones.llm.prompting import PurePromptBuilder
    from prismatic.vla.action_tokenizer import ActionTokenizer
    from prismatic.vla.datasets.datasets import RLDSBatchTransform

    from ftr.codec import Codec
    from ftr.data import build_example

    proc = AutoProcessor.from_pretrained(os.environ["FTR_P_DIR"], trust_remote_code=True)
    codec = Codec()
    # RLDSBatchTransform expects an ALREADY-normalized action (the RLDS pipeline normalized it); feed ours.
    a_rlds = sample["action"].astype(np.float64).copy()
    a_rlds[6] = 1.0 - float(np.clip(a_rlds[6], 0, 1))  # libero_dataset_transform gripper flip
    n = codec.normalize(a_rlds)
    ref = RLDSBatchTransform(ActionTokenizer(proc.tokenizer), proc.tokenizer, proc.image_processor.apply_transform, PurePromptBuilder)(
        {"dataset_name": "x", "action": np.asarray([n], dtype=np.float32),
         "observation": {"image_primary": np.asarray([sample["image"]])},
         "task": {"language_instruction": sample["lang"].encode()}})
    ours = build_example(proc.tokenizer, proc.image_processor.apply_transform, codec, Image.fromarray(sample["image"]), sample["lang"], a_rlds)
    assert ours["input_ids"].tolist() == ref["input_ids"].tolist()
    assert ours["labels"].tolist() == ref["labels"].tolist()


def test_pixels_match_eval_preprocessing():
    """Training images stored by build_data (envs.model_image) must equal what get_vla_action feeds the model."""
    import tensorflow as tf
    from transformers import AutoProcessor

    from experiments.robot.openvla_utils import crop_and_resize

    from ftr import envs

    proc = AutoProcessor.from_pretrained(os.environ["FTR_P_DIR"], trust_remote_code=True)
    rng = np.random.default_rng(0)
    obs = {"agentview_image": rng.integers(0, 255, (256, 256, 3), dtype=np.uint8)}
    ours = envs.model_image(obs, center_crop=True)
    # reference: the stock path, inlined from get_libero_image + get_vla_action(center_crop=True)
    from experiments.robot.libero.libero_utils import get_libero_image

    img = Image.fromarray(get_libero_image(obs, 224)).convert("RGB")
    t = tf.image.convert_image_dtype(tf.convert_to_tensor(np.array(img)), tf.float32)
    t = crop_and_resize(t, 0.9, 1)
    ref = tf.image.convert_image_dtype(tf.clip_by_value(t, 0, 1), tf.uint8, saturate=True).numpy()
    assert np.array_equal(ours, ref)
    pv_ours = proc.image_processor.apply_transform(Image.fromarray(ours))
    pv_ref = proc.image_processor.apply_transform(Image.fromarray(ref).convert("RGB"))
    assert torch.equal(pv_ours, pv_ref)
