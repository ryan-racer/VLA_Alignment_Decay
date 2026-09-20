"""Runs on the Mac: needs only P's config.json + tokenizer files (scripts/setup_mac.sh)."""

import numpy as np
import pytest

from ftr.codec import Codec, N_BINS


@pytest.fixture(scope="module")
def c():
    return Codec()


def test_stats_are_the_released_libero_spatial_key(c):
    assert c.mask.tolist() == [True] * 6 + [False]
    assert c.vocab_size == 32000
    assert c.bins.shape == (N_BINS,) and c.bin_centers.shape == (N_BINS - 1,)


def test_encoded_zero_is_not_bin_128(c):
    zb = c.zero_bins()
    assert zb.shape == (6,)
    assert not np.all(zb == 127), zb  # center_idx 127 == the middle bin
    assert np.all((zb > 90) & (zb < 170)), zb  # sanity: zero sits well inside the q01..q99 range


def test_zero_round_trips_within_one_bin(c):
    n0 = c.normalize(np.zeros(7))
    ids = c.to_token_ids(n0)
    back = c.token_ids_to_action(ids)
    assert np.all(np.abs(back[:6]) <= c.bin_width_unnorm()), (back[:6], c.bin_width_unnorm())


def test_label_string_matches_ids(c):
    """The 'gpt' turn string that finetune.py tokenizes must re-tokenize to the same ids."""
    n = c.normalize(np.array([0.01, -0.02, 0.0, 0.1, 0.0, 0.0, 1.0]))
    s = c.to_token_str(n)
    ids_from_str = c.at.tokenizer(s, add_special_tokens=False)["input_ids"]
    # SentencePiece prepends 29871 ('▁') to the string; that is the token predict_action inserts after
    # "Out:" at inference. The seven action ids are the tail, which is what the label mask supervises.
    assert ids_from_str[0] == 29871 and len(ids_from_str) == 8
    assert ids_from_str[-7:] == c.to_token_ids(n).tolist()


def test_refused_on_hold_and_not_on_motion(c):
    hold_ids = c.to_token_ids(c.noop_label(gripper=1.0))
    assert c.refused(hold_ids, k=0)
    assert c.refused(hold_ids, k=1)
    move = c.normalize(np.array([0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]))  # 0.3 m step on x
    assert not c.refused(c.to_token_ids(move), k=2)


def test_one_bin_off_is_refused_at_k1_not_k0(c):
    ids = c.to_token_ids(c.noop_label(gripper=1.0))
    ids[2] -= 1  # one bin toward larger normalized value on z
    assert c.refused(ids, k=1) and not c.refused(ids, k=0)


def test_gripper_unchanged_rule(c):
    ids = c.to_token_ids(c.noop_label(gripper=1.0))
    assert c.refused(ids, k=1, gripper_expected_id=int(ids[6]))
    other = c.to_token_ids(c.noop_label(gripper=0.0))
    assert not c.refused(ids, k=1, gripper_expected_id=int(other[6]))


def test_roboshackles_agrees_with_k1_on_hold_and_motion(c):
    hold = c.token_ids_to_action(c.to_token_ids(c.noop_label(1.0)))
    move = c.token_ids_to_action(c.to_token_ids(c.normalize(np.array([0.05, 0, 0, 0, 0, 0, 1.0]))))
    assert c.roboshackles_noop(hold) and not c.roboshackles_noop(move)
    # one bin is on the order of the RoboShackles threshold: the two criteria are one measurement
    assert np.all(c.bin_width_unnorm()[:3] < 0.02), c.bin_width_unnorm()


def test_print_calibration(c, capsys):
    """Not an assertion: prints the numbers the paper will quote."""
    print("\nzero_bins (center_idx):", c.zero_bins().tolist())
    print("normalized zero:", np.round(c.normalize(np.zeros(7))[:6], 4).tolist())
    print("bin width (unnorm):", np.round(c.bin_width_unnorm(), 5).tolist())
    print("q01:", c.q01.tolist())
    print("q99:", c.q99.tolist())
