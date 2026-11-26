import numpy as np
import pytest

from signverse.import_external import select_window


def test_selects_full_window_after_initial_missing_hands():
    frames = np.zeros((60, 132), dtype=np.float32)
    frames[30:, -2] = 1
    window, start, count = select_window(frames, 24)
    assert start == 30
    assert count == 30
    assert window.shape == (30, 132)
    assert not window[:, -1].any()


def test_one_hand_and_missing_frames_are_accepted():
    frames = np.zeros((30, 132), dtype=np.float32)
    frames[:24, -1] = 1
    _, _, count = select_window(frames, 24)
    assert count == 24


def test_rejects_insufficient_hand_detection():
    frames = np.zeros((30, 132), dtype=np.float32)
    frames[:23, -2] = 1
    with pytest.raises(ValueError, match='At least 24'):
        select_window(frames, 24)


def test_distinguishes_short_and_nonfinite():
    with pytest.raises(ValueError, match='short video'):
        select_window(np.zeros((29, 132)), 24)
    frames = np.zeros((30, 132))
    frames[0, 0] = np.nan
    with pytest.raises(ValueError, match='non-finite'):
        select_window(frames, 24)
