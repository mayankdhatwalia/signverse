"""Shared feature contract used by collection, training and inference."""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FEATURE_VERSION = "two_hands_wrist_trajectory_v1"
FEATURES = 132  # 126 normalized XYZ, 4 wrist XY coordinates, 2 presence flags


def load_config(path=None):
    config = json.loads(Path(path or ROOT / "config.json").read_text())
    labels = config["labels"]
    if len(labels) < 2 or len(set(labels)) != len(labels):
        raise ValueError("Provide at least two unique labels.")
    if any(not isinstance(label, str) or not label.strip() for label in labels):
        raise ValueError("Labels must be nonempty strings.")
    if config["sequence_length"] != 30:
        raise ValueError("This feature contract requires 30-frame sequences.")
    if not 1 <= config["minimum_hand_frames"] <= 30 or config["sample_fps"] <= 0:
        raise ValueError("Invalid capture settings.")
    if not 0 < config["confidence_threshold"] <= 1:
        raise ValueError("Confidence threshold must be in (0, 1].")
    return config


def normalize(hands):
    """Normalize common XY origin and palm scale, preserving two-hand positions.

    Input is (left, right), each a (21, 3) array or None. MediaPipe Z is
    already relative to each wrist. Missing hands use zeros plus a mask.
    """
    output = np.zeros(FEATURES, dtype=np.float32)
    present = [np.asarray(hand, dtype=np.float32) for hand in hands if hand is not None]
    if not present:
        return output
    if any(hand.shape != (21, 3) or not np.isfinite(hand).all() for hand in present):
        raise ValueError("Each hand must contain 21 finite XYZ landmarks.")
    origin = np.mean([hand[0, :2] for hand in present], axis=0)
    scale = max(float(np.mean([np.linalg.norm(hand[9, :2] - hand[0, :2]) for hand in present])), 1e-6)
    for index, hand in enumerate(hands):
        if hand is not None:
            points = np.asarray(hand, dtype=np.float32).copy()
            points[:, :2] -= origin
            points[:, 2] -= points[0, 2]
            output[index * 63:(index + 1) * 63] = (points / scale).flatten()
            output[126 + index * 2:128 + index * 2] = hand[0, :2]
            output[130 + index] = 1
    return output


def validate_sequence(sequence, minimum_hand_frames=24):
    if sequence.shape != (30, FEATURES) or not np.isfinite(sequence).all():
        raise ValueError(f"Expected finite (30, {FEATURES}) features.")
    masks = sequence[:, -2:]
    if not np.isin(masks, [0, 1]).all():
        raise ValueError("Presence flags must be binary.")
    for index in range(2):
        if np.any(sequence[masks[:, index] == 0, index * 63:(index + 1) * 63] != 0):
            raise ValueError("Missing hands must have zero landmark features.")
        if np.any(sequence[masks[:, index] == 0, 126 + index * 2:128 + index * 2] != 0):
            raise ValueError("Missing hands must have zero wrist coordinates.")
    if np.count_nonzero(masks.any(axis=1)) < minimum_hand_frames:
        raise ValueError(f"At least {minimum_hand_frames} frames must contain a hand.")
    return np.asarray(sequence, dtype=np.float32)
