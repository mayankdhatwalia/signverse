import hashlib
import json
import os
import re
import uuid
from pathlib import Path

import numpy as np
import pandas as pd

from .core import FEATURE_VERSION, validate_sequence


def safe_name(value):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", value):
        raise ValueError("Participant and session IDs must use letters, numbers, hyphens or underscores.")
    return value


def save_sample(root, sequence, label, participant, session, config):
    if label not in config["labels"]:
        raise ValueError("Unknown gesture label.")
    sequence = validate_sequence(sequence, config["minimum_hand_frames"])
    participant, session = safe_name(participant), safe_name(session)
    # The label index avoids interpreting user-configured labels as paths.
    directory = Path(root) / f"class_{config['labels'].index(label):03d}" / participant / session
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{uuid.uuid4().hex}.npz"
    temporary = path.with_suffix(".tmp")
    metadata = dict(label=label, participant=participant, session=session,
                    feature_version=FEATURE_VERSION, sample_fps=config["sample_fps"])
    try:
        with temporary.open("xb") as file:
            np.savez_compressed(file, sequence=sequence, metadata=json.dumps(metadata))
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def load_dataset(root, config):
    sequences, rows, hashes = [], [], set()
    for path in sorted(Path(root).rglob("*.npz")):
        try:
            with np.load(path, allow_pickle=False) as item:
                sequence = validate_sequence(item["sequence"], config["minimum_hand_frames"])
                metadata = json.loads(str(item["metadata"].item()))
            if metadata["feature_version"] != FEATURE_VERSION or metadata["sample_fps"] != config["sample_fps"]:
                raise ValueError("Feature version or sample rate does not match configuration.")
            if metadata["label"] not in config["labels"]:
                raise ValueError("Label is absent from the configured vocabulary.")
            safe_name(metadata["participant"])
            safe_name(metadata["session"])
            digest = hashlib.sha256(sequence.tobytes()).hexdigest()
            if digest in hashes:
                raise ValueError("Duplicate sequence detected. Record independent samples.")
            hashes.add(digest)
            sequences.append(sequence)
            rows.append({**metadata, "path": str(path)})
        except (ValueError, KeyError, OSError) as error:
            raise ValueError(f"Invalid sample {path}: {error}") from error
    if not sequences:
        raise ValueError("No recordings found. Collect genuine gesture sequences first.")
    return np.stack(sequences), pd.DataFrame(rows)


def split_dataset(rows, labels, mode="session", seed=42):
    """Select disjoint participant/session groups with every class in all splits."""
    from sklearn.model_selection import GroupShuffleSplit

    groups = rows.participant if mode == "participant" else rows.participant + "/" + rows.session
    if groups.nunique() < 3:
        raise ValueError("Record at least three independent sessions (or participants for participant splitting).")
    required = set(labels)
    for trial in range(200):
        outer = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed + trial)
        trainval, test = next(outer.split(rows, groups=groups))
        inner = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed + trial)
        train, validation = next(inner.split(rows.iloc[trainval], groups=groups.iloc[trainval]))
        splits = (trainval[train], trainval[validation], test)
        if all(set(rows.iloc[index].label) == required for index in splits):
            return splits
    raise ValueError("Cannot split with all labels in each partition. Record every label in at least three independent groups.")
