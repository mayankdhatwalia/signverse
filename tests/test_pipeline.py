"""Synthetic fixtures test software mechanics only, never gesture accuracy."""
import json

import numpy as np
import pandas as pd
import pytest
import torch

from signverse.capture import Recorder
from signverse.core import FEATURES, load_config, normalize, validate_sequence
from signverse.data import load_dataset, save_sample, split_dataset
from signverse.model import BiLSTM, Predictor, load_model
from signverse.train import train
from signverse.vision import HandExtractor


def sequence(seed=0):
    rng = np.random.default_rng(seed)
    frames = []
    hand = rng.uniform(0.2, 0.8, (21, 3)).astype(np.float32)
    for index in range(30):
        moving = hand.copy()
        moving[:, 0] += index / 300
        frames.append(normalize([moving, None]))
    return np.stack(frames)


def test_normalization_preserves_motion_and_masks():
    sample = sequence()
    assert sample.shape == (30, FEATURES)
    np.testing.assert_allclose(sample[0, :126], sample[-1, :126], atol=1e-5)
    assert sample[0, 126] != sample[-1, 126]
    assert np.all(sample[:, -2:] == [1, 0])
    assert not sample[:, 63:126].any()
    assert not normalize([None, None]).any()
    validate_sequence(sample)


@pytest.mark.parametrize("kind", ["shape", "nan", "missing", "mask", "ghost"])
def test_invalid_sequences(kind):
    sample = sequence()
    if kind == "shape":
        sample = sample[:29]
    elif kind == "nan":
        sample[0, 0] = np.nan
    elif kind == "missing":
        sample[:20] = 0
    elif kind == "mask":
        sample[0, -1] = 0.5
    else:
        sample[0, 64] = 1
    with pytest.raises(ValueError):
        validate_sequence(sample)


def test_round_trip_and_duplicate_rejection(tmp_path):
    config = load_config()
    path = save_sample(tmp_path, sequence(), "Hello", "p01", "s01", config)
    x, rows = load_dataset(tmp_path, config)
    np.testing.assert_array_equal(x[0], sequence())
    assert rows.iloc[0].label == "Hello"
    assert path.exists()
    save_sample(tmp_path, sequence(), "Hello", "p01", "s02", config)
    with pytest.raises(ValueError, match="Duplicate"):
        load_dataset(tmp_path, config)
    with pytest.raises(ValueError):
        save_sample(tmp_path, sequence(2), "Hello", "../escape", "s01", config)


def test_split_has_no_group_leakage():
    labels = ["Hello", "Stop"]
    rows = pd.DataFrame([dict(label=label, participant="p01", session=f"s{session}")
                         for session in range(5) for label in labels for _ in range(3)])
    splits = split_dataset(rows, labels)
    groups = [set(rows.iloc[index].session) for index in splits]
    assert not groups[0] & groups[1]
    assert not groups[0] & groups[2]
    assert not groups[1] & groups[2]
    assert sum(map(len, splits)) == len(rows)


def test_recorder_countdown_cadence_and_gap():
    recorder = Recorder(load_config())
    recorder.arm(0)
    assert recorder.update(sequence()[0], 1) is None
    result = None
    for index in range(30):
        result = recorder.update(sequence()[index], 3 + index / 15)
    assert result.shape == (30, FEATURES)
    assert recorder.started is None
    recorder.arm(10)
    recorder.update(sequence()[0], 13)
    assert recorder.update(sequence()[0], 14) is None
    assert recorder.started is None
    assert "too slow" in recorder.message


def test_real_mediapipe_blank_frame():
    extractor = HandExtractor()
    try:
        features = extractor.extract(np.zeros((480, 640, 3), dtype=np.uint8), draw=True)
        assert features.shape == (FEATURES,)
        assert not features.any()
    finally:
        extractor.close()


def test_training_checkpoint_and_inference(tmp_path):
    config = load_config()
    config["labels"] = ["fixture_a", "fixture_b"]
    for session in range(3):
        for index, label in enumerate(config["labels"]):
            for sample in range(2):
                save_sample(tmp_path / "data", sequence(session * 20 + index * 4 + sample),
                            label, "fixture_person", f"s{session}", config)
    run, metrics = train(tmp_path / "data", tmp_path / "artifacts", config, epochs=2)
    model, metadata = load_model(run / "model.pt")
    assert metrics["test_samples"] == 4
    assert len(metrics["history"]) == 2
    assert metadata["labels"] == config["labels"]
    assert (run / "confusion_matrix.png").exists()
    predictor = Predictor(run / "model.pt", threshold=0.01)
    result = None
    for index, frame in enumerate(sequence()):
        result = predictor.update(frame, index / 15)
    assert result["label"] in config["labels"]
    assert sum(result["probabilities"].values()) == pytest.approx(1)
    result = predictor.update(np.zeros(FEATURES, dtype=np.float32), 2)
    assert result["label"] == "No hands detected"
    assert len(predictor.frames) == 0
    predictor.update(sequence()[0], 3)
    predictor.update(sequence()[0], 5)
    assert len(predictor.frames) == 1
    manifest = pd.read_csv(run / "split_manifest.csv")
    assert manifest.groupby("session")["split"].nunique().max() == 1


def test_gradients_reach_both_lstm_directions():
    torch.set_num_threads(2)
    model = BiLSTM(2)
    before = model.lstm.weight_ih_l0.detach().clone()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss = torch.nn.functional.cross_entropy(model(torch.from_numpy(np.stack([sequence(1), sequence(2)]))), torch.tensor([0, 1]))
    loss.backward()
    assert model.lstm.weight_ih_l0_reverse.grad.abs().sum() > 0
    optimizer.step()
    assert not torch.equal(before, model.lstm.weight_ih_l0)
