from collections import deque
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .core import FEATURES, FEATURE_VERSION, validate_sequence


class BiLSTM(nn.Module):
    def __init__(self, classes, hidden_size=64):
        super().__init__()
        self.lstm = nn.LSTM(FEATURES, hidden_size, batch_first=True, bidirectional=True)
        self.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(hidden_size * 2, classes))

    def forward(self, sequence):
        _, (hidden, _) = self.lstm(sequence)
        return self.classifier(torch.cat((hidden[-2], hidden[-1]), dim=1))


def load_model(path):
    checkpoint = torch.load(Path(path), map_location="cpu", weights_only=True)
    if checkpoint.get("feature_version") != FEATURE_VERSION or checkpoint.get("sequence_length") != 30:
        raise ValueError("Incompatible model feature contract.")
    labels = checkpoint["labels"]
    if len(labels) < 2 or len(set(labels)) != len(labels):
        raise ValueError("Invalid model labels.")
    model = BiLSTM(len(labels), checkpoint["hidden_size"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint


class Predictor:
    def __init__(self, path, threshold=0.75):
        self.model, self.metadata = load_model(path)
        self.threshold = threshold
        self.frames = deque(maxlen=30)
        self.last_time = None

    def reset(self):
        self.frames.clear()
        self.last_time = None

    def update(self, features, timestamp):
        interval = 1 / self.metadata["sample_fps"]
        if self.last_time is not None:
            if timestamp <= self.last_time or timestamp - self.last_time > interval * 3:
                self.reset()
            elif timestamp - self.last_time < interval * 0.95:
                return None
        self.last_time = timestamp
        if not np.any(features[-2:]):
            self.frames.clear()
            return {"label": "No hands detected", "confidence": 0.0}
        self.frames.append(features)
        if len(self.frames) < 30:
            return {"label": f"Collecting frames {len(self.frames)}/30", "confidence": 0.0}
        sequence = validate_sequence(np.stack(self.frames))
        with torch.inference_mode():
            probabilities = self.model(torch.from_numpy(sequence).unsqueeze(0)).softmax(dim=1)[0]
        confidence, index = probabilities.max(dim=0)
        score = float(confidence)
        label = self.metadata["labels"][int(index)] if score >= self.threshold else "Uncertain"
        return {"label": label, "confidence": score,
                "probabilities": dict(zip(self.metadata["labels"], probabilities.tolist()))}
