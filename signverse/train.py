"""Deterministic training with grouped holdouts and portable CPU checkpoints."""
import copy
import json
import os
import random
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report, f1_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .core import FEATURE_VERSION
from .data import load_dataset, split_dataset
from .model import BiLSTM


def train(root, output, config, epochs=80, batch_size=16, device="cpu", split="session", seed=42):
    if epochs < 1 or batch_size < 1:
        raise ValueError("Epochs and batch size must be positive.")
    if device == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS is unavailable. Use --device cpu.")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    features, rows = load_dataset(root, config)
    train_indices, validation_indices, test_indices = split_dataset(rows, config["labels"], split, seed)
    y = np.array([config["labels"].index(label) for label in rows.label])
    x = torch.from_numpy(features)
    y = torch.from_numpy(y).long()
    loader = DataLoader(TensorDataset(x[train_indices], y[train_indices]), batch_size=batch_size, shuffle=True)
    model = BiLSTM(len(config["labels"])).to(device)
    counts = torch.bincount(y[train_indices], minlength=len(config["labels"]))
    weights = counts.sum() / (len(counts) * counts.float())
    criterion = nn.CrossEntropyLoss(weight=weights.to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    best_loss, patience, best_state = float("inf"), 0, None
    history = []
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for batch_x, batch_y in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(batch_x.to(device)), batch_y.to(device))
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss. Inspect landmark recordings.")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += float(loss.detach()) * len(batch_x)
        model.eval()
        with torch.inference_mode():
            validation_loss = float(criterion(model(x[validation_indices].to(device)), y[validation_indices].to(device)))
        history.append(dict(epoch=epoch + 1, train_loss=total_loss / len(train_indices), validation_loss=validation_loss))
        print(f"Epoch {epoch + 1}: train={history[-1]['train_loss']:.4f}, validation={validation_loss:.4f}", flush=True)
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_state = copy.deepcopy(model.cpu().state_dict())
            model.to(device)
            patience = 0
        else:
            patience += 1
        if patience >= 12:
            break
    if best_state is None:
        raise ValueError("Training did not produce a finite validation loss.")
    model.cpu().load_state_dict(best_state)
    model.eval()
    with torch.inference_mode():
        predictions = model(x[test_indices]).argmax(dim=1).numpy()
    truth = y[test_indices].numpy()
    metrics = dict(
        accuracy=float(accuracy_score(truth, predictions)),
        macro_f1=float(f1_score(truth, predictions, average="macro", zero_division=0)),
        classification_report=classification_report(truth, predictions, labels=list(range(len(config["labels"]))),
                                                    target_names=config["labels"], output_dict=True, zero_division=0),
        split_mode=split, seed=seed, history=history,
        train_samples=len(train_indices), validation_samples=len(validation_indices), test_samples=len(test_indices),
    )
    # Each run has its own directory, so model and metrics are always paired.
    Path(output).mkdir(parents=True, exist_ok=True)
    run = Path(tempfile.mkdtemp(prefix="run-", dir=output))
    torch.save(dict(state_dict=best_state, labels=config["labels"], hidden_size=64,
                    feature_version=FEATURE_VERSION, sequence_length=30, sample_fps=config["sample_fps"],
                    isl_definitions_verified=config["isl_definitions_verified"]), run / "model.pt")
    (run / "metrics.json").write_text(json.dumps(metrics, indent=2))
    rows = rows.copy()
    for name, indices in zip(["train", "validation", "test"], [train_indices, validation_indices, test_indices]):
        rows.loc[indices, "split"] = name
    rows.to_csv(run / "split_manifest.csv", index=False)
    figure, axis = plt.subplots(figsize=(8, 6))
    ConfusionMatrixDisplay.from_predictions(truth, predictions, labels=list(range(len(config["labels"]))),
                                            display_labels=config["labels"], ax=axis, xticks_rotation=45)
    figure.tight_layout()
    figure.savefig(run / "confusion_matrix.png", dpi=150)
    plt.close(figure)
    figure, axis = plt.subplots()
    axis.plot([row["train_loss"] for row in history], label="Train")
    axis.plot([row["validation_loss"] for row in history], label="Validation")
    axis.set(xlabel="Epoch (zero indexed)", ylabel="Cross-entropy loss")
    axis.legend()
    figure.tight_layout()
    figure.savefig(run / "training_curves.png", dpi=150)
    plt.close(figure)
    return run, metrics
