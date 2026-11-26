# SignVerse AI

An academic **isolated gesture recognition** application using webcam frames, MediaPipe hand landmarks, and a genuine PyTorch bidirectional LSTM. The initial configurable vocabulary is **Hello, Thank You, Yes, No, Stop**.

**Current status:** collection, validation, training, evaluation, and live inference software are implemented. No genuine recordings or gesture-trained checkpoint are included. Recognition becomes available after you record and train. Recognition accuracy has yet to be measured on real signers. Synthetic test fixtures exercise software mechanics and are excluded from application datasets and models.

## ISL verification

The five labels are provisional. Their hand configurations and movements have **not been verified as Indian Sign Language signs**. Consult the [Indian Sign Language Research and Training Centre](https://islrtc.nic.in/) dictionary or a qualified ISL instructor. Record the selected dictionary entry, regional variant, demonstration reference, and reviewer for each label before capture. Keep the same verified variant across recordings. Set `isl_definitions_verified` in `config.json` only after this review. This flag records your review status and does not certify linguistic correctness.

The system recognizes a fixed vocabulary of isolated gestures. Full continuous translation, facial expression, body landmarks, grammar, and automatic sentence segmentation are outside this MVP.

## MacBook Air M4 setup

Use a native arm64 Terminal and Python **3.12** (3.11 is also permitted by the package). The primary tested interpreter is Python 3.12. Avoid a Rosetta Python installation. If Python 3.12 is missing, install it with your preferred Python manager, for example `brew install python@3.12` when Homebrew is available.

```bash
cd signverse
bash scripts/setup.sh
source .venv/bin/activate
python -m pytest -q
streamlit run app.py
```

The setup script creates `.venv`, installs pinned primary dependencies, and checks dependency consistency. MediaPipe 0.10.21 provides its hand models in its wheel and supports Python 3.12 via a macOS universal2 wheel. PyTorch 2.6.0 supplies an arm64 wheel. NumPy 1.26.4, OpenCV 4.11, JAX 0.4.35, and PyArrow 19 are selected to avoid incompatible NumPy 2 dependencies. Install only the declared OpenCV distribution to avoid conflicting `cv2` packages. Linux x86_64 setup installs the CPU-only PyTorch wheel to avoid CUDA downloads.

CPU training is the default for this small model. Optional Apple Metal acceleration is available through `--device mps`; it requires a working PyTorch MPS backend. CPU is the validated backend in the cloud. Actual M4 camera operation and MPS execution require validation on the Mac.

## Record the dataset

Grant camera access to Terminal/Python in macOS System Settings → Privacy & Security → Camera. Start with good, even lighting and a fixed camera at 640 × 480. Keep hands in frame. Perform each complete isolated gesture during the two-second recording period.

```bash
python -m signverse collect --label "Hello" --participant p01 --session session01
```

Press **Space** for a three-second countdown followed by 30 frames sampled at 15 Hz. Press **Q** to close. Capture pauses between recordings, so every accepted sample is an independent performance. Samples with fewer than 24 hand-visible frames are rejected. Slow capture with a gap above 200 ms is aborted. Reposition and repeat the gesture for each new sample.

Repeat the command for `Thank You`, `Yes`, `No`, and `Stop`. A practical collection plan is **5 distinct sessions × 10 independent recordings per class = 50 samples per class, 250 samples total**. Record every class in each session. Separate sessions by time and vary lighting or background moderately. Keep participant IDs consistent. For signer-independent evaluation, collect all labels from at least three different participants and use participant splitting. Obtain participant consent before recording.

The Streamlit **Collect** page also records through a browser webcam, with the same feature extraction and sample format. Click Start, allow camera access, then Record 30 frames. The default WebRTC configuration targets local use. A remote deployment may require HTTPS and deployment-specific STUN/TURN settings. The OpenCV command accesses the camera attached to the machine running Python. A cloud container cannot directly open your Mac camera.

Only landmark features and metadata are saved. Raw webcam video is not retained. Browser mode sends frames to the running Streamlit server for processing.

```text
data/
  class_000/
    p01/
      session01/
        <unique-id>.npz
```

Each sample contains a `float32` array of shape `(30, 132)` and JSON metadata with label, participant, session, feature version, and sampling rate. Files are saved atomically. Dataset paths are indexed by label and the true label is stored in metadata.

## Validate, train, evaluate

```bash
python -m signverse validate
python -m signverse train --epochs 80 --batch-size 16
# Optional, after collecting enough distinct signers:
python -m signverse train --split participant
# Optional Apple GPU:
python -m signverse train --device mps
```

Validation checks shapes, finite values, presence flags, missing-hand features, feature version, sample rate, labels, duplicate sequences, and whether grouped splits can contain every class. The recording target is 40–50 per class. Small datasets are allowed for debugging but cannot establish generalization.

Training selects a deterministic split of approximately 60% train / 20% validation / 20% test by entire sessions. Group counts can alter the exact proportions. No session occurs in multiple partitions. Participant splitting holds out complete signers. Every class must be present in all three partitions. The test set is evaluated after early stopping has selected the checkpoint using validation loss. Repeated tuning against test results invalidates the held-out interpretation; reserve fresh sessions for final academic evaluation.

Each run creates an independent `artifacts/run-*/` directory containing:

- `model.pt` with learned weights, labels, feature contract, and sampling rate.
- `metrics.json` with test accuracy, macro F1, per-class precision/recall/F1, and epoch losses.
- `split_manifest.csv` recording sample assignments for audit.
- `confusion_matrix.png` and `training_curves.png` for the academic report.

The dashboard's **Train & evaluate** page exposes CPU training and saved reports. Datasets and artifacts are Git-ignored. Retain your genuine recordings and selected model separately when transferring the project.

## Live prediction

Select a trained checkpoint in the dashboard's **Recognize** page or run:

```bash
python -m signverse predict --model artifacts/run-YOUR_RUN/model.pt
```

Use the exact run directory printed by training. The checkpoint supplies its own vocabulary and feature settings. Perform one isolated sign over approximately two seconds. Lower your hands between signs to reset the rolling buffer. No-hand frames and large capture gaps clear the buffer to prevent stale predictions. A full 30-frame window is required before inference. Below the configured confidence threshold, the UI displays `Uncertain`.

Softmax confidence is uncalibrated. The model can confidently misclassify unknown movements. The five-class MVP has no learned background/unknown class, so evaluate unfamiliar motions explicitly before any broader use. Live predictions are provisional during transitions.

## Architecture

```text
Unmirrored webcam BGR frame
  → MediaPipe Hands (up to two hands, 21 XYZ landmarks each)
  → stable anatomical left/right slots
  → common XY wrist origin and palm-size normalization
  → raw normalized-image wrist XY coordinates retain hand trajectories
  → missing-hand zeros and two presence flags
  → 30 × 132 feature window at 15 Hz
  → bidirectional LSTM (64 hidden units per direction)
  → dropout 0.3 + linear classification head
  → softmax scores and confidence threshold
```

Common normalization preserves relative positions between hands. MediaPipe depth is relative to each hand's wrist and cannot provide true inter-hand depth. Image coordinates use MediaPipe's axis conventions, so collection and inference should use the same camera aspect ratio. Left/right labels are corrected for unmirrored input; previews are mirrored only after extraction. Raw wrist coordinates retain motion that per-frame normalization would otherwise remove. Handedness swaps and occlusion can still degrade predictions.

Training uses weighted cross entropy, AdamW, gradient clipping, early stopping after 12 unimproved validation epochs, and a fixed seed. Checkpoints load with PyTorch `weights_only=True`. CPU inference keeps memory and startup costs modest. The BiLSTM operates on each complete rolling window, adding approximately two seconds of initial observation latency.

## Development and validation

```bash
python -m pytest -q
python -m pip check
```

Tests cover normalization and motion preservation, invalid and duplicate samples, group leakage, recording cadence, actual MediaPipe initialization and no-hand inference, gradients through both recurrent directions, training/checkpoint/inference integration, and all empty-state dashboard pages. Training tests use temporary synthetic fixtures solely to verify execution. The cloud run passed 13 tests and browser navigation through all four dashboard pages. Browser camera signaling succeeded, but media transport could not connect because the managed browser exposed no usable UDP ICE candidates. Browser video delivery remains unverified in this cloud environment. Hardware webcam, actual hand detection on a live signer, real gesture recognition accuracy, and MPS behavior require physical validation.

Primary modules are `core.py` (feature contract), `vision.py` (MediaPipe), `capture.py` (recording), `data.py` (validation and grouping), `model.py` (BiLSTM and prediction), `train.py` (learning and evaluation), and `app.py` (dashboard).

To expand the vocabulary, add labels in `config.json`, collect independent examples across sessions, and train a new checkpoint. Old checkpoints retain their own class mapping. Changing capture rate requires recollecting compatible recordings. Feature layout changes require a new feature version and retraining.
