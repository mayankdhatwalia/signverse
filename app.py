"""Streamlit dashboard. Launch with `streamlit run app.py`."""
import json
import threading
import time
from pathlib import Path

import av
import cv2
import pandas as pd
import streamlit as st
from streamlit_webrtc import VideoProcessorBase, webrtc_streamer

from signverse.capture import Recorder
from signverse.core import ROOT, load_config
from signverse.data import load_dataset, safe_name, save_sample
from signverse.model import Predictor
from signverse.vision import HandExtractor

st.set_page_config(page_title="SignVerse AI", page_icon="✋", layout="wide")
st.markdown("""<style>
.stApp {background: #0d1424; color: #edf2fa; font-family: Arial, sans-serif;}
.stApp h1, .stApp h2, .stApp h3, .stApp p, .stApp label, .stApp button {font-family: Arial, sans-serif;}
[data-testid="stSidebar"] {background: #141f33;}
h1, h2, h3 {letter-spacing: -.035em;}
[data-testid="stMetric"] {background: #19263b; border-radius: 12px; padding: 20px;}
.stButton>button {border-radius: 8px; border-color: #45ceb0;}
</style>""", unsafe_allow_html=True)
config = load_config()
dataset = ROOT / "data"
artifacts = ROOT / "artifacts"
st.title("SignVerse AI")
st.caption("Isolated gesture recognition · MediaPipe landmarks + Bidirectional LSTM")
if not config["isl_definitions_verified"]:
    st.warning("ISL sign definitions are awaiting verification. Hello, Thank You, Yes, No and Stop are provisional labels. Confirm each sign with an authoritative ISL source or qualified instructor before recording.")
page = st.sidebar.radio("Workspace", ["Dataset", "Collect", "Train & evaluate", "Recognize"])
st.sidebar.caption("30 frames · 15 samples/second · up to 2 hands")
models = sorted(artifacts.glob("run-*/model.pt"), key=lambda path: path.stat().st_mtime, reverse=True)


def dataset_summary():
    try:
        _, rows = load_dataset(dataset, config)
        return rows
    except ValueError as error:
        if list(dataset.rglob("*.npz")):
            st.error(str(error))
        return None


class CameraProcessor(VideoProcessorBase):
    def __init__(self, model_path=None):
        self.lock = threading.Lock()
        self.extractor = None
        self.predictor = Predictor(model_path, config["confidence_threshold"]) if model_path else None
        self.recorder = Recorder(config)
        self.destination = None
        self.message = "Ready"
        self.result = None

    def arm(self, label, participant, session):
        safe_name(participant)
        safe_name(session)
        with self.lock:
            if self.recorder.started is not None:
                raise ValueError("A recording is already in progress.")
            self.destination = (label, participant, session)
            self.recorder.arm(time.monotonic())

    def recv(self, frame):
        image = frame.to_ndarray(format="bgr24")
        with self.lock:
            try:
                if self.extractor is None:
                    self.extractor = HandExtractor()
                features = self.extractor.extract(image, draw=True)
                now = time.monotonic()
                if self.predictor:
                    result = self.predictor.update(features, now)
                    if result:
                        self.result = result
                        self.message = f"{result['label']}  {result['confidence']:.0%}"
                else:
                    sequence = self.recorder.update(features, now)
                    if sequence is not None:
                        save_sample(dataset, sequence, *self.destination, config)
                        self.recorder.message = "Saved. Ready for another recording."
                    self.message = self.recorder.message
            except (ValueError, RuntimeError, OSError) as error:
                self.recorder.started = None
                self.recorder.message = f"Capture error: {error}"
                self.message = self.recorder.message
            preview = cv2.flip(image, 1)
            cv2.rectangle(preview, (0, 0), (preview.shape[1], 40), (24, 29, 42), -1)
            cv2.putText(preview, self.message, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        return av.VideoFrame.from_ndarray(preview, format="bgr24")

    def on_ended(self):
        with self.lock:
            if self.extractor:
                self.extractor.close()
                self.extractor = None
            if self.predictor:
                self.predictor.reset()
            self.recorder.started = None


def camera(key, model_path=None):
    return webrtc_streamer(
        key=key, video_processor_factory=lambda: CameraProcessor(model_path),
        media_stream_constraints={"video": {"width": {"ideal": 640}, "height": {"ideal": 480}}, "audio": False},
        async_processing=False,
        rtc_configuration={"iceServers": []},
    )


if page == "Dataset":
    st.header("Build your gesture vocabulary")
    rows = dataset_summary()
    counts = rows.groupby("label").size() if rows is not None else pd.Series(dtype=int)
    table = pd.DataFrame({"Gesture": config["labels"]})
    table["Recorded"] = [int(counts.get(label, 0)) for label in config["labels"]]
    table["Target"] = config["target_samples_per_class"]
    left, middle, right = st.columns(3)
    left.metric("Gesture classes", len(config["labels"]))
    middle.metric("Genuine recordings", int(table.Recorded.sum()))
    right.metric("Trained models", len(models))
    st.dataframe(table, hide_index=True, use_container_width=True)
    st.info("Record 40–50 independent samples per class across at least three sessions. Include every class in each session. Participant-separated evaluation requires at least three participants.")
    if rows is not None:
        st.bar_chart(rows.groupby(["label", "session"]).size().unstack(fill_value=0))
    st.markdown("The model classifies a small vocabulary of isolated gestures. Hand landmarks alone omit facial expression and body cues used in ISL.")
elif page == "Collect":
    st.header("Record a gesture")
    st.write("Start the camera, select the label and press Record. A three-second countdown precedes each two-second sample. Perform one complete gesture while keeping both hands visible.")
    label = st.selectbox("Gesture", config["labels"])
    participant = st.text_input("Participant ID", "p01")
    session = st.text_input("Session ID", "session01")
    st.caption("Use a new session ID for a separate recording session. Keep participant IDs consistent.")
    context = camera("collection")
    if st.button("Record 30 frames", disabled=not context.state.playing):
        if context.video_processor:
            try:
                context.video_processor.arm(label, participant, session)
                st.success("Recording armed. Follow the countdown in the video.")
            except ValueError as error:
                st.error(str(error))
    st.caption("Camera access requires a local browser or HTTPS. For reliable Mac capture, use the OpenCV collection command in the README. Remote WebRTC connections may require deployment-specific TURN configuration.")
elif page == "Train & evaluate":
    st.header("Train a BiLSTM")
    st.write("Training uses genuine saved recordings. Validation selects the best epoch. The held-out test partition is evaluated once after selection.")
    split = st.selectbox("Holdout grouping", ["session", "participant"])
    epochs = st.number_input("Maximum epochs", min_value=1, max_value=300, value=80)
    if st.button("Train model"):
        from signverse.train import train
        try:
            with st.spinner("Training on CPU… See terminal for epoch progress."):
                run, metrics = train(dataset, artifacts, config, epochs=int(epochs), split=split)
            st.success(f"Saved {run.name}. Test accuracy {metrics['accuracy']:.1%}; macro F1 {metrics['macro_f1']:.3f}.")
            st.rerun()
        except (ValueError, RuntimeError, OSError) as error:
            st.error(str(error))
    if models:
        selected = Path(st.selectbox("Evaluation run", [str(path.parent) for path in models]))
        metrics = json.loads((selected / "metrics.json").read_text())
        first, second = st.columns(2)
        first.metric("Held-out accuracy", f"{metrics['accuracy']:.1%}")
        second.metric("Macro F1", f"{metrics['macro_f1']:.3f}")
        st.caption(f"Split by {metrics['split_mode']} · {metrics['train_samples']} train / {metrics['validation_samples']} validation / {metrics['test_samples']} test")
        st.image(str(selected / "confusion_matrix.png"))
        st.image(str(selected / "training_curves.png"))
        st.dataframe(pd.DataFrame(metrics["classification_report"]).T, use_container_width=True)
else:
    st.header("Live recognition")
    if not models:
        st.info("Collect labeled recordings and train a model to enable recognition.")
    else:
        selected = st.selectbox("Model checkpoint", [str(path) for path in models])
        st.caption("Perform one isolated gesture over approximately two seconds. Lower your hands between gestures to clear the sequence. Softmax confidence is uncalibrated and unrelated movements can still receive a high score.")
        try:
            camera(f"recognize-{selected}", selected)
        except (ValueError, RuntimeError, OSError) as error:
            st.error(str(error))
