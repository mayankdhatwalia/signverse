"""Local OpenCV capture shared by the command-line collection and prediction tools."""
import time

import cv2
import numpy as np

from .data import save_sample
from .model import Predictor
from .vision import HandExtractor


class Recorder:
    def __init__(self, config):
        self.config = config
        self.frames = []
        self.started = None
        self.last_time = None
        self.message = "Press Space to record. Q quits."

    def arm(self, timestamp):
        self.frames = []
        self.last_time = None
        self.started = timestamp + 3
        self.message = "Get ready"

    def update(self, features, timestamp):
        if self.started is None:
            return None
        if timestamp < self.started:
            self.message = f"Start in {int(self.started - timestamp) + 1}"
            return None
        interval = 1 / self.config["sample_fps"]
        if self.last_time is not None:
            if timestamp - self.last_time > 3 * interval:
                self.started = None
                self.message = "Capture too slow. Retry with better lighting or lower resolution."
                return None
            if timestamp - self.last_time < interval * 0.95:
                return None
        self.last_time = timestamp
        self.frames.append(features.copy())
        self.message = f"Recording {len(self.frames)}/30"
        if len(self.frames) == 30:
            self.started = None
            return np.stack(self.frames)
        return None


def webcam(config, camera=0, dataset=None, label=None, participant=None, session=None, model=None):
    capture = cv2.VideoCapture(camera)
    if not capture.isOpened():
        capture.release()
        raise RuntimeError("Cannot open webcam. Run locally on the Mac and grant camera permission to Terminal.")
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    extractor = None
    try:
        extractor = HandExtractor()
        predictor = Predictor(model, config["confidence_threshold"]) if model else None
        recorder = Recorder(config)
        message = "Waiting for hands" if model else recorder.message
        while True:
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError("Camera stopped providing frames.")
            timestamp = time.monotonic()
            features = extractor.extract(frame, draw=True)
            if predictor:
                result = predictor.update(features, timestamp)
                if result:
                    message = f"{result['label']}  {result['confidence']:.0%}"
            else:
                sequence = recorder.update(features, timestamp)
                if sequence is not None:
                    try:
                        path = save_sample(dataset, sequence, label, participant, session, config)
                        recorder.message = f"Saved {label}. Space records again."
                        print(path, flush=True)
                    except ValueError as error:
                        recorder.message = f"Rejected: {error}"
                message = recorder.message
            preview = cv2.flip(frame, 1)
            cv2.rectangle(preview, (0, 0), (640, 65), (24, 29, 42), -1)
            cv2.putText(preview, message, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(preview, "Q: quit" if model else f"{label} | Space: record | Q: quit", (12, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 230, 210), 1)
            cv2.imshow("SignVerse AI", preview)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or cv2.getWindowProperty("SignVerse AI", cv2.WND_PROP_VISIBLE) < 1:
                break
            if not predictor and key == 32 and recorder.started is None:
                recorder.arm(time.monotonic())
    finally:
        capture.release()
        if extractor:
            extractor.close()
        cv2.destroyAllWindows()
