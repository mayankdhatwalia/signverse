import cv2
import mediapipe as mp
import numpy as np

from .core import normalize


class HandExtractor:
    """Accept unmirrored BGR images. Mirror only the displayed preview."""

    def __init__(self):
        self.detector = mp.solutions.hands.Hands(
            max_num_hands=2, model_complexity=1,
            min_detection_confidence=0.6, min_tracking_confidence=0.6,
        )

    def extract(self, frame, draw=False):
        result = self.detector.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        hands = [None, None]
        scores = [0.0, 0.0]
        for landmarks, handedness in zip(result.multi_hand_landmarks or [], result.multi_handedness or []):
            classification = handedness.classification[0]
            # MediaPipe labels assume mirrored input. Our input is unmirrored.
            slot = 0 if classification.label == "Right" else 1
            if classification.score > scores[slot]:
                hands[slot] = np.array([[p.x, p.y, p.z] for p in landmarks.landmark], dtype=np.float32)
                scores[slot] = classification.score
            if draw:
                mp.solutions.drawing_utils.draw_landmarks(frame, landmarks, mp.solutions.hands.HAND_CONNECTIONS)
        return normalize(hands)

    def close(self):
        self.detector.close()
