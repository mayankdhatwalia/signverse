"""Convert audited clips to the webcam feature contract and report detection quality."""
import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from .core import FEATURE_VERSION, load_config, validate_sequence
from .vision import HandExtractor

LABELS = {'hello': 'Hello', 'thank you': 'Thank You', 'yes': 'Yes', 'no': 'No', 'help': 'Help'}


def select_window(features, minimum_hand_frames):
    """Choose the earliest complete 30-frame window with the most hand detections."""
    values = np.asarray(features, dtype=np.float32)
    if len(values) < 30:
        raise ValueError(f'short video: {len(values)} sampled frames, need 30')
    if values.ndim != 2 or values.shape[1] != 132:
        raise ValueError(f'invalid feature shape: {values.shape}')
    if not np.isfinite(values).all():
        raise ValueError('non-finite feature values')
    present = values[:, -2:].any(axis=1).astype(int)
    totals = np.convolve(present, np.ones(30, dtype=int), mode='valid')
    start = int(np.argmax(totals))
    best = values[start:start + 30]
    validate_sequence(best, minimum_hand_frames)
    return best, start, int(totals[start])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('root')
    parser.add_argument('--audit', default='data/audit/audit_records.csv')
    parser.add_argument('--output', required=True, help='New output directory; existing directories are never overwritten')
    args = parser.parse_args()
    root = Path(args.root).resolve()
    destination = Path(args.output)
    config = load_config()
    rows = pd.read_csv(args.audit, dtype=str, keep_default_na=False)
    destination.mkdir(parents=True, exist_ok=False)
    results = []
    for row in rows.to_dict('records'):
        if row['audit_status'] != 'accepted_readable':
            continue
        path = (root / row['video_path']).resolve()
        result = {'video_path': row['video_path'], 'label': row['audit_label'],
                  'status': 'processed', 'sampled_frames': 0, 'frames_with_hand': 0,
                  'frames_with_two_hands': 0, 'best_window_start_seconds': '',
                  'best_window_hand_frames': '', 'fps': '', 'width': '', 'height': ''}
        extractor = None
        cap = None
        try:
            if not path.is_relative_to(root) or not path.is_file():
                raise ValueError('missing or invalid path')
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest != row['actual_sha256']:
                raise ValueError('video changed since audit')
            cap = cv2.VideoCapture(str(path))
            fps = cap.get(cv2.CAP_PROP_FPS)
            result.update(fps=fps, width=cap.get(cv2.CAP_PROP_FRAME_WIDTH), height=cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if not np.isfinite(fps) or fps < config['sample_fps']:
                raise ValueError('invalid or insufficient video frame rate')
            extractor = HandExtractor()
            samples = []
            index = 0
            while True:
                ok, image = cap.read()
                if not ok:
                    break
                if index / fps + 1e-6 >= len(samples) / config['sample_fps']:
                    samples.append(extractor.extract(image))
                index += 1
            result['sampled_frames'] = len(samples)
            if samples:
                masks = np.asarray(samples)[:, -2:]
                result['frames_with_hand'] = int(masks.any(axis=1).sum())
                result['frames_with_two_hands'] = int(masks.all(axis=1).sum())
                if len(samples) >= 30:
                    totals = np.convolve(masks.any(axis=1).astype(int), np.ones(30, dtype=int), mode='valid')
                    result['best_window_hand_frames'] = int(totals.max())
                    result['best_window_start_seconds'] = int(totals.argmax()) / config['sample_fps']
            sequence, start, hand_frames = select_window(samples, config['minimum_hand_frames'])
            source = hashlib.sha256(row['dataset'].encode()).hexdigest()[:12]
            signer = hashlib.sha256((row['dataset'] + '\0' + row['signer']).encode()).hexdigest()[:16]
            metadata = dict(row, label=LABELS[row['audit_label']], participant=signer, session=source,
                            feature_version=FEATURE_VERSION, sample_fps=config['sample_fps'],
                            window_start_seconds=start / config['sample_fps'], window_hand_frames=hand_frames)
            with (destination / (digest + '.npz')).open('xb') as file:
                np.savez_compressed(file, sequence=sequence, metadata=json.dumps(metadata))
        except (ValueError, OSError, KeyError) as error:
            result['status'] = str(error)
        finally:
            if cap is not None:
                cap.release()
            if extractor is not None:
                extractor.close()
        results.append(result)
        print(row['video_path'], result['status'], flush=True)
    report = pd.DataFrame(results, columns=['video_path', 'label', 'status', 'sampled_frames', 'frames_with_hand',
                                           'frames_with_two_hands', 'best_window_start_seconds', 'best_window_hand_frames',
                                           'fps', 'width', 'height'])
    report.to_csv(destination / 'import_report.csv', index=False)
    if not report.empty:
        summary = report.groupby(['label', 'status'], dropna=False).size().rename('videos').reset_index()
        summary.to_csv(destination / 'class_diagnostics.csv', index=False)
        totals = report.groupby('label').agg(
            attempted=('status', 'size'),
            successful=('status', lambda statuses: int((statuses == 'processed').sum())),
            sampled_frames=('sampled_frames', 'sum'),
            frames_with_hand=('frames_with_hand', 'sum'),
            frames_with_two_hands=('frames_with_two_hands', 'sum'),
        )
        totals['hand_detection_rate'] = totals['frames_with_hand'] / totals['sampled_frames'].replace(0, np.nan)
        totals.to_csv(destination / 'class_detection_rates.csv')
        print(summary.to_string(index=False))
        print(totals.to_string())


if __name__ == '__main__':
    main()
