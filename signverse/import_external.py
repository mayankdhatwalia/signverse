"""Convert audited clips to the existing webcam feature contract."""
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('root')
    parser.add_argument('--audit', default='data/audit/audit_records.csv')
    parser.add_argument('--output', default='data/processed_external')
    args = parser.parse_args()
    root = Path(args.root).resolve()
    destination = Path(args.output)
    destination.mkdir(parents=True, exist_ok=True)
    config = load_config()
    rows = pd.read_csv(args.audit, dtype=str, keep_default_na=False)
    results = []
    for row in rows.to_dict('records'):
        if row['audit_status'] != 'accepted_readable':
            continue
        path = (root / row['video_path']).resolve()
        status = 'processed'
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
            if not np.isfinite(fps) or fps <= 0:
                raise ValueError('invalid video frame rate')
            extractor = HandExtractor()
            samples = []
            index = 0
            # Use the first two seconds at the same 15 Hz as webcam capture.
            while len(samples) < 30:
                ok, image = cap.read()
                if not ok:
                    break
                if index / fps + 1e-6 >= len(samples) / config['sample_fps']:
                    samples.append(extractor.extract(image))
                index += 1
            sequence = validate_sequence(np.asarray(samples, dtype=np.float32), config['minimum_hand_frames'])
            source = hashlib.sha256(row['dataset'].encode()).hexdigest()[:12]
            signer = hashlib.sha256((row['dataset'] + '\0' + row['signer']).encode()).hexdigest()[:16]
            metadata = dict(row, label=LABELS[row['audit_label']], participant=signer, session=source,
                            feature_version=FEATURE_VERSION, sample_fps=config['sample_fps'])
            target = destination / (digest + '.npz')
            with target.open('xb') as file:
                np.savez_compressed(file, sequence=sequence, metadata=json.dumps(metadata))
        except (ValueError, OSError, KeyError) as error:
            status = str(error)
        finally:
            if cap is not None:
                cap.release()
            if extractor is not None:
                extractor.close()
        results.append({'video_path': row['video_path'], 'label': row['audit_label'], 'status': status})
        print(row['video_path'], status, flush=True)
    pd.DataFrame(results).to_csv(destination / 'import_report.csv', index=False)


if __name__ == '__main__':
    main()
