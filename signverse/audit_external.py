"""Audit the published ISL dataset metadata against locally available videos."""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import cv2
import pandas as pd

LABELS = ('hello', 'thank you', 'yes', 'no', 'help')


def audit(root, output):
    root = Path(root).resolve()
    frame = pd.read_csv(root / 'metadata.csv', dtype=str, keep_default_na=False)
    required = {'normalized_word', 'review_status', 'video_path', 'signer', 'dataset', 'license', 'sha256', 'original_label'}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f'Metadata is missing columns: {sorted(missing)}')
    rows = []
    seen = {}
    for record in frame.to_dict('records'):
        label = ' '.join(record['normalized_word'].casefold().split())
        if label not in LABELS:
            continue
        result = dict(record)
        result['audit_label'] = label
        result['audit_status'] = 'rejected_review'
        if record['review_status'].strip().casefold() == 'accepted':
            path = (root / record['video_path']).resolve()
            if not path.is_relative_to(root):
                result['audit_status'] = 'invalid_path'
            elif not path.is_file():
                result['audit_status'] = 'missing_video'
            else:
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                result['actual_sha256'] = digest
                if record['sha256'] and record['sha256'].casefold() != digest:
                    result['audit_status'] = 'checksum_mismatch'
                elif digest in seen:
                    result['audit_status'] = 'duplicate'
                    result['duplicate_of'] = seen[digest]
                else:
                    seen[digest] = record['video_path']
                    video = cv2.VideoCapture(str(path))
                    try:
                        ok, image = video.read()
                        result['audit_status'] = 'accepted_readable' if ok and image is not None else 'unreadable'
                    finally:
                        video.release()
        rows.append(result)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output / 'audit_records.csv', index=False)
    counts = {label: dict(Counter(row['audit_status'] for row in rows if row['audit_label'] == label)) for label in LABELS}
    summary = {'metadata_columns': list(frame.columns), 'counts': counts,
               'caution': 'Review status is not expert verification. Signer fields may not identify people; verify identities before claiming signer-independent evaluation.'}
    (output / 'audit_summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('root')
    parser.add_argument('--output', default='data/audit')
    args = parser.parse_args()
    audit(args.root, args.output)


if __name__ == '__main__':
    main()
