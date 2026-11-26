# External dataset integration status

The dataset videos remain on the Mac. Audit them with:

```bash
python -m signverse.audit_external data/external/isl_basic_5
python -m signverse.import_external data/external/isl_basic_5
```

Audit reports are saved in `data/audit/` and conversion reports in `data/processed_external/`. Only review-accepted, present, readable, checksum-matching, nonduplicate videos are processed. The converter extracts the first two seconds at 15 Hz. Videos that are too short or lack sufficient hand detections are rejected. Each processed file retains original metadata. Use the separate processed directory to exclude provisional webcam recordings.

The published signer field contains values such as `gesture`, `comparator`, and clip-specific INCLUDE filenames. These are not reliably person identities. Consequently, **do not claim signer-independent performance** from a split based on these fields. Verify identities using source datasets before any held-out signer evaluation. Dataset review status does not establish expert verification. Inspect the per-source license metadata and source dataset terms.

This integration is incomplete. The dashboard does not yet expose the external dataset and the training workflow still needs a defensible grouping strategy for unknown signers. Do not use the default session split on the processed data as evidence of generalization. The app uses the original five-class configuration including Stop. The external label configuration is `config.external.json`.
