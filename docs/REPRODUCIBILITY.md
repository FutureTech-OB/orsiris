# Reproduction guide

## CPU analysis

Download [Data_Recompute_v6.zip](../artifacts/Data_Recompute_v6.zip) and extract it to `artifacts/unpacked/data/`. From the repository root:

```bash
python artifacts/unpacked/data/submission_experiments/verify.py
python artifacts/unpacked/data/submission_experiments/recompute.py --output artifacts/recomputed/data
```

The analysis requires only Python 3.10+ and its standard library. It recomputes the selected experiments and compares the outputs with sealed targets. Expected values are used for verification, not as analysis inputs.

## Model execution

Download [Full_Reproduction_v6.zip](../artifacts/Full_Reproduction_v6.zip) and extract it to `artifacts/unpacked/full/`. Enter its `submission_experiments/` directory and follow `RUNNING.md`. It contains working commands for both social-choice models, passive readout, three managerial source calibrations, and frozen CEO-to-restaurant transfer.

Provide local model and large SAE/J-Lens checkpoints. Small frozen intervention assets are included. Install CUDA-compatible PyTorch and the specified model dependencies. New runs must use new output directories. For a newly calibrated source-to-transfer chain, supply the newly frozen CEO controller rather than mixing it with the historical controller.

## Public artifact preparation

The public archives retain experimental prompts, response records, target effects, selected feature tensors, and reported numerical results. Documentation and source comments are in English. Original Chinese labels in data and frozen protocols are retained as experimental metadata.

Machine-specific model/controller paths are replaced by generic paths. Local wheel URLs are replaced by version pins with their original wheel hashes. Integrity manifests are regenerated for these public copies. Original sealed archives are retained outside this repository. See `artifacts/V6_VALIDATION.json` for the v6 saved-data checks; `artifacts/PUBLICATION_CHECKS.json` retains the prior release audit.
