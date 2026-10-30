# Managerial calibration and frozen reuse

The selected tasks are director compensation (`HRM_1_1`), employee voice (`AMJ_1_3`), CEO trust source calibration (`JAP_4_2b`), and frozen transfer to restaurant-manager trust (`JAP_4_1_new`).

- `analyze.py` reconstructs expected item scores, reverse scoring, treatment contrasts, and human-effect errors from option probabilities.
- `run.py` prepares source calibration and frozen transfer using the execution engines and materials in a supplied full bundle.
- `collect.py` collects the frozen primary configuration and prompt baseline from completed runs.
- [Execution engines](../../experiments/managerial/code/) are included as source for inspection and reuse.

Extract [the full archive](../../artifacts/Full_Reproduction_v6.zip) under `artifacts/unpacked/full/`. From the repository root, inspect the execution plan:

```bash
python cookbook/executive/run.py plan --bundle artifacts/unpacked/full/submission_experiments/managerial
```

For calibration and transfer, follow [the reproduction guide](../../docs/REPRODUCIBILITY.md) and `RUNNING.md` inside the archive. Supply the bundle path explicitly when using this Cookbook launcher; its data, frozen assets, and engine paths belong to that bundle.

All four studies average model treatment effects across option orderings, calculate absolute differences from human effects, divide by the response scale span, and average equally across effects. Per-order sensitivity is retained as a diagnostic. The source tasks use known human effects for fitting and selection. Restaurant transfer freezes the CEO controller. Complete scoring conventions and records accompany the data.
