# OSIRIS

**Organizational and Social Inquiry through Representation Inspection and Steering**

OSIRIS is a Python research toolkit for reading and intervening on language-model representations in organizational and social decision experiments. It combines J-Lens concept directions, sparse autoencoder (SAE) features, and frozen controllers with reproducible behavioral analysis.

## Start here

| Goal | Entry point |
| --- | --- |
| Recompute the reported results without a GPU | [Data and analysis archive](artifacts/Data_Recompute_v6.zip) |
| Run the original model experiments | [Full reproduction archive](artifacts/Full_Reproduction_v6.zip) and [reproduction guide](docs/REPRODUCIBILITY.md) |
| Use the reusable API or define a new experiment | [Quickstart Cookbook](cookbook/quickstart/README.md) and [API reference](docs/API.md) |
| Inspect social-choice experiments | [Social-choice Cookbook](cookbook/social_choices/README.md) |
| Inspect managerial calibration and transfer | [Managerial Cookbook](cookbook/executive/README.md) and [execution code](experiments/managerial/code/) |
| Inspect decision-time readout | [Readout experiment](experiments/readout/README.md) |

## Install and try the API

Use Python 3.10 or later. Install a PyTorch build appropriate for your CPU or CUDA environment, then:

```bash
python -m pip install -e .
python -m repsteer demo --output runs/cpu-demo
python cookbook/quickstart/python_api.py
```

The demo uses randomly initialized tiny models and requires no checkpoint download. Its outputs are software examples, not research findings. Use a new output directory when rerunning.

For Hugging Face model execution:

```bash
python -m pip install -e '.[hf]'
python -m repsteer validate cookbook/quickstart/qwen27_jspace_organization.json --schema-only
```

Prepare compatible model and representation assets before running a real-model configuration. Models are loaded locally. See [API conventions](docs/API.md) for architecture and numerical details.

## Included research examples

- **Social choices:** Qwen3.5-9B and 27B; unfair ablation and unfair-to-fair replacement in a 90:10 ultimatum game; calm and anger injection after a manager takes credit for an employee's work. The supplied data contain 60,000 generated responses.
- **Managerial calibration and reuse:** director compensation, employee voice, CEO trust source calibration, and frozen transfer to restaurant-manager trust; 606 option-probability records across 20 benchmark contrasts.
- **Passive readout:** four model-by-scenario forward passes, with 32 emotion/feeling and six appraisal candidates per pass, recorded at decision time.

These are distinct observation types. Sampling counts, error aggregation, controller selection, doses, and environment requirements are documented with the experiments.

## Repository layout

```text
src/repsteer/       Reusable representations, interventions, backends, runner and metrics
cookbook/          API examples and research recipes
experiments/       Managerial execution engines and decision-time readout
tests/             CPU unit and local Hugging Face integration tests
tools/             Validation and companion-data utilities
docs/              API, reproduction and structure documentation
artifacts/         Separate downloadable data and full reproduction archives
```

The GitHub repository slug is `orsiris`; the project name and command are **OSIRIS**. The Python module remains `repsteer` and the distribution name is `osiris-research`.

## Validate

```bash
python tools/verify.py
```

This runs software tests without downloading model checkpoints. The companion archives also provide independent integrity checks and CPU result recomputation. See [validation scope](docs/VALIDATION.md).

This repository contains the research toolkit and reproduction code. Proposal drafts, author details, development history, unused explorations, model weights, and credentials are excluded.
