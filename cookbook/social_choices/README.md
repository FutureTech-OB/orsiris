# Four social-choice interventions in Qwen3.5-9B and 27B

| Scenario | Condition name | Operation | Target measure |
| --- | --- | --- | --- |
| Ultimatum game, 90:10 split | `unfair` | Ablate the unfair direction | Acceptance rate |
| Ultimatum game, 90:10 split | `fair` | Replace the unfair projection with the fair direction | Acceptance rate |
| Organizational injustice, manager takes credit | `calm` | Inject the calm direction | Reporting rate |
| Organizational injustice, manager takes credit | `anger` | Inject the anger direction | Reporting rate |

These names identify intervention conditions. The design contains two fixed scenarios, each with an unmodified baseline, rather than four independent questions.

## Files

- `run.py`: loads the 9B or 27B protocol and generates responses token by token, recording answers, tokens, sampling probabilities, and intervention norms.
- `numerics.py`: preserves the original BF16 arithmetic through the core hook interface.
- `extract.py`: extracts the required word directions and layers from the corresponding J-Lens asset.
- `launch.py`: divides the two scenarios into independent GPU shards; random draws for a sample ID do not change with sharding.
- `analyze.py`: recomputes choice rates, paired differences, intervals, and tests using the Python standard library.
- `configs/9b.json` and `configs/27b.json`: specify prompts, layers, doses, sampling settings, and historical condition-name mappings.

Historical runners, the old 50-draw experiment, and other direction searches are not required by these entry points.

## Recompute the supplied data

From the submission package root:

```bash
python verify.py
python recompute.py --output recomputed
```

Or run only the social-choice analysis:

```bash
python social_choices/data_processing/analyze.py --configs social_choices/configs --data social_choices/data --output recomputed_social
```

Stored records retain their original condition names. `source_conditions` maps `ablate_unfair`, `swap_fair`, and 9B `anger_x3_4` to display names. Original JSONL rows are unchanged. The new runner outputs display names, and the analyzer accepts both conventions.

## Generate new responses

With CUDA-compatible PyTorch installed, run from the submission package root:

```bash
python -m pip install -e './toolkit[hf]'
python toolkit/cookbook/social_choices/run.py --protocol social_choices/configs/9b.json --study reporting --model /models/Qwen3.5-9B --directions social_choices/assets/9b/directions.pt --output new_runs/9b/reporting_0
python toolkit/cookbook/social_choices/run.py --protocol social_choices/configs/27b.json --study ultimatum --model /models/Qwen3.5-27B --directions social_choices/assets/27b/directions.pt --output new_runs/27b/ultimatum_0
```

Multi-GPU execution can assign two GPUs to ultimatum and three to reporting:

```bash
python toolkit/cookbook/social_choices/launch.py --protocol social_choices/configs/27b.json --model /models/Qwen3.5-27B --directions social_choices/assets/27b/directions.pt --output new_runs/27b --study-gpus 0,1:2,3,4
```

Extract directions again:

```bash
python toolkit/cookbook/social_choices/extract.py --config social_choices/configs/27b.json --model /models/Qwen3.5-27B --lens /models/Qwen3.5-27B_jacobian_lens.pt --output new_assets/27b-directions.pt
```

Weights are loaded locally. The runner verifies model dimensions, layer count, configuration hashes, and relevant output-matrix rows. Output directories must be new. Original runs used A800 80GB GPUs; the 27B batch size of 64 needs sufficient memory. Hardware and batch differences can affect floating-point results.

## Protocol details

Each model has six condition cells with 5,000 draws each: 30,000 per model and 60,000 in total. Decoding uses temperature 0.7, top-p 0.95, top-k 50, at most 16 new tokens, and batch size 64, with thinking disabled. Conditions within a scenario share token-level random draws.

9B anger uses three-quarters of its original dose (11.30625), following unstable generation at the original dose. 27B anger retains 15.075. Calm uses 15.075 in both models, and ultimatum projection coefficients are 1. Each model uses its own vectors and layers mapped by relative depth; the 9B J-Lens was trained on the Base model.

The release organizes selected existing experiments. The operating configurations differ across models. All supplied primary responses are valid; dose explorations and other unused experiments are excluded. Confidence intervals quantify generation randomness for the fixed prompts and model configurations.

After rerunning both scenarios and models, analyze completed shards directly:

```bash
python toolkit/cookbook/social_choices/analyze.py --configs social_choices/configs --data new_runs --output new_analysis
```

Each model directory should contain `ultimatum_0` and `reporting_0` shards, with additional numbered shards where applicable. Only complete, non-smoke runs are accepted. Missing or duplicate sample IDs raise an error.

The commands above run inside the extracted full reproduction archive. In this source repository, the same runner is at `cookbook/social_choices/`; supply protocol, data, and direction paths from the extracted archive. See [downloads](../../artifacts/README.md).
