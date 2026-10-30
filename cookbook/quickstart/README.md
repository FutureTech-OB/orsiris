# Quickstart

Run from the repository root:

```bash
python cookbook/quickstart/python_api.py
python -m repsteer validate cookbook/quickstart/qwen27_jspace_organization.json --schema-only
```

`python_api.py` demonstrates the intervention API without a large model. The JSON files illustrate scenario and response-option schemas, J-Space directions, SAE feature combinations, and frozen controllers. Paths are relative to each configuration file.

The feature IDs in `qwen27_sae_organization.json` are format examples, not the fitted research controller. Prepare the matching local model and representation weights before execution. For exact experimental recipes and frozen assets, use the companion reproduction archive and the research Cookbooks.
