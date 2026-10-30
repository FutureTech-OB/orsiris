# Source structure

| Component | Responsibility |
| --- | --- |
| `src/repsteer/interventions.py` | Add, ablate, swap, and hook lifecycle management |
| `src/repsteer/representations.py` | J-Space directions, TopK SAE features, and frozen controllers |
| `src/repsteer/backends.py` | Tiny-model and local Hugging Face adapters |
| `src/repsteer/config.py` | Experiment and material validation |
| `src/repsteer/runner.py` | Baseline/intervention runs and option-order evaluation |
| `src/repsteer/metrics.py` | Choice probabilities and treatment-effect errors |
| `src/repsteer/io.py` | Asset IO, hashes, and output provenance |
| `src/repsteer/cli.py`, `demo.py` | Command-line interface and CPU demonstration |
| `cookbook/social_choices/` | Direction extraction, generation, GPU sharding, numerical compatibility, and analysis |
| `cookbook/executive/` | Managerial launch, collection, and scoring interfaces |
| `experiments/` | Selected managerial execution engines and passive readout |
| `tests/` | Core, Hugging Face, recipe, and scoring tests |

The companion archives preserve complete experiment layouts so historical runtime paths can be reconstructed without access to the original machines. The reusable toolkit and selected experiment sources are available directly in this repository. Checkpoints, generated runs, temporary environments, and credentials are excluded from source control.
