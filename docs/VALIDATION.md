# Validation scope

Run `python tools/verify.py` after installing the toolkit. It covers core intervention arithmetic, hook cleanup, J-Space and SAE operations, frozen controllers, tiny local Hugging Face adapters, deterministic sampling, recipe configurations, and hand-calculable managerial scoring fixtures. Tests do not download models or use external APIs.

The CPU demo uses random tiny weights and is marked synthetic. Software validation is distinct from rerunning the research models.

Both public companion archives are independently extracted, checked against file manifests, and recomputed from their stored records before publication. The release includes 60,000 generated responses, 606 managerial probability records, and four passive readout records. Packaging does not add new GPU observations. Version 6 uses one reporting rule for all four managerial studies: average model effects across option orders, then compute scale-normalized absolute error. Original controller fitting and selection are unchanged.

See `artifacts/PUBLICATION_CHECKS.json` for checks on this public release. Local detailed test logs are written under ignored `local/`.
