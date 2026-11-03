# Contributing

This repository contains the implementation accompanying the paper. Keep changes tied to a method, input contract, configuration or reproducibility check. Project diaries, historical reports, proposals, internal coordination records and unrelated experiments do not belong here.

Use paths relative to the repository or explicit command-line inputs. Do not embed machine addresses, credentials, host-specific defaults or private data locations. Never include benchmark question text, identifiers or reasoning traces from GPQA or HLE. Fixtures must be synthetic.

Preserve causal prefix inputs, availability masks, question-level splits, natural END versus censoring, and the specified loss weighting. Changes to targets, features, splits or selection rules require an explicit configuration change.

Run `make check` and `make test` before committing. `make weights` checks local model bytes against the LFS manifest. The optional benchmark-content scan is `python tools/check_no_leaks.py --strict`; reference data are configured outside this repository.

Final published weights use Git LFS. Datasets, generated predictions, intermediate checkpoints, environments and execution logs stay outside Git. Keep machine-specific services and scheduler submissions out of importable modules.
