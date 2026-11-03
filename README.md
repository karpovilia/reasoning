# Modeling LLM Reasoning Dynamics with Structural Representations

Code accompanying the English manuscript revised on 2 November 2026.

The implementation distinguishes three tasks: observing events and reasoning modes in a completed prefix, predicting future structure, and selecting a continuation action. T denotes text, E local events, and S spans of a reasoning mode. Retrospective annotations and prefix-only detector outputs are separate inputs.

## Implementation

- `src/ptrain`: segment Transformer and Qwen3-4B LoRA predictor; T/E/S ablations, annotation masks, question-weighted losses, training, recovery and prediction.
- `src/clean_detectors`: causal Qwen prefix detector with recurrent memory, a jointly trained structural prior and append-only revisions.
- `src/tes_process`: explicit event memory, semi-Markov, GRU and Neural Hawkes models; fitting, structural continuations and event-memory controls.
- `src/reasoning_control`: lexical observation, action-value prediction, gate training, calibration and inference.
- `src/effort_control`: fitting and calibration of the gpt-oss effort policy.
- `src/retrieval_control`: paired answer/search continuations, prefix features, controller calibration and answer scoring.
- `src/pfgate` and `src/gate_rebuild`: control from 29 process-forecast features and matched action-value fitting.
- `src/structural_transfer`: held-out solver-family and benchmark evaluation.
- `src/loop_control`: repetition heuristics and learned confirmation.
- `configs`: model, training and annotation configurations.
- `tests`: synthetic checks of causal masking, targets, objectives, recovery and action selection.

The paper-to-code mapping and commands are in [docs/usage.md](docs/usage.md). `CODE_ORIGINS.json` records the manuscript checksum and the source checksums used to prepare this distribution.

## Install and check

Use Python 3.12:

```bash
git clone https://github.com/karpovilia/reasoning.git
cd reasoning
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[neural,retrieval,test]'
git lfs install --local
git lfs pull
make check
make test
```

The included lexical-observer and GRU baseline weights use Git LFS. The large predictor checkpoints and training data are not bundled:

```bash
git lfs install --local
git lfs pull
python tools/release/check_release.py --weights
```

## Inputs and reproducibility

Training and evaluation commands consume explicitly supplied local data. Raw benchmark questions, reasoning traces, annotation records and saved per-question predictions are not included. The examples are synthetic. The code preserves unknown-label masks, question-level separation and natural termination versus generation caps.

Tests exercise the implementation on synthetic inputs; they do not repeat full training or generation experiments. Pretrained base language models are obtained separately under their model terms. Configurations pin the revisions used by the methods.
