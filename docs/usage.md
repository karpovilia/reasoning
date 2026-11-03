# Running the paper implementation

Commands below run from the repository root after installing the package. Paths beginning with `data/local`, `models/base`, and `outputs` denote caller-supplied assets or generated output, not downloads provided by this repository. Use a new output directory for each independent experiment.

## Paper-to-code map

| Method in the manuscript | Implementation | Configuration / input |
| --- | --- | --- |
| Causal observation of events and reasoning modes | `clean_detectors.model`, `history_rollout`, `history_inference`; `prefix_api.detector_api_v1` | `configs/detector` |
| Segment Transformer and Qwen LoRA predictors, T/E/S input ablations | `ptrain.model_a`, `model_b`, `losses`, `runner`, `predict` | `configs/predictor` |
| Native annotation vocabularies and unknown-label masks | `ptrain.targets_native`, `namespace_policy`, `data` | Namespace definitions in `PACK_MANIFEST.json` |
| Retrospective structural observations and auxiliary corruption task | `ptrain.attach_retro_es`, `attach_group4` | `configs/observations` |
| Explicit event memory, semi-Markov, GRU, Neural Hawkes; structural rollouts | `tes_process.models`, `event_memory`, `fit_explicit`, `rollout_process` | `configs/process/readout.json`, prepared namespace arrays |
| Conditional/marginal readouts of the same process state | `tes_process.experiment`, `structure_diagnostics` | same frozen namespace membership and checkpoints |
| Transfer across solver families and benchmarks | `structural_transfer.holdout`, `holdout_eval`, `holdout_summary_v3` | `configs/transfer.json`, question-level regimes |
| Matched text/structure action-value models | `gate_rebuild.models`, `training`, `evaluation`, `policy` | paired action outcomes, disjoint question folds |
| Process forecasts as controller inputs | `matched_forecast`, `pfgate.pes_forward`, `matched_forward`, `worker`, `evaluate_matched` | 29 forecast columns, matched controller recipe |
| Effort selection, switching and stopping | `effort_control.fit_v3`, `calibrate_v3`, `generate`, `gate_analyze_v3` | difficulty/cycle models, observer weights, questions |
| Lexical observation and saved-continuation control | `reasoning_control.detector`, `gate_runtime.training`, `calibrate_gate`, `gate` | `configs/gate`, included baseline weights |
| Answer now or continue Wikipedia retrieval | `retrieval_control.branch_worker`, `extract_features`, `fit_evaluate`, `grade` | paired branch manifest, frozen CAL/EVAL membership |
| Repetition heuristics and learned loop confirmation | `loop_control.audit_baseline`, `run`, `losses` | loop labels and a supplied predictor checkpoint |

`CODE_ORIGINS.json` binds the manuscript and the source/release bytes. Packaging changes make imports and asset paths portable. A source checksum identifies the file copied from the experiment; a release checksum identifies the distributed file. Synthetic tests check implementation properties, not the paper's empirical claims.

## Offline example

This trains a small segment Transformer and predicts on synthetic DEV sequences. It needs neither model downloads nor benchmark data:

```bash
python examples/train_predictor.py --out outputs/synthetic_predictor
```

The example uses the same training, loss, checkpoint and prediction code as the full model, with a smaller architecture. Its generated pack is marked synthetic and cannot be used in `reproduce` mode.

The included lexical baseline can also observe segments and choose an action:

```bash
reasoning-gate --config configs/gate/matched_gru_joint.json \
  --detector models/detector/lexical-event-span-v1.pkl \
  --input examples/pipeline.jsonl --output outputs/decisions.jsonl
```

Use the detector path listed in `models/MANIFEST.json` if selecting another included baseline. These baseline weights are distinct from the Qwen prefix detector and large predictors.

## Predictor inputs and training

The unit interface is implemented by `ptrain.build_pack`. A JSONL roster binds each physical trace to its question component and `TRAIN`/`DEV`/protected role. Each producer directory has a `UNIT_MANIFEST_V1.json` and `units/` payloads. The builder verifies roster and unit hashes and never opens protected-role payloads. Termination receipts distinguish `natural`, `confirmed_truncated`, and `unknown`.

```bash
python -m ptrain.build_pack \
  --roster data/local/roster.jsonl --roster-sha256 "$ROSTER_SHA256" \
  --units data/local/units --out data/local/pack_base \
  --production-admissible
python -m ptrain.attach_retro_es \
  --pack data/local/pack_base --out data/local/pack_retro \
  --units data/local/units --roster data/local/roster.jsonl \
  --roster-sha256 "$ROSTER_SHA256" --decision configs/observations/retrospective.json
```

Declare admissibility only after validating the membership of your own dataset. The paper's frozen membership is not reconstructed from the latest available corpus. Unknown annotations remain masked; an absent label is not a negative target. Retrospective annotation inputs and outputs of a causal detector must carry different `es_input_source.kind` values.

Text is supplied in a SQLite `texts(sha, text)` table keyed by SHA-256 of UTF-8 text. For architecture A, compute frozen MiniLM vectors using `ptrain.embed --pack ... --texts ... --encoder ... --binding ...`; the binding JSON supplies `files: {filename: {sha256: ...}}` and the encoder revision. Pooling, normalization and truncation are implemented in `embed.py`. For architecture B, use `ptrain.tokenize --pack ... --texts ... --tokenizer ... --config ...` with the same model configuration that will be trained.

```bash
python -m ptrain.runner \
  --config configs/predictor/A_CTES_RETRO_seed42.json \
  --pack data/local/pack_retro --out outputs/predictor_A_CTES_42 \
  --seed 42 --device cuda --max-seconds 3600 --purpose reproduce
python -m ptrain.predict --run outputs/predictor_A_CTES_42 --device cuda
```

The time limit is an execution cap, not a replacement for the configured number of passes. Repeating the training command resumes a compatible checkpoint and charges previous fit time. Use the configuration's `fit_budget_seconds` to set the full cap. Architecture B requires the pinned Qwen3-4B-Instruct-2507 base model and tokenizer; it uses LoRA rank 32, alpha 64. The auxiliary G4 task is attached with `ptrain.attach_group4` and `configs/observations/corruption.json`; it is not applied to validation labels.

## Prefix detector

Train with a frozen dataset/configuration and explicit runtime settings:

```bash
python -m clean_detectors.train --config configs/detector/detector_config.json \
  --frozen data/local/detector/frozen --runtime data/local/detector/runtime.json \
  --output outputs/detector
```

`clean_detectors.dataset` defines the frozen data interface: a directory containing `manifest.json` and `samples.sqlite`. Set `REASONING_DETECTOR_DATA` to its data root. Runtime JSON must set `runtime_deadline_epoch` to a future Unix timestamp; an optional `plan_sha256` binds a separate execution plan. Training, calibration and own-history inference are separate steps. Calibration files must belong to the supplied checkpoint.

Assemble the inference package with its trained heads, adapter, two tokenizers and calibrated thresholds:

```bash
python tools/release/build_prefix_bundle.py \
  --checkpoint outputs/detector/checkpoint \
  --tokenizers models/base/detector_tokenizers \
  --calibration outputs/detector/calibration --output outputs/prefix_bundle
```

The bundle verifies file hashes, runtime code and the base model revision. `configs/detector/API_CONTRACT.json` and `schema.json` define the interface. Call `observe` only after a segment boundary is confirmed. The primary prediction is immutable; later revisions retain their target position and lag. The input uses the question, the completed prefix and the detector's own earlier probabilities. Future text, final correctness and annotation labels are not inference inputs.

## Structural process models

Set `TES_COMPARISON_ROOT` to an output/input working directory. Place the run recipe at `RUN.json` and prepared arrays at `prepared/ns9.npz`, with `prepared/ns9.json` holding their SHA-256, mode/event vocabularies and sequence membership. Each sequence specifies `start`, `n`, question and role: 0 for fit, 1 for tuning within TRAIN, 2 for DEV. Array fields are listed in `tes_process.fit_explicit.FIELDS`.

```bash
mkdir -p outputs/process
cp configs/process/readout.json outputs/process/RUN.json
export TES_COMPARISON_ROOT=outputs/process
python -m tes_process.fit_explicit 9 explicit_full_text 42
python -m tes_process.fit_explicit 9 explicit_erasure_text 42
python -m tes_process.fit_explicit 9 gru_text 42
python -m tes_process.experiment 42
```

Prepare the frozen arrays before these commands. The readout experiment consumes the already fitted process models; its `source_root` identifies that fit directory. Repeat the configured seeds for the paper comparison. Do not pool incompatible annotation vocabularies into a single label index.

## Transfer and forecast-augmented control

`structural_transfer.build_regimes` builds question-level training/held-out memberships. `holdout` takes `--v2` (directory containing prepared `data/` and model `config.json`), `--inputs` (target NPZ), `--regimes-dir`, `--regimes`, `--seeds 17,29`, `--out`, and `--mode train|check`. `holdout_summary_v3` scores natural completion and censored traces separately and compares models only on their common observed positions.

The forecast-control pipeline consists of:

1. `pfgate.build_pes_records`: build causal structural records from completed segments.
2. `pfgate.pes_forward` or `matched_forward`: export 29 forecasts using supplied, checksum-verified checkpoints. `src/matched_forecast` contains the matched predictor and reference interfaces.
3. `pfgate.build_gate_table` / `build_gate_table_variant`: join forecasts to the fixed decision states.
4. `pfgate.worker init CONFIG OUTPUT`, then `pfgate.worker run CONFIG OUTPUT`: fit matched action-value models on their question folds.
5. `pfgate.evaluate_matched`: evaluate the selected paths and paired question-level differences.

Invoke these as `python -m MODULE`. Positional arguments are documented in each module's docstring. The worker configuration supplies data paths and hashes, `project_code` (the installed `src` tree), expected outer-fold checksum, feature width, time cap, and the recipe `{epochs, learning_rate, weight_decay}`. It rejects changes to frozen inputs when resuming. Forecast weights and per-question action outcomes are supplied separately.

## Effort control

Fit the difficulty/cycle/event heads with `python -m effort_control.fit_v3 --help`; calibrate them with `effort_control.calibrate_v3`. Generation is implemented in `effort_control.generate`. It requires the question JSONL, fitted packs, student observer checkpoint and a generation-capable gpt-oss serving environment. Its default budget grid is 200, 287, 411, 589, 845, 1211, 1736 and 2489 tokens; the two controller settings are 0.3 and 0.5. `gate_analyze_v3` compares fixed-effort caps and the low/standard-effort envelope using saved outcomes.

The vLLM generation environment is separate from the CPU test environment. No generation service is launched by installation or tests.

## Retrieval control

`branch_worker` generates the two actions from a saved prefix. `extract_features` computes common counters, frozen text embeddings and structural features. Their command-line interfaces require explicit serving endpoints, model assets and input/output paths.

`cases.jsonl` has one row per question: `case_id`, `role` (`CAL` or `EVAL`), `benchmark` (`hotpotqa` or `musique`), `prefix_sha256`, and `selection_hash`. `tasks.jsonl` supplies exactly one `answer` and one `continue` task per case, with unique `task_id`. Feature filenames are the first 24 hex characters of SHA-256 of `case_id`; response filenames are `task_id.json`. Feature and response schemas are consumed by `feature_matrix` and `outcome_arrays`.

```bash
python -m retrieval_control.fit_evaluate \
  --cases data/local/retrieval/cases.jsonl --tasks data/local/retrieval/tasks.jsonl \
  --features data/local/retrieval/features --responses data/local/retrieval/responses \
  --gold data/local/retrieval/cal_gold.jsonl --gold-eval data/local/retrieval/eval_gold.jsonl \
  --output outputs/retrieval --mode all
```

Calibration is fixed at 25 HotpotQA and 25 MuSiQue questions. Evaluation uses the explicit membership in the cases file. For the manuscript this is the 65 HotpotQA questions with both continuations complete at its analysis cutoff, without outcome-based selection. Do not replace it with a later 450-question run and call that a reproduction of the same result.

The fitter reads CAL outcomes first and writes `POLICY_FREEZE.json` before reading EVAL outcomes. It verifies the freeze and input hashes during evaluation. Missing branches block completion. Unknown generation costs prevent a confirmed cost-saving claim. `RESULT.json` contains answer F1, exact match, output-token counts, tool calls, measured timing components and paired intervals; `COMPLETE.json` records the evaluated task coverage.

## Loop confirmation

```bash
python -m loop_control.audit_baseline --input data/local/loops.jsonl \
  --output outputs/loop_audit --workers 1
```

The learned confirmation model is `loop_control.run`. Its explicit inputs are `REASONING_LOOP_RUN`, `REASONING_LOOP_WEIGHTS`, `REASONING_LOOP_PACK`, and `REASONING_QWEN_BASE`; the run directory contains the task specification and loop-label data. Base predictor training and loop-label fitting use separate memberships.

## Available assets

Only the small lexical-observer and recurrent-gate baseline weights listed in `models/MANIFEST.json` are distributed through Git LFS. The causal Qwen observer, the large T/E/S predictors, process-model checkpoints, training corpora, benchmark questions, annotation payloads and per-question alternative continuations are not bundled. Full numerical reproduction requires those inputs; a successful synthetic test is not evidence that a paper result has been regenerated.
