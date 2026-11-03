"""Contract of the rebuilt gate: actions, the decision position, state sources, examples.

The first experiment is ONE decision at a common fixed position, with the three actions that the historical bank covers by real
branches for every active prefix (the 'a3' set). Limits are those of the bank generator
(event_seq_e2e_2026_09_25/pilot_src/runner.py): reasoning allowance per action and a separate final answer cap.

A state source is a named, versioned table of records computed from the available prefix only. For a sequence source, record s
is what is known AFTER completed segment s and before any later text is read. The gate never sees ids, the benchmark label,
future length, answers or outcomes; metadata used for grouping lives outside the tensors.

Changes against version 1: the cost of an example is the marginal cost of the action in the UNIT of the cost model, with the
solver tokens kept next to it; the example carries what the full cost of its attempt needs (prompt tokens, prefix tokens); the
sources of the first required matrix only (the separate process model of the P0 pair is a pending secondary diagnostic and has no
source here); the matched control of process pretraining is a state source made by the same producer, class, configuration and
inputs as the main candidate with randomly initialised weights.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

ACTIONS = ('stop', 'continue_512', 'continue_2048')
REASONING_ALLOWANCE = {'stop': 0, 'continue_512': 512, 'continue_2048': 2048}      # additional reasoning tokens
FINAL_ANSWER_CAP = 512                                                            # tokens, every action
DECISION_POSITION = 1024                                                          # native reasoning tokens already spent
SPLITS = ('train', 'dev', 'test')
QUESTION_HASH_WIDTH = 1024
BUNDLE = 'LoRA joint bundle 68f87ba1'

# name -> width of one record, whether records form a sequence over completed segments, and the observer that must have run
STATE_SOURCES = {
    'length_budget': {'width': 4, 'sequence': False, 'observer': None,
                      'records': 'prompt tokens, tokens spent, completed segments, mean tokens per segment; nothing of the question or of the reasoning is read'},
    'question_length_budget': {'width': QUESTION_HASH_WIDTH + 4, 'sequence': False, 'observer': None,
                               'records': 'hashed character n-gram counts of the QUESTION text and the four length values; no model is called, the reasoning is not read'},
    'old_lexical': {'width': 40, 'sequence': True, 'observer': 'lexical tagger detector_fold0.pkl',
                    'records': 'the 40 named channels of event_seq_ablation_2026_09_25/features.py, one record per completed segment'},
    'old_lexical_aggregate': {'width': 40, 'sequence': False, 'observer': 'lexical tagger detector_fold0.pkl',
                              'records': 'ONE record of the same historical encoder, variant counts_spans: the aggregate input of the best historical gate'},
    'lora_outputs': {'width': 25, 'sequence': True, 'observer': BUNDLE + ', accepted API',
                     'records': 'primary 6 event and 6 mode probabilities of segment s and the 13 values of the dynamics forecast made for it before its text was read'},
    'lora_process_state': {'width': 512, 'sequence': True, 'observer': BUNDLE + ', frozen pretrained dynamics transformer',
                           'records': 'output of the final norm of the dynamics transformer over the entries completed up to segment s (its forecast state for s+1); NOT an output of the accepted API'},
    'random_process_state': {'width': 512, 'sequence': True, 'observer': BUNDLE + ' inputs, dynamics transformer of the same class and configuration with RANDOM weights',
                             'records': 'the same computation as lora_process_state on the same entries with a seeded random initialisation instead of the pretrained weights'},
    'text_only': {'width': 2560, 'sequence': True, 'observer': 'frozen base of the bundle WITHOUT the LoRA adapter and without any head',
                  'records': 'pooled focus representation of segment s under the same serialization and context as the detector, adapter disabled'},
}


@dataclass(frozen=True)
class Example:
    """One decision state. Arrays are float32; np.nan marks an unknown target, never a failure."""
    key: str                      # prefix key of the bank
    group: str                    # question cluster: the unit of splitting and of every interval
    split: str
    states: dict                  # source name -> array [T, width] (T >= 1; T == 1 for a non-sequence source)
    budget: np.ndarray            # [len(ACTIONS), 2]: reasoning allowance / 2048 and final answer cap / 2048, known before the decision
    success: np.ndarray           # [len(ACTIONS)] fraction of known correct final answers over the sampled branches, nan when none is known
    cost: np.ndarray              # [len(ACTIONS)] marginal cost of the action in the unit of the cost model, nan when unknown
    tokens: np.ndarray            # [len(ACTIONS)] additional solver tokens of the action including its final answer, nan when unknown
    supported: np.ndarray         # [len(ACTIONS)] bool: a real branch of this action exists for this prefix; an unsupported action is never a failure
    prompt_tokens: int            # tokens of the prompt of this attempt
    prefix_tokens: int            # reasoning tokens decoded before the decision


def budget_features():
    return np.array([[REASONING_ALLOWANCE[a] / 2048., FINAL_ANSWER_CAP / 2048.] for a in ACTIONS], dtype=np.float32)


def validate(examples, split, sources):
    """Shape, range and leakage checks of one split. Raises ValueError; returns nothing."""
    if split not in SPLITS or not examples or any(e.split != split for e in examples):
        raise ValueError('expected a nonempty list of examples of the split %s only' % split)
    if len({e.key for e in examples}) != len(examples):
        raise ValueError('duplicate prefix key')
    for e in examples:
        if not e.key or not e.group:
            raise ValueError('example without key or question group')
        for name in sources:
            if name not in STATE_SOURCES or name not in e.states:
                raise ValueError('state source %s is unknown or missing for %s' % (name, e.key))
            state = np.asarray(e.states[name])
            spec = STATE_SOURCES[name]
            if state.ndim != 2 or state.shape[1] != spec['width'] or len(state) < 1 or (not spec['sequence'] and len(state) != 1) or not np.isfinite(state).all():
                raise ValueError('state %s of %s has an invalid shape or a non-finite value' % (name, e.key))
        success, cost, tokens, supported = (np.asarray(e.success, np.float32), np.asarray(e.cost, np.float32), np.asarray(e.tokens, np.float32), np.asarray(e.supported))
        if success.shape != (len(ACTIONS),) or cost.shape != success.shape or tokens.shape != success.shape or supported.shape != success.shape or supported.dtype != np.bool_:
            raise ValueError('targets of %s have an invalid shape' % e.key)
        if np.asarray(e.budget).shape != (len(ACTIONS), 2):
            raise ValueError('budget features of %s have an invalid shape' % e.key)
        if isinstance(e.prompt_tokens, bool) or isinstance(e.prefix_tokens, bool) or int(e.prompt_tokens) != e.prompt_tokens or int(e.prefix_tokens) != e.prefix_tokens \
                or e.prompt_tokens < 1 or e.prefix_tokens < 1:
            raise ValueError('%s has no valid prompt or prefix length' % e.key)
        known_s, known_c = success[np.isfinite(success)], cost[np.isfinite(cost)]
        if np.isinf(success).any() or np.isinf(cost).any() or np.isinf(tokens).any() or ((known_s < 0) | (known_s > 1)).any() or (known_c < 0).any() \
                or (tokens[np.isfinite(tokens)] < 0).any() or (np.isfinite(cost) != np.isfinite(tokens)).any():
            raise ValueError('targets of %s are outside their range' % e.key)
        if (np.isfinite(success) & ~supported).any() or (np.isfinite(cost) & ~supported).any():
            raise ValueError('%s has a target for an action without a real branch' % e.key)


def assert_question_disjoint(*splits):
    """Every question cluster lives in exactly one split."""
    seen = {}
    for examples in splits:
        for e in examples:
            if seen.setdefault(e.group, e.split) != e.split:
                raise ValueError('question %s appears in the splits %s and %s' % (e.group, seen[e.group], e.split))
