#!/usr/bin/env python3
"""The causal bridge from the observations of the LoRA detector to the inputs of the process predictor P_ES
(PROCESS_FORECAST_GATE_20261013). numpy only, no model.

P_ES (event_span_transformer_ablation_20261013/endfix_v1/src/ablation.py, cell ES: 12 layers, width 512, context 256) reads one
record per COMPLETED segment: 41 numbers, the type of the current span, the type of the previous span, the solver, the benchmark.
It was trained on HARD labels of a retrospective annotation: events 0/1, one span type per segment. The detector gives
probabilities, so they are made hard by the rules the accepted detector bundle 68f87ba1 itself publishes, nothing chosen here:
  event      1 when its probability of the immutable primary answer is at least the validation-only F1 threshold of the bundle
             (calibration/thresholds.json, key f1, in the order of the events), else 0; all six are 'known';
  span type  the argmax of the six mode probabilities (a tie: the first); a span is a run of equal argmax, as in the API of the
             bundle (spans = runs of the latest argmax mode). The seventh type of P_ES, GROUNDING, is never produced: the
             detector has no such mode.
Record j uses the segments 0..j only: no future length of the span, no end of the trace, no label of a later segment.
The 41 columns, exactly as prepare_ablation.build makes them from the corpus (span_transformer_2026_10_07/v1/prepare.py and
v2/prepare.py for the old columns): characters are len() of the segment text, j counts from 0, positions from 1.
   0      log1p(j + 1) / 10                               1   log1p(characters of segment j) / 10
   2      log1p(characters of segments 0..j) / 10         3   log1p(age of the current span in segments, from 1) / 10
   4      log1p(characters of the current span so far) / 10
   5:11   the six events of segment j                     11:17  their known flags (all 1 here)
   17:23  log1p(memory with decay exp(-1/4)) / 5          23:29  log1p(memory with decay exp(-1/32)) / 5
   29:35  log1p(count of the event over segments 0..j) / 5
   35:41  log1p(position - position of the last segment with the event, or - 0 when there was none) / 10
The memories never reset at a span boundary: memory[j] = memory[j-1] * decay + events[j]."""
from __future__ import annotations

import numpy as np

EVENTS = ('backtrack', 'verify', 'branch', 'subgoal_done', 'failed_attempt', 'commit')
MODES = ('SETUP', 'DERIVING', 'EXPLORING', 'VERIFYING', 'REVISING', 'CONCLUDING', 'GROUNDING')        # the operators of P_ES; the detector emits the first six
NO_PREVIOUS_SPAN = 7
DECAYS = (np.exp(-1 / 4), np.exp(-1 / 32))


def hard_events(event_probabilities, thresholds):
    p, t = np.asarray(event_probabilities, np.float64), np.asarray(thresholds, np.float64)
    if p.ndim != 2 or p.shape[1] != 6 or t.shape != (6,) or not np.isfinite(p).all() or not np.isfinite(t).all() or (p < 0).any() or (p > 1).any():
        raise ValueError('six finite event probabilities per segment and six thresholds are required')
    return (p >= t[None]).astype(np.float64)


def hard_modes(mode_probabilities):
    p = np.asarray(mode_probabilities, np.float64)
    if p.ndim != 2 or p.shape[1] != 6 or not np.isfinite(p).all():
        raise ValueError('six finite mode probabilities per segment are required')
    return p.argmax(1).astype(np.int64)


def records(characters, events, modes):
    """(xin [n, 41] float32, mode [n], prev_mode [n]) from HARD events [n, 6] in {0, 1} and span types [n] in 0..6."""
    chars, ev, modes = np.asarray(characters, np.float64), np.asarray(events, np.float64), np.asarray(modes, np.int64)
    n = len(chars)
    if n < 1 or ev.shape != (n, 6) or modes.shape != (n,) or (chars < 0).any() or not np.isin(ev, (0., 1.)).all() or (modes < 0).any() or (modes > 6).any():
        raise ValueError('characters [n], hard events [n, 6] and span types [n] of one prefix are required')
    age, span_chars, prev_mode, memory = np.zeros(n), np.zeros(n), np.full(n, NO_PREVIOUS_SPAN, np.int64), np.zeros((n, 12))
    start, previous = 0, NO_PREVIOUS_SPAN
    decay = np.repeat(DECAYS, 6)
    for j in range(n):
        if j and modes[j] != modes[j - 1]:
            previous, start = int(modes[j - 1]), j
        age[j], span_chars[j], prev_mode[j] = j - start + 1, chars[start:j + 1].sum(), previous
        if j:
            memory[j] = memory[j - 1] * decay
        memory[j] += np.tile(ev[j], 2)
    position = np.arange(1, n + 1, dtype=np.float64)[:, None]
    last = np.maximum.accumulate(np.where(ev > 0, position, 0.0), 0)
    xin = np.zeros((n, 41), np.float64)
    xin[:, 0] = np.log1p(np.arange(n) + 1) / 10
    xin[:, 1] = np.log1p(chars) / 10
    xin[:, 2] = np.log1p(np.cumsum(chars)) / 10
    xin[:, 3] = np.log1p(age) / 10
    xin[:, 4] = np.log1p(span_chars) / 10
    xin[:, 5:11] = ev
    xin[:, 11:17] = 1.0
    xin[:, 17:29] = np.log1p(memory) / 5
    xin[:, 29:35] = np.log1p(ev.cumsum(0)) / 5
    xin[:, 35:41] = np.log1p(position - last) / 10
    return xin.astype(np.float32), modes, prev_mode


def bridge(characters, event_probabilities, mode_probabilities, thresholds):
    """The records of one prefix from the detector probabilities of its completed segments."""
    return records(characters, hard_events(event_probabilities, thresholds), hard_modes(mode_probabilities))
