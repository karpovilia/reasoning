"""Adapters from the pilot's JSONL layers to Observation/Example objects.

Three separate inputs, never merged into one blob:
* observer JSONL: key, group, fold, trace_key, now, segments[...] (native coords);
* confidence JSONL: key, confidence (the separately elicited probe at `now`);
* branch labels JSONL: key, action, seed, <correctness column>, <cost column>.

The correctness column and the cost column are chosen explicitly by name so a
future measured cost or a second grading layer can be used without renaming.
The file hash is recorded as provenance. Unknown correctness (null) stays unknown.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np

from .features import ACTIONS, EVENTS, Confidence, Observation, Segment, validate
from .training import Example, aggregate_draws

FOLD_SPLITS = {3: 'train', 4: 'validation', 0: 'test'}
OBSERVER_METADATA = ('key', 'group', 'fold', 'split', 'trace_key')


def sha256(path) -> str:
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def jsonl(path):
    with Path(path).open() as f:
        return [json.loads(line) for line in f if line.strip()]


def observation_from_row(row: dict, confidence: float | None = None) -> Observation:
    """Whitelist copy: only now/segments/confidences reach the Observation."""
    segments = tuple(Segment(token_start=s['token_start'], token_end=s['token_end'],
                             available_at=s['available_at'],
                             events=tuple(float(v) for v in s['events']), span=int(s['span']))
                     for s in row['segments'])
    history = tuple(Confidence(c['available_at'], float(c['probability']))
                    for c in row.get('confidences', []))
    if any(c.available_at >= row['now'] for c in history):
        raise ValueError('Observer row carries a probe at/after now; pass it via the confidence file')
    if confidence is not None:
        history = history + (Confidence(row['now'], float(confidence)),)
    observation = Observation(now=int(row['now']), segments=segments, confidences=history)
    validate(observation)
    return observation


def load_observations(observer_path, confidence_path=None, *, require_confidence=False):
    """Returns {key: (Observation, metadata)} plus provenance. Metadata is separate."""
    rows = jsonl(observer_path)
    confidence = {}
    if confidence_path:
        for r in jsonl(confidence_path):
            if r['key'] in confidence:
                raise ValueError(f'Duplicate confidence for {r["key"]}')
            if r.get('parse_error') not in (None, '') or r.get('confidence') is None:
                raise ValueError(f'Unparsed confidence for {r["key"]} must not be silently dropped')
            confidence[r['key']] = float(r['confidence'])
    result = {}
    for r in rows:
        if r['key'] in result:
            raise ValueError(f'Duplicate observer key {r["key"]}')
        if require_confidence and r['key'] not in confidence:
            raise ValueError(f'Missing confidence for {r["key"]}')
        observation = observation_from_row(r, confidence.get(r['key']))
        result[r['key']] = (observation, {m: r[m] for m in OBSERVER_METADATA if m in r})
    provenance = {'observer_sha256': sha256(observer_path),
                  'confidence_sha256': sha256(confidence_path) if confidence_path else None,
                  'keys': len(result), 'confidence_keys': len(confidence),
                  'keys_without_confidence': sorted(set(result) - set(confidence)) if confidence_path else sorted(result)}
    return result, provenance


def load_branch_labels(path, *, correctness_field='correct', cost_field='additional_tokens'):
    """{(key, action, seed): (correct|None, cost|None, row)}; duplicates rejected."""
    index, seeds = {}, set()
    for r in jsonl(path):
        slot = (r['key'], r['action'], r['seed'])
        if slot in index:
            raise ValueError(f'Duplicate branch label {slot}')
        if r['action'] not in ACTIONS:
            raise ValueError(f'Unknown action {r["action"]}')
        if correctness_field not in r or cost_field not in r:
            raise ValueError(f'Label row lacks {correctness_field!r} or {cost_field!r}')
        y = r[correctness_field]
        if y is not None and not isinstance(y, (bool, np.bool_)):
            raise ValueError('Correctness must be bool or null, never a score or a prediction')
        c = r[cost_field]
        if c is not None and (not np.isfinite(c) or c < 0):
            raise ValueError('Invalid cost')
        index[slot] = (y, None if c is None else float(c), r)
        seeds.add(r['seed'])
    return {'index': index, 'seeds': tuple(sorted(seeds)), 'sha256': sha256(path),
            'path': str(path), 'correctness_field': correctness_field, 'cost_field': cost_field,
            'rows': len(index)}


def build_examples(observations, labels, *, keys=None, fold_splits=FOLD_SPLITS):
    """One Example per prefix key with masked action targets over all seed slots.

    Missing branch rows are unknown for both correctness and cost; missing
    correctness with a present cost keeps the cost target (masked training).
    Keys outside `keys` are reported, not silently dropped.
    """
    seeds = labels['seeds']
    if not seeds:
        raise ValueError('Label file has no seeds')
    chosen = sorted(observations) if keys is None else sorted(keys)
    missing = sorted(set(chosen) - set(observations))
    if missing:
        raise ValueError(f'Requested keys without observations: {missing[:5]}')
    examples, excluded, coverage = [], [], Counter()
    for key in chosen:
        observation, meta = observations[key]
        by_fold = fold_splits.get(meta.get('fold'))
        declared = meta.get('split')
        if declared is not None and by_fold is not None and declared != by_fold:
            raise ValueError(f'{key}: declared split {declared!r} disagrees with fold {meta.get("fold")}')
        split_name = declared if declared is not None else by_fold
        if split_name not in ('train', 'validation', 'test'):
            excluded.append({'key': key, 'reason': f'fold {meta.get("fold")} / split {declared} not in splits'})
            continue
        correct, cost = [], []
        for action in ACTIONS:
            rows = [labels['index'].get((key, action, s)) for s in seeds]
            correct.append([None if r is None else r[0] for r in rows])
            cost.append([None if r is None else r[1] for r in rows])
            coverage[f'{action}:missing_rows'] += sum(r is None for r in rows)
            coverage[f'{action}:unknown_correctness'] += sum(r is not None and r[0] is None for r in rows)
        y, c, cov = aggregate_draws(correct, cost)
        examples.append(Example(key=key, group=meta['group'], split=split_name,
                                observation=observation, success=y, cost=c, coverage=cov))
    unused = sorted({k for k, _, _ in labels['index']} - set(chosen))
    return examples, {'excluded': excluded, 'coverage': dict(coverage),
                      'label_keys_not_used': unused, 'seeds': list(seeds),
                      'split_prefixes': dict(Counter(e.split for e in examples)),
                      'split_questions': {s: len({e.group for e in examples if e.split == s})
                                          for s in sorted({e.split for e in examples})}}


def split(examples):
    parts = {'train': [], 'validation': [], 'test': []}
    for e in examples:
        parts[e.split].append(e)
    groups = {s: {e.group for e in v} for s, v in parts.items()}
    if groups['train'] & groups['validation'] or groups['train'] & groups['test'] \
            or groups['validation'] & groups['test']:
        raise ValueError('Question leakage across splits')
    return parts


def common_keys(a: dict, b: dict):
    """Explicit intersection of two observation layers with the exclusions listed."""
    shared = sorted(set(a) & set(b))
    return shared, {'only_first': sorted(set(a) - set(b)), 'only_second': sorted(set(b) - set(a))}


def label_source_kind(observations) -> dict:
    """Are segment event values indicators (0/1) or probabilities? Reported, not mixed."""
    values = np.array([v for o, _ in observations.values() for s in o.segments for v in s.events])
    binary = bool(len(values)) and bool(np.isin(values, (0., 1.)).all())
    return {'segment_event_values': int(len(values)),
            'values_are_indicators': binary,
            'counts_meaning': 'confirmed events' if binary else 'expected events (sum of probabilities)',
            'events': list(EVENTS)}
