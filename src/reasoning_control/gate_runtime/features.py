"""Versioned, content-blind observer input contract and the input ablations.

What the observer may see is an explicit whitelist (`Observation`): the number of
native tokens already spent (`now`), completed labelled segments in native token
coordinates with their event probabilities and span type, and the separately
elicited confidence probes. Nothing else: no text, embeddings, hidden states,
references, question/benchmark ids, future length, correctness or action outcomes.
Metadata used for grouping and reproducibility lives outside this module.

Every variant is a named channel mask over ONE feature layout (`FEATURE_NAMES`),
so a variant can only remove information, never add a hidden channel. The mask is
checked at encoding time and again by the model at prediction time.

Fixed semantics (decided before any result was read, see README.md):

* Records follow native token order, one per completed segment plus one final
  query record at `now`. `available_at` only gates admissibility (<= now) and
  feeds the `label_delay` channel; a late batch tagger does not change where an
  event sits on the token axis.
* Only the LAST confidence probe is exposed, in the query record, in all
  variants that allow confidence. Probe history never adds records and is not
  used by any variant here (a future, separate ablation).
* Counts are running sums of the segment event values. With the cheap tagger these
  values are probabilities, so a count is an EXPECTED number of events, not a
  number of confirmed events. No threshold is applied anywhere.
* Span age is measured up to the last annotated token of the current run, never
  to `available_at` and never to `now`. The unlabeled tail `now - last token_end`
  is a separate channel (`staleness` of the query record) and no span state is
  claimed for it.
* A gap of at most GAP_SLACK_TOKENS native tokens between consecutive segments is
  a tokenisation separator and keeps a span run continuous. A larger gap is an
  unlabeled stretch: the run restarts and the gap is accumulated in
  `uncovered_tokens`, so nothing is silently attributed to either side.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import math
from numbers import Integral
from typing import Sequence

import numpy as np

VERSION = 'eventseq_ablation_v2'
EVENTS = ('backtrack', 'verify', 'branch', 'subgoal_done', 'failed_attempt', 'commit')
SPANS = ('SETUP', 'DERIVING', 'EXPLORING', 'VERIFYING', 'REVISING', 'CONCLUDING', 'GROUNDING')
ACTIONS = ('stop', 'continue_512', 'continue_2048', 'verify_512', 'neutral_512')
GAP_SLACK_TOKENS = 3

MAIN_VARIANTS = ('time', 'confidence', 'counts', 'event_sequence', 'joint')
PRIMARY_ABLATIONS = ('joint_no_confidence', 'joint_no_timing')
DIAGNOSTIC_VARIANTS = ('counts_spans', 'event_sequence_shuffled', 'timeline',
                       'joint_no_age', 'spans_only')
COMPATIBILITY_VARIANTS = ('counts_narrow',)
VARIANTS = MAIN_VARIANTS + PRIMARY_ABLATIONS + DIAGNOSTIC_VARIANTS + COMPATIBILITY_VARIANTS
AGGREGATES = ('time', 'confidence', 'counts', 'counts_narrow', 'counts_spans')
SEQUENCES = tuple(v for v in VARIANTS if v not in AGGREGATES)

# Named columns are the input contract; nothing depends on anonymous offsets.
BASE_CHANNELS = ('now', 'record:segment', 'record:query')
SEGMENT_COUNT_CHANNELS = ('segment_count',)
TIMING_CHANNELS = ('token_start', 'token_end', 'native_gap', 'label_delay',
                   'staleness', 'uncovered_tokens')
EVENT_CHANNELS = tuple(f'event:{e}' for e in EVENTS)
COUNT_CHANNELS = tuple(f'count:{e}' for e in EVENTS)
SPAN_TYPE_CHANNELS = ('span_last:unknown',) + tuple(f'span_last:{s}' for s in SPANS)
SPAN_AGE_CHANNELS = ('observed_span_age',)
SPAN_DURATION_CHANNELS = tuple(f'covered_tokens:{s}' for s in SPANS)
CONFIDENCE_CHANNELS = ('confidence', 'confidence_present')
FEATURE_NAMES = (BASE_CHANNELS[:1] + SEGMENT_COUNT_CHANNELS + BASE_CHANNELS[1:]
                 + TIMING_CHANNELS + EVENT_CHANNELS + COUNT_CHANNELS
                 + SPAN_TYPE_CHANNELS + SPAN_AGE_CHANNELS + SPAN_DURATION_CHANNELS
                 + CONFIDENCE_CHANNELS)
INDEX = {name: i for i, name in enumerate(FEATURE_NAMES)}
WIDTH = len(FEATURE_NAMES)

_JOINT = (BASE_CHANNELS + SEGMENT_COUNT_CHANNELS + TIMING_CHANNELS + EVENT_CHANNELS
          + COUNT_CHANNELS + SPAN_TYPE_CHANNELS + SPAN_AGE_CHANNELS
          + SPAN_DURATION_CHANNELS + CONFIDENCE_CHANNELS)
VARIANT_CHANNELS = {
    'time': BASE_CHANNELS,
    'confidence': BASE_CHANNELS + CONFIDENCE_CHANNELS,
    'counts_narrow': BASE_CHANNELS + COUNT_CHANNELS + CONFIDENCE_CHANNELS,
    'counts': BASE_CHANNELS + SEGMENT_COUNT_CHANNELS + COUNT_CHANNELS + CONFIDENCE_CHANNELS,
    'counts_spans': (BASE_CHANNELS + SEGMENT_COUNT_CHANNELS + COUNT_CHANNELS
                     + SPAN_TYPE_CHANNELS + SPAN_AGE_CHANNELS + SPAN_DURATION_CHANNELS
                     + CONFIDENCE_CHANNELS),
    'event_sequence': (BASE_CHANNELS + SEGMENT_COUNT_CHANNELS + TIMING_CHANNELS
                       + EVENT_CHANNELS + COUNT_CHANNELS + CONFIDENCE_CHANNELS),
    'event_sequence_shuffled': (BASE_CHANNELS + SEGMENT_COUNT_CHANNELS + TIMING_CHANNELS
                                + EVENT_CHANNELS + COUNT_CHANNELS + CONFIDENCE_CHANNELS),
    'timeline': BASE_CHANNELS + SEGMENT_COUNT_CHANNELS + TIMING_CHANNELS + CONFIDENCE_CHANNELS,
    'joint': _JOINT,
    'joint_no_confidence': tuple(c for c in _JOINT if c not in CONFIDENCE_CHANNELS),
    'joint_no_timing': tuple(c for c in _JOINT if c not in TIMING_CHANNELS
                             + SPAN_AGE_CHANNELS + SPAN_DURATION_CHANNELS),
    'joint_no_age': tuple(c for c in _JOINT if c not in SPAN_AGE_CHANNELS + SPAN_DURATION_CHANNELS),
    'spans_only': tuple(c for c in _JOINT if c not in EVENT_CHANNELS + COUNT_CHANNELS),
}
assert set(VARIANT_CHANNELS) == set(VARIANTS)
CHANNEL_MASKS = {v: np.array([name in VARIANT_CHANNELS[v] for name in FEATURE_NAMES])
                 for v in VARIANTS}


@dataclass(frozen=True)
class Segment:
    """One completed labelled segment in NATIVE generated-token coordinates.

    token_start/token_end: half-open native span of the segment text;
    available_at: native position at which its label became available;
    events: six values in [0, 1] (probabilities or indicators, one label source);
    span: index into SPANS, or -1 for unknown.
    """
    token_start: int
    token_end: int
    available_at: int
    events: tuple[float, ...]
    span: int


@dataclass(frozen=True)
class Confidence:
    available_at: int
    probability: float


@dataclass(frozen=True)
class Observation:
    now: int
    segments: tuple[Segment, ...] = ()
    confidences: tuple[Confidence, ...] = ()


def _integer(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError(f'{name} must be a nonnegative integer')


def validate(observation: Observation):
    if not isinstance(observation, Observation):
        raise TypeError('Use the explicit Observation whitelist, not a metadata/label dict')
    _integer(observation.now, 'now')
    previous_end = 0
    previous_available = 0
    for segment in observation.segments:
        if not isinstance(segment, Segment):
            raise TypeError('segments must be Segment instances')
        for name in ('token_start', 'token_end', 'available_at'):
            _integer(getattr(segment, name), name)
        if not (previous_end <= segment.token_start < segment.token_end
                <= segment.available_at <= observation.now):
            raise ValueError('Overlapping, unordered, empty, duplicate or future segment')
        if segment.available_at < previous_available:
            raise ValueError('Out-of-order availability; align observations before encoding')
        if isinstance(segment.span, bool) or not isinstance(segment.span, Integral) \
                or segment.span not in range(-1, len(SPANS)):
            raise ValueError('Invalid span type')
        probabilities = np.asarray(segment.events, dtype=float)
        if probabilities.shape != (len(EVENTS),) or not np.isfinite(probabilities).all() \
                or ((probabilities < 0) | (probabilities > 1)).any():
            raise ValueError('Invalid event probabilities')
        previous_end, previous_available = segment.token_end, segment.available_at
    previous_confidence = -1
    for confidence in observation.confidences:
        if not isinstance(confidence, Confidence):
            raise TypeError('confidences must be Confidence instances')
        _integer(confidence.available_at, 'confidence.available_at')
        if not previous_confidence < confidence.available_at <= observation.now:
            raise ValueError('Duplicate, unordered or future confidence')
        if isinstance(confidence.probability, bool) \
                or not math.isfinite(confidence.probability) \
                or not 0 <= confidence.probability <= 1:
            raise ValueError('Invalid confidence')
        previous_confidence = confidence.available_at


def _log(value):
    if value < 0:
        raise ValueError('Negative magnitude')
    return np.log1p(value) / 10.


def shuffled_events(observation: Observation, seed: int) -> Observation:
    """Permute whole six-value event vectors across the fixed segment grid.

    Token coordinates, spans, availability and confidence stay in place; only the
    assignment event-vector -> segment moves. Sums are preserved by construction
    and the running counts are recomputed from the permuted sequence by encode().
    The seed must come from the prefix key, never from an outcome.
    """
    validate(observation)
    order = np.random.default_rng(seed).permutation(len(observation.segments))
    segments = tuple(replace(segment, events=observation.segments[j].events)
                     for segment, j in zip(observation.segments, order))
    return replace(observation, segments=segments)


def trajectory(observation: Observation) -> list[dict]:
    """Running state after each segment, in native token order (no variant logic).

    Returned keys: index, gap, continuous, uncovered_tokens, state, run_start,
    observed_span_age, counts (copy), durations (copy).
    """
    validate(observation)
    counts = np.zeros(len(EVENTS), dtype=np.float64)
    durations = np.zeros(len(SPANS), dtype=np.float64)
    state, run_start, previous_end, uncovered = -1, 0, 0, 0
    rows = []
    for i, segment in enumerate(observation.segments):
        gap = segment.token_start - previous_end
        continuous = gap <= GAP_SLACK_TOKENS
        if not continuous:
            uncovered += gap
        if segment.span != state or not continuous:
            state, run_start = segment.span, segment.token_start
        counts += segment.events
        if state >= 0:
            durations[state] += segment.token_end - segment.token_start
        rows.append({'index': i + 1, 'gap': gap, 'continuous': continuous,
                     'uncovered_tokens': uncovered, 'state': state, 'run_start': run_start,
                     'observed_span_age': segment.token_end - run_start if state >= 0 else 0,
                     'counts': counts.copy(), 'durations': durations.copy(),
                     'token_start': segment.token_start, 'token_end': segment.token_end,
                     'label_delay': segment.available_at - segment.token_end})
        previous_end = segment.token_end
    return rows


def encode(observation: Observation, variant: str, *, permutation_seed: int = 0) -> np.ndarray:
    """Encode one observation as [records, WIDTH]; aggregates give one record."""
    validate(observation)
    if variant not in VARIANTS:
        raise ValueError(f'Unknown variant: {variant}')
    if variant == 'event_sequence_shuffled':
        observation = shuffled_events(observation, permutation_seed)
    sequence = variant in SEQUENCES
    rows = trajectory(observation)
    last_end = observation.segments[-1].token_end if observation.segments else 0

    def record(kind, count, state, age, counts, durations, uncovered):
        x = np.zeros(WIDTH, dtype=np.float32)
        x[INDEX['now']] = _log(observation.now)
        x[INDEX['record:' + kind]] = 1
        x[INDEX['segment_count']] = _log(count)
        for e, value in zip(EVENTS, counts):
            x[INDEX['count:' + e]] = _log(value)
        x[INDEX['span_last:' + ('unknown' if state == -1 else SPANS[state])]] = 1
        x[INDEX['observed_span_age']] = _log(age)
        for span, duration in zip(SPANS, durations):
            x[INDEX['covered_tokens:' + span]] = _log(duration)
        x[INDEX['uncovered_tokens']] = _log(uncovered)
        return x

    result = []
    if sequence:
        for row, segment in zip(rows, observation.segments):
            x = record('segment', row['index'], row['state'], row['observed_span_age'],
                       row['counts'], row['durations'], row['uncovered_tokens'])
            for event, value in zip(EVENTS, segment.events):
                x[INDEX['event:' + event]] = value
            x[INDEX['token_start']] = _log(segment.token_start)
            x[INDEX['token_end']] = _log(segment.token_end)
            x[INDEX['native_gap']] = _log(row['gap'])
            x[INDEX['label_delay']] = _log(row['label_delay'])
            x[INDEX['staleness']] = _log(observation.now - segment.token_end)
            result.append(x)
    # Stable final sums: aggregate inputs must not encode float summation order.
    totals = np.array([math.fsum(s.events[j] for s in observation.segments)
                       for j in range(len(EVENTS))])
    last = rows[-1] if rows else {'state': -1, 'observed_span_age': 0,
                                  'durations': np.zeros(len(SPANS)), 'uncovered_tokens': 0}
    x = record('query', len(observation.segments), last['state'], last['observed_span_age'],
               totals, last['durations'], last['uncovered_tokens'])
    x[INDEX['staleness']] = _log(observation.now - last_end)
    if observation.confidences:
        x[INDEX['confidence']] = observation.confidences[-1].probability
        x[INDEX['confidence_present']] = 1
    result.append(x)
    encoded = np.stack(result)
    encoded[:, ~CHANNEL_MASKS[variant]] = 0
    if not np.isfinite(encoded).all():
        raise ValueError('Nonfinite features')
    check_mask(encoded, variant)
    return encoded


def check_mask(encoded: np.ndarray, variant: str):
    """Runtime guard: channels outside the variant's whitelist must be exactly zero."""
    if encoded.ndim != 2 or encoded.shape[1] != WIDTH:
        raise ValueError('Encoded block must be [records, WIDTH]')
    leaked = np.nonzero(np.abs(encoded[:, ~CHANNEL_MASKS[variant]]).sum(0))[0]
    if len(leaked):
        names = [n for n, m in zip(FEATURE_NAMES, ~CHANNEL_MASKS[variant]) if not m]
        raise ValueError(f'Channel leak in {variant}: {[names[i] for i in leaked]}')
    if variant in AGGREGATES and len(encoded) != 1:
        raise ValueError('Aggregate variant must be a single record')


def batch_sequences(sequences: Sequence[np.ndarray]):
    import torch
    from torch.nn.utils.rnn import pad_sequence
    if not sequences or any(x.ndim != 2 or x.shape[1] != WIDTH or len(x) == 0
                            or not np.isfinite(x).all() for x in sequences):
        raise ValueError('Invalid encoded batch')
    return (pad_sequence([torch.as_tensor(x, dtype=torch.float32) for x in sequences],
                         batch_first=True),
            torch.tensor([len(x) for x in sequences], dtype=torch.long))


def describe() -> dict:
    """Feature contract for manifests: names, groups, masks, fixed semantics."""
    return {
        'feature_version': VERSION, 'width': WIDTH, 'feature_names': list(FEATURE_NAMES),
        'events': list(EVENTS), 'spans': list(SPANS), 'actions': list(ACTIONS),
        'variants': list(VARIANTS), 'main_variants': list(MAIN_VARIANTS),
        'primary_ablations': list(PRIMARY_ABLATIONS),
        'diagnostic_variants': list(DIAGNOSTIC_VARIANTS),
        'compatibility_variants': list(COMPATIBILITY_VARIANTS),
        'aggregates': list(AGGREGATES),
        'variant_channels': {v: list(c) for v, c in VARIANT_CHANNELS.items()},
        'gap_slack_tokens': GAP_SLACK_TOKENS,
        'magnitude_transform': 'log1p(value)/10 for token positions, counts, ages, durations',
        'counts_are_sums_of_segment_event_values': True,
        'counts_are_expected_not_confirmed_events_when_values_are_probabilities': True,
        'confidence_exposed': 'last probe only, query record only',
        'span_age_measured_to': 'last annotated token of the run, not available_at, not now',
        'unlabeled_tail_channel': 'staleness of the query record',
        'record_order': 'native token order; available_at gates admissibility only',
    }
