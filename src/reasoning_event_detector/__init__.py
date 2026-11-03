"""Reasoning-event detector for SERVICE models.

A service model is one you reach through an API: you get the reasoning text, or
a stream of it, and nothing else. No logits, no hidden states, no second forward
pass. Everything in this package works under that constraint.

Four entry points:

    detect(text, domain="M")     rule layer -> typed events with char spans
    score_stream(chunks)         trained scorer -> incremental typed counters
    candidate(text)              first answer-candidate anchor in the text
    spend_forecast(text)         remaining-budget forecast (no weights shipped)

Read MODEL_CARD.md before quoting any number this package produces. In short:
the trained scorer inherits its definition of "event" from its teacher, a
DeepSeek judge, and event definitions do not transfer between annotators
(judge vs judge 0.745, judge vs templates 0.513, detector transfer 0.05 and
-0.08). Report counts and correlations; per-element localisation is weak by
construction (honest macro-F1 0.19).
"""
from __future__ import annotations

from typing import Iterable, List, Optional

from .candidate import first_candidate as _first_candidate
from .models import (Candidate, Event, SpendForecast, StreamUpdate, TraceScore)
from .scorer import EventScorer, count_by_type, detect
from .spend import SpendWeightsUnavailable, spend_forecast
from .stream import StreamScorer, score_stream
from .weights import (DEFAULT_WEIGHTS, WeightsCorrupted, WeightsUnavailable,
                      available as available_weights)

__version__ = "0.1.0"

__all__ = [
    "detect", "score_stream", "candidate", "spend_forecast",
    "EventScorer", "StreamScorer", "score_trace", "count_by_type",
    "Event", "Candidate", "TraceScore", "StreamUpdate", "SpendForecast",
    "available_weights", "DEFAULT_WEIGHTS",
    "WeightsUnavailable", "WeightsCorrupted", "SpendWeightsUnavailable",
    "__version__",
]


def candidate(text: str) -> Optional[Candidate]:
    """First answer-candidate anchor in the reasoning text, or None.

    Coverage on the frozen validation set: 94.3 per cent of traces (181/192).
    """
    return _first_candidate(text)


def score_trace(text: str, weights: str = DEFAULT_WEIGHTS,
                scorer: Optional[EventScorer] = None) -> TraceScore:
    """Score a finished trace with the trained detector (counts + reward scalar).

    Loads the checkpoint on every call; hold an `EventScorer` yourself if you
    are scoring more than a handful of traces.
    """
    sc = scorer or EventScorer.load(weights)
    return sc.score(text)
