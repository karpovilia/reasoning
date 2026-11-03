"""Incremental scoring of a reasoning stream.

What "incremental" means here, precisely:

  * A segment is CLOSED once text exists after it. The last segment of the
    buffer is always considered open, because the next chunk may extend it.
    On `finalize()` the last segment closes too.
  * On every recompute the scorer is re-run over the closed prefix. The GRU is
    causal, so a prefix decode uses no future text — but the regex-to-segment
    mapping of the frozen feature axis is expressed as a FRACTION of the trace
    (token_pos / token_total), so feature rows of early segments shift slightly
    as the trace grows. Consequence, measured on the frozen fixtures and quoted
    in MODEL_CARD.md: the counters at the end of a stream are close to, but not
    identical with, `score(full_text)`.
  * Counters are MONOTONE. An event that was emitted is never withdrawn, even
    if a later recompute no longer finds it; such cases are counted in
    `suppressed_retractions` so the drift is visible rather than hidden.

Cost: one recompute rescans the regex layer over the closed prefix and runs the
GRU over all closed segments, so a full stream is quadratic in trace length.
`recompute_every` trades latency for cost — see README for measured numbers.
"""
from __future__ import annotations

from typing import Dict, Iterable, Iterator, List, Optional, Set, Tuple

import numpy as np

from .candidate import first_candidate
from .models import Candidate, Event, StreamUpdate
from .scorer import EventScorer, text_to_payload


class StreamScorer:
    """Feed chunks in, get incremental typed counters out."""

    DEFAULT_COMMIT_LAG = 8

    def __init__(self, scorer: Optional[EventScorer] = None,
                 recompute_every: int = 1, track_candidate: bool = True,
                 weights: str = None, commit_lag: int = DEFAULT_COMMIT_LAG,
                 mapping: str = "char"):
        if scorer is None:
            scorer = EventScorer.load(weights) if weights else EventScorer.load()
        # "char" is the default here and only here: with the canonical "frac"
        # mapping the regex impulses slide between segments as the stream grows,
        # and a monotone counter then over-counts by up to 2.4x on long traces.
        # With mapping="char" and commit_lag >= 3 the stream reproduces
        # score(full_text) exactly on every frozen fixture.
        self.scorer = scorer.with_mapping(mapping)
        self.recompute_every = max(int(recompute_every), 1)
        self.commit_lag = max(int(commit_lag), 0)
        self.track_candidate = track_candidate
        self.reset()

    def reset(self) -> None:
        self._buf: List[str] = []
        self._text = ""
        self._chunk_index = -1
        self._closed_at_last_recompute = 0
        self._emitted: Set[Tuple[int, str]] = set()
        self._events: List[Event] = []
        self._counts: Dict[str, int] = {t: 0 for t in self.scorer.types}
        self._suppressed = 0
        self._candidate: Optional[Candidate] = None

    # -- state ------------------------------------------------------------
    @property
    def text(self) -> str:
        return self._text

    @property
    def events(self) -> List[Event]:
        return list(self._events)

    @property
    def counts(self) -> Dict[str, int]:
        return dict(self._counts)

    # -- driving ----------------------------------------------------------
    def update(self, chunk: str) -> StreamUpdate:
        """Append a chunk; recompute if enough new segments closed."""
        self._chunk_index += 1
        if chunk:
            self._buf.append(chunk)
            self._text = "".join(self._buf)
        return self._advance(final=False)

    def finalize(self) -> StreamUpdate:
        """Close the trailing segment and do a last recompute."""
        return self._advance(final=True)

    def _advance(self, final: bool) -> StreamUpdate:
        text = self._text
        payload, spans, ws_total = text_to_payload(text, self.scorer.domain)
        n_seg = len(payload["segments"])
        n_closed = n_seg if final else max(n_seg - 1, 0)

        if self.track_candidate and self._candidate is None and text:
            self._candidate = first_candidate(text)

        recompute = (n_closed > 0 and
                     (final or
                      n_closed - self._closed_at_last_recompute >= self.recompute_every))
        new_events: List[Event] = []
        if recompute:
            new_events = self._recompute(text, payload, spans, n_closed, final)
            self._closed_at_last_recompute = n_closed

        tgt = self.scorer.target_density(ws_total)
        scalar = float(np.clip(len(self._events) / tgt, 0.0, 1.0)) if tgt > 0 else 0.0
        return StreamUpdate(
            chunk_index=self._chunk_index,
            new_events=new_events,
            events=list(self._events),
            counts=dict(self._counts),
            n_events=len(self._events),
            n_chars=len(text),
            n_words=ws_total,
            n_segments_closed=n_closed,
            recomputed=recompute,
            scalar=scalar,
            density_per_1k=round(1000.0 * len(self._events) / max(ws_total, 1), 4),
            candidate=self._candidate,
            suppressed_retractions=self._suppressed,
        )

    def _recompute(self, text, payload, spans, n_closed, final) -> List[Event]:
        """Decode the closed prefix and commit everything outside the lag window.

        The commit boundary matters. Peak picking uses maximum_filter1d with
        edge replication, so the LAST segment of any prefix is always a local
        maximum and fires whenever its probability clears the threshold — a pure
        boundary artifact that would inflate a streaming counter by roughly
        threefold on long traces. Segments within `commit_lag` of the end of the
        prefix are therefore left uncommitted until more text arrives.
        """
        prefix_end = spans[n_closed - 1][1]
        prefix = text[:prefix_end]
        pl, pspans, _ = text_to_payload(prefix, self.scorer.domain)
        n_prefix_seg = len(pl["segments"])
        if n_prefix_seg == 0:
            return []
        X = self.scorer._features(pl)
        raw = self.scorer._decode_from_X(
            X, seed_events=self.scorer._regex_seed(pl, n_prefix_seg))

        boundary = n_prefix_seg if final else max(n_prefix_seg - self.commit_lag, 0)
        if boundary <= 0:
            return []
        decoded = {(seg, self.scorer.types[k]) for seg, k in raw if seg < boundary}
        self._suppressed += len({e for e in self._emitted if e[0] < boundary}
                                - decoded)
        fresh = sorted(decoded - self._emitted)
        out: List[Event] = []
        for seg, ty in fresh:
            k = self.scorer.types.index(ty)
            ev = self.scorer._to_event(seg, k, pspans, prefix)
            self._emitted.add((seg, ty))
            self._events.append(ev)
            self._counts[ty] = self._counts.get(ty, 0) + 1
            out.append(ev)
        self._events.sort(key=lambda e: (e.char_start, e.type))
        return out


def score_stream(chunks: Iterable[str], scorer: Optional[EventScorer] = None,
                 weights: str = None, recompute_every: int = 1,
                 track_candidate: bool = True, commit_lag: int = None,
                 mapping: str = "char",
                 finalize: bool = True) -> Iterator[StreamUpdate]:
    """Consume a chunk iterable, yield one StreamUpdate per chunk.

    >>> for upd in score_stream(chunks):           # doctest: +SKIP
    ...     print(upd.n_events, upd.counts)

    The final yielded update is the closing one (trailing segment included)
    unless finalize=False.
    """
    st = StreamScorer(scorer=scorer, weights=weights,
                      recompute_every=recompute_every,
                      track_candidate=track_candidate, mapping=mapping,
                      commit_lag=(StreamScorer.DEFAULT_COMMIT_LAG
                                  if commit_lag is None else commit_lag))
    for ch in chunks:
        yield st.update(ch)
    if finalize:
        yield st.finalize()
