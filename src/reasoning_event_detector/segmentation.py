"""Deterministic segmentation of a reasoning trace into units.

VENDORED, FROZEN. `segment` is a verbatim copy of the project's canonical
segmenter:
  internal_signals_poc/select_and_segment.py :: segment
  sha256 of source file 8e0d346991a64a979233767366fb018e34e8593cd4e8219c868ddc9e1995ce7b

The segment grid is the unit the trained scorer was fitted on: gold events are
addressed by seg_id, the GRU runs one step per segment. Changing the segmenter
changes the model input, so this function is frozen together with the weights.
`tests/test_parity_with_canonical.py` re-checks it against the source when the
canonical repository is available.

`segment_spans` is new here: it recovers the character span of every segment in
the original text. Payload files never stored char offsets (the segmenter strips
whitespace, so offsets were unrecoverable there), but when we hold the raw text
we can find them exactly by sequential search. Positions of events reported by
this package come from those spans.
"""
from __future__ import annotations

import re
from typing import List, Tuple


def segment(text: str) -> List[str]:
    """Грубая, но детерминированная сегментация: по \\n и по границам предложений,
    с защитой от дробления формул (не режем на '.', если вокруг цифры/символы)."""
    # сначала по переводам строк
    chunks = re.split(r"\n+", text)
    segs = []
    for ch in chunks:
        ch = ch.strip()
        if not ch:
            continue
        # делим на предложения: точка/!/? + пробел + заглавная/«We»/«So»/«Wait» и т.п.
        parts = re.split(r"(?<=[.!?])\s+(?=[A-Z(])", ch)
        for p in parts:
            p = p.strip()
            if p:
                segs.append(p)
    return segs


def segment_spans(text: str, segs: List[str] | None = None
                  ) -> Tuple[List[str], List[Tuple[int, int]]]:
    """(segments, [(char_start, char_end), ...]) for `text`.

    Mirrors seg_char_starts of
    internal_signals_poc/event_validation_2026_08_03/scripts/exp4a_events_hist.py:
    a forward-only search keeps the mapping monotone; if an exact match fails we
    retry on the first 40 characters, and fall back to the running cursor. The
    fallback is degenerate only for texts the segmenter rewrote, which it does
    not do (it only strips).
    """
    if segs is None:
        segs = segment(text)
    spans: List[Tuple[int, int]] = []
    cursor = 0
    for s in segs:
        pos = text.find(s, cursor)
        if pos < 0:
            pos = text.find(s[:40], cursor)
        if pos < 0:
            pos = cursor
        spans.append((pos, pos + len(s)))
        cursor = pos + len(s)
    return segs, spans
