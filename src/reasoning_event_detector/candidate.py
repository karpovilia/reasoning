"""First answer candidate inside the reasoning text (not in the final block).

VENDORED, FROZEN. Pattern list copied verbatim from
  internal_signals_poc/event_validation_2026_08_03/scripts/exp4a_candidates.py
  sha256 faeba1fb8e5afe33ca872e5f9216ef892933a205145efb04b6f7bbd99ffa883a
which is the v2 list: fixed before any threshold tuning, revised once after an
eyeball check of 20 random traces, and never touched again. Coverage measured
on that frozen list: 181 of 192 traces, 94.3 per cent (P4 128, P2 50, P5 3).

Known bias, stated in the source and repeated here: P4 often anchors on an
intermediate terminal equality rather than the final candidate, which places the
anchor early. Treat the position as "the first moment the trace looks like it
has an answer", not as "the answer".

The only difference from the source is the token coordinate. The original
reported `cand_tok` in gpt-oss BPE; this package has no tokenizer dependency and
reports whitespace-word position instead. The character anchor is identical.
"""
from __future__ import annotations

import re
from typing import Optional

from .models import Candidate
from .patterns import char_to_token_pos

PATTERNS = [
    ("P1_boxed", re.compile(r"\\boxed\{")),
    ("P2_answer_is", re.compile(
        r"(?<!to )(?<!To )\b(?:the\s+)?(?:final\s+)?(?:answer|result)"
        r"\s*(?:is\b|:|=)", re.IGNORECASE)),
    ("P3_otvet", re.compile(r"\bответ\b", re.IGNORECASE)),
    ("P4_terminal_eq", re.compile(
        r"=\s*(?:-?\d[\d,./]*|\(?[A-E]\)?)\s*(?:[.!?]|\n|$)")),
    ("P5_answer_letter", re.compile(
        r"\banswer\s+\(?([A-E])\)?(?:[.!?\s]|$)")),
]

CONTEXT_RADIUS = 80


def first_candidate(text: str, context_radius: int = CONTEXT_RADIUS
                    ) -> Optional[Candidate]:
    """Earliest match over the frozen pattern list, or None."""
    best = None
    for name, rx in PATTERNS:
        m = rx.search(text)
        if m and (best is None or m.start() < best[1]):
            best = (name, m.start())
    if best is None:
        return None
    name, cp = best
    lo = max(0, cp - context_radius)
    hi = min(len(text), cp + context_radius)
    return Candidate(pattern=name, char_pos=int(cp),
                     word_pos=char_to_token_pos(text, cp),
                     context=text[lo:hi].replace("\n", " "))
