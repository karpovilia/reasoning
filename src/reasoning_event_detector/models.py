"""Return types. Plain dataclasses, JSON-serialisable through `.to_dict()`."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional


@dataclass(frozen=True)
class Event:
    """One reasoning event located in the text.

    `char_start`/`char_end` are offsets into the text that was passed in.
    For `source="regex"` they delimit the matched trigger; for `source="scorer"`
    they delimit the whole segment the trained head fired on — the head has no
    finer resolution than a segment, and pretending otherwise would overstate
    localisation the model does not have (see MODEL_CARD.md, honest F1 0.19).
    """
    type: str
    seg_id: int
    char_start: int
    char_end: int
    source: str
    match: Optional[str] = None
    word_pos: Optional[int] = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class Candidate:
    """First place in the reasoning text that looks like an answer candidate."""
    pattern: str
    char_pos: int
    word_pos: int
    context: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class TraceScore:
    """Result of scoring a finished trace."""
    events: List[Event]
    n_events: int
    counts: Dict[str, int]
    scalar: float
    n_segments: int
    n_words: int
    density_per_1k: float
    weights: str

    def to_dict(self) -> dict:
        d = asdict(self)
        d["events"] = [e.to_dict() for e in self.events]
        return d


@dataclass
class StreamUpdate:
    """State of the stream after one chunk."""
    chunk_index: int
    new_events: List[Event] = field(default_factory=list)
    events: List[Event] = field(default_factory=list)
    counts: Dict[str, int] = field(default_factory=dict)
    n_events: int = 0
    n_chars: int = 0
    n_words: int = 0
    n_segments_closed: int = 0
    recomputed: bool = False
    scalar: float = 0.0
    density_per_1k: float = 0.0
    candidate: Optional[Candidate] = None
    suppressed_retractions: int = 0

    def to_dict(self) -> dict:
        d = asdict(self)
        d["events"] = [e.to_dict() for e in self.events]
        d["new_events"] = [e.to_dict() for e in self.new_events]
        d["candidate"] = self.candidate.to_dict() if self.candidate else None
        return d


@dataclass
class SpendForecast:
    """Forecast of how much reasoning is still to come."""
    remaining_tokens: float
    consumed_words: int
    model: str
    note: str = ""

    def to_dict(self) -> dict:
        return asdict(self)
