"""Frozen canonical segmentation and stable character spans."""
from __future__ import annotations

import re
from dataclasses import dataclass

PARAGRAPH_SPLIT_RE = re.compile(r"\n+")


@dataclass(frozen=True)
class Segment:
    seg_id: int
    text: str
    char_start: int
    char_end: int
    starts_paragraph: bool


def segment(text: str) -> list[str]:
    """Verbatim behavior of the production canonical segmenter."""
    chunks = re.split(r"\n+", text)
    segs: list[str] = []
    for chunk in chunks:
        chunk = chunk.strip()
        if not chunk:
            continue
        for part in re.split(r"(?<=[.!?])\s+(?=[A-Z(])", chunk):
            part = part.strip()
            if part:
                segs.append(part)
    return segs


def segment_records(text: str) -> list[Segment]:
    """Return canonical segments with monotone spans in the untouched trace text."""
    segs = segment(text)
    out: list[Segment] = []
    cursor = 0
    for seg_id, value in enumerate(segs):
        start = text.find(value, cursor)
        if start < 0:
            raise ValueError(f"cannot recover segment {seg_id} after character {cursor}")
        end = start + len(value)
        between = text[cursor:start]
        starts_paragraph = seg_id == 0 or bool(PARAGRAPH_SPLIT_RE.search(between))
        out.append(Segment(seg_id, value, start, end, starts_paragraph))
        cursor = end
    return out


def physical_newlines_between(text: str, left_end: int, right_end: int) -> int:
    if not (0 <= left_end <= right_end <= len(text)):
        raise ValueError("invalid character interval")
    return text.count("\n", left_end, right_end)
