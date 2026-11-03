"""Feature axis of the trained scorer — regex channels and own-history channels.

VENDORED, FROZEN. Every function here is an inference-only extraction from
  internal_signals_poc/detector_no_gold.py
  sha256 4c53a96cc75e557e324aeec58a7ba7aab1ebbd6328e59d777d2d6d9fb3f85616
(_regex_to_seg, build_features_payload, regex_marks_payload, Standardizer,
_history_feats, _peaks, M_TYPES, R_TYPES, REGEX_DECAY). Training code, gold
loading and the npz internal-signal branch are deliberately NOT carried over:
this package is for service models, where logits and hidden states do not exist.

The only substantive rewrite is `max_filter1d`, a numpy replacement for
scipy.ndimage.maximum_filter1d(mode="nearest") so that the runtime needs numpy
only. tests/test_parity_with_canonical.py asserts the two agree.
"""
from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np

# ── frozen constants of the trained axis ─────────────────────────────────────
M_TYPES = ["backtrack", "verify", "branch", "subgoal_done", "failed_attempt",
           "commit"]
R_TYPES = ["retrieval_call", "evidence_merge", "contradiction", "decomposition",
           "query_rewrite", "commit", "backtrack", "verify", "internal_use",
           "retrieval_inadequate"]
REGEX_DECAY = 0.7            # затухание импульсных regex-каналов
SEED = 42
TOL = 3


def types_for_domain(domain: str) -> List[str]:
    return R_TYPES if domain == "R" else M_TYPES


# ── regex event -> segment index ─────────────────────────────────────────────

def regex_to_seg(payload: dict, n_seg: int) -> List[Tuple[int, str]]:
    """Каждое regex-событие -> индекс сегмента по ОТНОСИТЕЛЬНОЙ позиции.

    Verbatim behaviour of detector_no_gold._regex_to_seg. token_pos/token_total
    gives the fraction along the trace; segments are weighted by text length and
    located with searchsorted. Monotone and robust to the segmenter stripping
    whitespace (exact char reconstruction was impossible on payload files).
    """
    segs = payload.get("segments", [])
    if n_seg <= 0:
        return []
    lens = np.array([max(len(s.get("text", "")), 0) for s in segs],
                    dtype=np.float64)
    cum = np.cumsum(lens)
    total = float(cum[-1]) if len(cum) and cum[-1] > 0 else 1.0
    tt = max(int(payload.get("token_total", 0) or 0), 1)
    out = []
    for e in payload.get("regex_events", []):
        ty = e.get("type")
        tp = e.get("token_pos")
        frac = min(max((tp / tt) if tp is not None else 0.0, 0.0), 1.0)
        seg = int(np.searchsorted(cum, frac * total, side="left"))
        out.append((min(max(seg, 0), n_seg - 1), ty))
    return out


def regex_to_seg_char(payload: dict, n_seg: int) -> List[Tuple[int, str]]:
    """Exact variant of `regex_to_seg`, for when the real text is in hand.

    NOT part of the frozen training axis. The training axis had to infer the
    segment of a regex event from a token FRACTION, because payload files stored
    no character offsets. That mapping is unstable under prefix growth: the
    denominator (token_total) grows with the stream, so an event can land on a
    different segment at every recompute, which is what makes a naive streaming
    counter inflate. With the raw text we know each segment's character span, so
    the mapping is exact and, more importantly, immutable as text arrives.

    Requires payload["segment_spans"] = [(char_start, char_end), ...].
    """
    spans = payload.get("segment_spans") or []
    if n_seg <= 0 or not spans:
        return []
    starts = [s for s, _ in spans]
    out = []
    for e in payload.get("regex_events", []):
        cp = int(e.get("char_pos", 0))
        seg = int(np.searchsorted(starts, cp, side="right")) - 1
        out.append((min(max(seg, 0), n_seg - 1), e.get("type")))
    return out


def build_features_payload(payload: dict, types: Sequence[str],
                           mapping: str = "frac"):
    """[n_seg, K+1]: K decaying regex channels per type + the summed impulse.

    No internal signals of the generating model — this is exactly the
    `no_internal` ablation axis, which is what makes the scorer usable on a
    service model that only returns text.
    """
    segs = payload.get("segments", [])
    n = len(segs)
    K = len(types)
    tindex = {t: i for i, t in enumerate(types)}
    imp = np.zeros((max(n, 1), K))
    to_seg = regex_to_seg_char if mapping == "char" else regex_to_seg
    for seg, ty in to_seg(payload, n):
        if ty in tindex:
            imp[seg, tindex[ty]] = 1.0
    dec = np.zeros_like(imp)
    run = np.zeros(K)
    for t in range(max(n, 1)):
        run = run * REGEX_DECAY + imp[t]
        dec[t] = run
    cols = [dec[:, j] for j in range(K)]
    names = [f"rgx_{t}" for t in types]
    cols.append(imp.sum(1))
    names.append("rgx_any")
    X = np.stack(cols, axis=1).astype(np.float32)
    return (X[:n] if n else X[:0]), names


def regex_marks_payload(payload: dict, n_seg: int, types: Sequence[str],
                        mapping: str = "frac"):
    tset = set(types)
    to_seg = regex_to_seg_char if mapping == "char" else regex_to_seg
    return sorted((seg, ty) for seg, ty in to_seg(payload, n_seg) if ty in tset)


# ── standardizer (train statistics live in the checkpoint) ───────────────────

class Standardizer:
    """Клип по перцентилям train + z-score train. Только train-статистики."""

    def __init__(self, lo=None, hi=None, mu=None, sd=None):
        self.lo, self.hi, self.mu, self.sd = lo, hi, mu, sd

    def transform(self, X):
        return ((np.clip(X, self.lo, self.hi) - self.mu) / self.sd).astype(np.float64)


# ── history of own predictions ───────────────────────────────────────────────

def history_feats(n: int, events, K: int = len(M_TYPES), last_k: int = 3):
    """История ИЗ УЖЕ ВЫЯВЛЕННЫХ событий (каузально, до сегмента t невключительно):
    счётчики log1p по маркам, log1p tokens-since-last по маркам, one-hot последних
    last_k марок."""
    H = np.zeros((n, K + K + last_k * (K + 1)), dtype=np.float32)
    cnt = np.zeros(K)
    last = np.full(K, -1)
    recent: List[int] = []
    ei = 0
    evs = sorted(events)
    for t in range(n):
        while ei < len(evs) and evs[ei][0] < t:
            tok, mk = evs[ei]
            cnt[mk] += 1
            last[mk] = tok
            recent.append(mk)
            ei += 1
        H[t, :K] = np.log1p(cnt)
        tsl = np.where(last >= 0, np.log1p(t - last), 0.0)
        H[t, K:2 * K] = tsl
        for j in range(last_k):
            mk = recent[-1 - j] if len(recent) > j else K
            H[t, 2 * K + j * (K + 1) + mk] = 1.0
    return H


# ── peak picking (numpy replacement for scipy.ndimage.maximum_filter1d) ──────

def max_filter1d(x: np.ndarray, size: int) -> np.ndarray:
    """maximum_filter1d(x, size, mode="nearest") for odd `size`, numpy only."""
    r = size // 2
    if r == 0 or x.size == 0:
        return np.asarray(x)
    xp = np.pad(np.asarray(x), (r, r), mode="edge")
    win = np.lib.stride_tricks.sliding_window_view(xp, size)
    return win.max(axis=-1)


def peaks(p_evt: np.ndarray, radius: int = 2) -> np.ndarray:
    """Локальные максимумы p_evt с прореживанием плато (+-radius)."""
    mx = max_filter1d(p_evt, size=2 * radius + 1)
    idx = np.where(p_evt >= mx)[0]
    keep, last = [], -10 ** 9
    for t in idx:
        if t - last > radius:
            keep.append(int(t))
            last = t
    return np.array(keep, dtype=np.int64)
