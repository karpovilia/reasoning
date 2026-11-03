"""The trained event scorer, running on text only.

Port of internal_signals_poc/grpo_new_labeling_2026_07_15/event_scorer.py
(sha256 4bd76b1d9cf701d292d89e31887ee6f38653d0b128aeeb88e3c041c36e72b38b),
inference half only: no training, no gold loading, no LLM forward pass.

Two changes versus the canonical implementation, both verified in
tests/test_parity_with_canonical.py:

  * the GRU runs in numpy, so the runtime needs numpy alone. Reading the
    original `.pt` blob still needs torch; `tools/export_npz.py` produces a
    torch-free `.npz` twin with the same numbers.
  * events carry character spans, recovered from the segment spans of the real
    text. The canonical code returned `(seg_id, type)` pairs because payload
    files had no offsets.
"""
from __future__ import annotations

import bisect
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import weights as _weights
from .features import (Standardizer, build_features_payload, history_feats,
                       peaks, regex_marks_payload, types_for_domain)
from .models import Event, TraceScore
from .patterns import char_to_token_pos, detect_events
from .segmentation import segment_spans


# ── numpy GRU (PyTorch nn.GRU, batch_first, 1 layer, tanh/sigmoid) ───────────

def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def gru_forward(x: np.ndarray, w_ih: np.ndarray, w_hh: np.ndarray,
                b_ih: np.ndarray, b_hh: np.ndarray) -> np.ndarray:
    """[T, D] -> [T, H]. Same gate order as torch: r, z, n."""
    T = x.shape[0]
    H = w_hh.shape[1]
    gi_all = x @ w_ih.T + b_ih                      # [T, 3H]
    out = np.empty((T, H), dtype=np.float32)
    h = np.zeros(H, dtype=np.float32)
    for t in range(T):
        gi = gi_all[t]
        gh = w_hh @ h + b_hh
        r = _sigmoid(gi[:H] + gh[:H])
        z = _sigmoid(gi[H:2 * H] + gh[H:2 * H])
        n = np.tanh(gi[2 * H:] + r * gh[2 * H:])
        h = (1.0 - z) * n + z * h
        out[t] = h
    return out


def _softmax(z: np.ndarray) -> np.ndarray:
    m = z.max(axis=-1, keepdims=True)
    e = np.exp(z - m)
    return e / e.sum(axis=-1, keepdims=True)


# ── text -> the payload dict the feature builder expects ─────────────────────

def text_to_payload(text: str, domain: str = "M"):
    """Reproduce the payload fields build_features_payload consumes.

    Identical to event_scorer.text_to_payload: canonical regex events, whitespace
    token positions, deterministic segmentation. `spans` is extra — the char
    span of every segment in `text`.
    """
    segs, spans = segment_spans(text)
    segments = [{"seg_id": i, "text": s} for i, s in enumerate(segs)]
    ws_total = max(len(text.split()), 1)
    regex_events = []
    for e in detect_events(text, domain):
        regex_events.append({
            "type": e["type"],
            "char_pos": int(e["char_pos"]),
            "char_end": int(e["char_end"]),
            "match": e["match"],
            "token_pos": int(char_to_token_pos(text, e["char_pos"])),
        })
    payload = {"segments": segments, "regex_events": regex_events,
               "token_total": ws_total, "segment_spans": spans}
    return payload, spans, ws_total


class EventScorer:
    """Trained detector, text in — typed events out. CPU, no LLM."""

    def __init__(self, params: Dict[str, np.ndarray], std: Standardizer,
                 thr: Sequence[float], cols: Sequence[int],
                 types: Sequence[str], config: dict, name: str = "custom",
                 mapping: str = "frac"):
        self.p = {k: np.asarray(v, dtype=np.float32) for k, v in params.items()}
        self.std = std
        self.thr = np.asarray(thr, dtype=np.float64)
        self.cols = list(cols)
        self.types = list(types)
        self.config = dict(config)
        self.name = name
        if mapping not in ("frac", "char"):
            raise ValueError("mapping must be 'frac' (canonical training axis) "
                             "or 'char' (exact, stable under streaming)")
        self.mapping = mapping
        self.K = int(config["K"])
        self.use_history = bool(config.get("use_history", True))
        self.decode_iters = int(config.get("decode_iters", 2))
        self.seed_from_regex = bool(config.get("seed_from_regex", True))
        self.domain = config.get("domain", "M")
        self.target_density_per_1k = float(
            config.get("target_density_per_1k", 8.0))

    # -- loading ----------------------------------------------------------
    @classmethod
    def load(cls, name: Optional[str] = None, path: Optional[str] = None,
             verify: bool = True, prefer: str = "npz") -> "EventScorer":
        """Load a registered checkpoint (sha256-checked) or an explicit path.

        With `prefer="npz"` (default) the torch-free twin of a checkpoint is
        used when it is present, so nothing but numpy is imported. Pass
        prefer="torch" to read the original `.pt` blob instead — same numbers,
        but it needs torch installed.
        """
        if path and not name:
            p = Path(path)
            if not p.is_file():
                raise _weights.WeightsUnavailable(f"no such checkpoint: {p}")
            loader = cls._from_npz if p.suffix == ".npz" else cls._from_torch
            return loader(p, p.stem)

        name = name or _weights.DEFAULT_WEIGHTS
        order = [name]
        if prefer == "npz" and not name.endswith("_npz") \
                and f"{name}_npz" in _weights.REGISTRY:
            order.insert(0, f"{name}_npz")
        last: Exception = _weights.WeightsUnavailable(f"unknown weights '{name}'")
        for cand in order:
            try:
                p = _weights.resolve(cand, path=path, verify=verify)
            except _weights.WeightsUnavailable as exc:
                last = exc
                continue
            spec = _weights.REGISTRY[cand]
            loader = cls._from_npz if spec.fmt == "npz" else cls._from_torch
            return loader(p, cand)
        raise last

    @classmethod
    def _from_torch(cls, path, name):
        import torch                                    # optional dependency
        blob = torch.load(path, map_location="cpu", weights_only=False)
        sd = {k: v.detach().cpu().numpy() for k, v in blob["state_dict"].items()}
        params = {"w_ih": sd["gru.weight_ih_l0"], "w_hh": sd["gru.weight_hh_l0"],
                  "b_ih": sd["gru.bias_ih_l0"], "b_hh": sd["gru.bias_hh_l0"],
                  "fc_w": sd["fc.weight"], "fc_b": sd["fc.bias"]}
        s = blob["standardizer"]
        std = Standardizer(*[np.asarray(s[k]) for k in ("lo", "hi", "mu", "sd")])
        return cls(params, std, np.asarray(blob["thresholds"]), blob["cols"],
                   blob["types"], blob["config"], name=name)

    @classmethod
    def _from_npz(cls, path, name):
        import json
        z = np.load(path, allow_pickle=False)
        params = {k: z[k] for k in ("w_ih", "w_hh", "b_ih", "b_hh", "fc_w", "fc_b")}
        std = Standardizer(*[z[f"std_{k}"] for k in ("lo", "hi", "mu", "sd")])
        config = json.loads(str(z["config_json"]))
        types = [str(t) for t in z["types"]]
        return cls(params, std, z["thresholds"], [int(c) for c in z["cols"]],
                   types, config, name=name)

    # -- core decode -------------------------------------------------------
    def _decode_once(self, Z: np.ndarray, events) -> List[Tuple[int, int]]:
        n = len(Z)
        if self.use_history:
            H = history_feats(n, events, K=self.K)
        else:
            H = np.zeros((n, 0), dtype=np.float32)
        Xin = np.concatenate([Z, H], axis=1).astype(np.float32)
        hs = gru_forward(Xin, self.p["w_ih"], self.p["w_hh"],
                         self.p["b_ih"], self.p["b_hh"])
        logits = hs @ self.p["fc_w"].T + self.p["fc_b"]
        P = _softmax(logits.astype(np.float64))[:, :self.K]
        out = []
        for k in range(self.K):
            pk = P[:, k]
            for t in peaks(pk):
                if pk[t] >= self.thr[k]:
                    out.append((int(t), k))
        return sorted(out)

    def _decode_from_X(self, X: np.ndarray, seed_events=None, iters=None):
        """decode_all replayed on a single trace: standardise, GRU, per-type
        NMS + calibrated threshold, history bootstrapped from the model's own
        predictions (seeded with the regex marks, which exist at stream time)."""
        Xc = X[:, self.cols]
        Z = self.std.transform(Xc).astype(np.float32)
        if len(Z) == 0:
            return []
        events = sorted(seed_events) if seed_events else []
        for _ in range(max(iters if iters is not None else self.decode_iters, 1)):
            new_events = self._decode_once(Z, events)
            if new_events == events:
                break
            events = new_events
        return events

    def _regex_seed(self, payload, n_seg):
        if not self.seed_from_regex:
            return None
        tindex = {t: i for i, t in enumerate(self.types)}
        marks = regex_marks_payload(payload, n_seg, self.types, mapping=self.mapping)
        return [(seg, tindex[ty]) for seg, ty in marks if ty in tindex]

    def with_mapping(self, mapping: str) -> "EventScorer":
        """Cheap clone sharing the same weights, with another regex->segment map."""
        if mapping == self.mapping:
            return self
        return EventScorer(self.p, self.std, self.thr, self.cols, self.types,
                           self.config, name=self.name, mapping=mapping)

    def _features(self, payload):
        return build_features_payload(payload, self.types, mapping=self.mapping)[0]

    # -- public ------------------------------------------------------------
    def decode_text(self, text: str) -> Tuple[List[Event], dict, list, int]:
        payload, spans, ws_total = text_to_payload(text, self.domain)
        n_seg = len(payload["segments"])
        X = self._features(payload)
        raw = self._decode_from_X(X, seed_events=self._regex_seed(payload, n_seg))
        events = [self._to_event(seg, k, spans, text) for seg, k in raw]
        return events, payload, spans, ws_total

    def _to_event(self, seg: int, k: int, spans, text: str) -> Event:
        lo, hi = spans[seg] if seg < len(spans) else (0, 0)
        return Event(type=self.types[k], seg_id=int(seg), char_start=int(lo),
                     char_end=int(hi), source="scorer",
                     word_pos=char_to_token_pos(text, lo))

    def score(self, text: str) -> TraceScore:
        """Full score of a finished trace: events, counters, reward scalar."""
        events, payload, _, ws_total = self.decode_text(text)
        counts = {t: 0 for t in self.types}
        for e in events:
            counts[e.type] += 1
        tgt = self.target_density(ws_total)
        scalar = float(np.clip(len(events) / tgt, 0.0, 1.0)) if tgt > 0 else 0.0
        return TraceScore(events=events, n_events=len(events), counts=counts,
                          scalar=scalar, n_segments=len(payload["segments"]),
                          n_words=ws_total,
                          density_per_1k=round(1000.0 * len(events) / max(ws_total, 1), 4),
                          weights=self.name)

    def target_density(self, n_words: int) -> float:
        """Reference event count for a trace this long (teacher's density)."""
        return self.target_density_per_1k * max(n_words, 1) / 1000.0


# ── regex-only layer, no weights needed ──────────────────────────────────────

def detect(text: str, domain: str = "M") -> List[Event]:
    """Rule layer: typed events with exact character spans. No model, no weights.

    domain="M" -> 6 math/science types; domain="R" -> 10 retrieval types.
    The R domain has a rule layer only: no trained scorer was ever fitted for it.
    """
    segs, spans = segment_spans(text)
    starts = [s for s, _ in spans]
    out: List[Event] = []
    for e in detect_events(text, domain):
        cp = int(e["char_pos"])
        seg = max(bisect.bisect_right(starts, cp) - 1, 0) if starts else 0
        out.append(Event(type=e["type"], seg_id=seg, char_start=cp,
                         char_end=int(e["char_end"]), source="regex",
                         match=e["match"], word_pos=char_to_token_pos(text, cp)))
    return out


def count_by_type(events: Sequence[Event], domain: str = "M") -> Dict[str, int]:
    counts = {t: 0 for t in types_for_domain(domain)}
    for e in events:
        counts[e.type] = counts.get(e.type, 0) + 1
    return counts
