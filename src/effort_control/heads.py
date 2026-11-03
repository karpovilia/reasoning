#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Три головы панели рассуждения: события, интенсивности, бюджет цикла.

Голова А, события. Дистиллированный ученик судьи (рекуррентная голова поверх
замороженных векторов предложений, арм sft_judge_distill_2026_08_13) либо, если
нет тяжёлых зависимостей, обученный числовой детектор из этого же пакета.
Поток идёт по закрытым сегментам: сегмент считается закрытым, когда после него
появился текст. Каждый пересчёт переразмечает закрытый префикс целиком, а
наружу выдаются только НОВЫЕ события, поэтому счётчики монотонны и одно и то же
место не считается дважды. Это тот же порядок, что в потоковом режиме детектора.

Голова Б, интенсивности. Самовозбуждающийся процесс с экспоненциальным ядром,
параметры перефитованы на судейской разметке (арм hawkes_heads_2026_08_13).
Наружу отдаются две вещи и только они: доли типов для столбиков и интервал
ожидания ближайшего события. Интервал считается по точному условному
распределению времени до следующего события, а не по точечной оценке.

Голова В, бюджет цикла. Цикл — отрезок от предыдущей консолидации (или от
начала рассуждения) до ближайшей следующей. Сколько токенов осталось до конца
цикла, берётся из эмпирической таблицы условных квантилей по состоянию: фаза,
сколько событий уже было в цикле, сколько токенов цикл длится, был ли кандидат
ответа. Таблица считается офлайн на размеченном корпусе и здесь только читается.

Модуль намеренно обходится одним numpy для голов Б и В: тем же кодом считается и
живая панель, и проверка на реплее. Тяжёлые зависимости головы А подгружаются
лениво и только если выбран ученик.
"""
from __future__ import annotations

import bisect
import json
import math
import os
import re
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

M_TYPES = ["backtrack", "verify", "branch", "subgoal_done", "failed_attempt", "commit"]
K = len(M_TYPES)
TIDX = {t: i for i, t in enumerate(M_TYPES)}
COMMIT = TIDX["commit"]

OPERATORS = ["SETUP", "DERIVING", "EXPLORING", "VERIFYING", "REVISING", "CONCLUDING"]

# человеческие слова для панели; внутренние ярлыки наружу не показываются
PHASE_WORDS = {
    "SETUP": "разбирает условие",
    "DERIVING": "ведёт выкладку",
    "EXPLORING": "перебирает варианты",
    "VERIFYING": "проверяет решение",
    "REVISING": "переделывает начатое",
    "CONCLUDING": "сводит к ответу",
}
EVENT_WORDS = {
    "backtrack": "возврат",
    "verify": "проверка",
    "branch": "ветвление",
    "subgoal_done": "шаг готов",
    "failed_attempt": "тупик",
    "commit": "решение",
}

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PACK_DIR = os.path.join(HERE, "packs")


# ── разбивка состояния на ячейки таблицы ─────────────────────────────────────

NEV_EDGES = [1, 2, 3, 5, 8]          # 0 | 1 | 2 | 3-4 | 5-7 | 8+
# ступени возраста цикла в токенах по умолчанию; настоящие ступени лежат в
# наборе параметров рядом с ячейками, потому что ячейки под них и считались
AGE_EDGES = [60.0, 150.0, 350.0, 800.0]


def nev_bucket(n: int, edges=None) -> int:
    return bisect.bisect_right(edges or NEV_EDGES, int(n))


def age_bucket(tok: float, edges=None) -> int:
    return bisect.bisect_right(edges or AGE_EDGES, float(tok))


def pos_bucket(frac: float) -> int:
    """Доля пройденного пути по трассе, если она известна; иначе средняя ячейка."""
    if frac is None:
        return 1
    return min(int(max(frac, 0.0) * 4.0), 3)


# ── голова Б: самовозбуждающийся процесс ─────────────────────────────────────

class HawkesHead:
    """Интенсивности шести типов и интервал ожидания ближайшего события."""

    def __init__(self, mu, A, beta: float, types: Sequence[str] = M_TYPES):
        self.mu = np.asarray(mu, dtype=np.float64)
        self.A = np.asarray(A, dtype=np.float64)
        self.beta = float(beta)
        self.types = list(types)
        self.rowsum = self.A.sum(axis=1)
        self.mu_tot = float(self.mu.sum())
        self.reset()

    @classmethod
    def from_pack(cls, pack: dict, model: str = None) -> "HawkesHead":
        p = pack.get(model) or pack.get("__pooled__") or next(iter(pack.values()))
        return cls(p["mu"], p["A"], p["beta"], p.get("types", M_TYPES))

    def reset(self) -> None:
        self.S = np.zeros(K, dtype=np.float64)   # накопленное возбуждение по типам
        self.t = 0.0

    def advance(self, t: float) -> None:
        dt = float(t) - self.t
        if dt > 0:
            self.S *= math.exp(-self.beta * dt)
            self.t = float(t)

    def observe(self, t: float, k: int) -> None:
        self.advance(t)
        self.S[int(k)] += self.beta

    def intensities(self, t: float = None) -> np.ndarray:
        if t is not None:
            self.advance(t)
        return self.mu + self.A.T.dot(self.S)

    def next_probs(self, t: float = None) -> np.ndarray:
        lam = self.intensities(t)
        s = float(lam.sum())
        return lam / s if s > 0 else np.full(K, 1.0 / K)

    # доля типа в ближайшем событии по точному интегралу, а не мгновенная;
    # используется как контрольная величина в проверке на реплее
    def next_probs_exact(self, t: float = None, horizon_mult: float = 12.0,
                         steps: int = 400) -> np.ndarray:
        if t is not None:
            self.advance(t)
        G = float(self.rowsum.dot(self.S))
        u_max = horizon_mult / max(self.mu_tot, 1e-9)
        u = np.linspace(0.0, u_max, steps + 1)
        decay = np.exp(-self.beta * u)
        # lambda_k(t+u) = mu_k + (A^T S)_k * exp(-beta u)
        lam_k = self.mu[None, :] + (self.A.T.dot(self.S))[None, :] * decay[:, None]
        comp = self.mu_tot * u + (G / self.beta) * (1.0 - decay)
        surv = np.exp(-comp)
        w = surv[:, None] * lam_k
        p = np.trapezoid(w, u, axis=0) if hasattr(np, "trapezoid") else \
            np.trapz(w, u, axis=0)
        s = float(p.sum())
        return p / s if s > 0 else np.full(K, 1.0 / K)

    def compensator(self, u: float) -> float:
        """Интеграл полной интенсивности на [t, t+u] при отсутствии новых событий."""
        G = float(self.rowsum.dot(self.S))
        return self.mu_tot * u + (G / self.beta) * (1.0 - math.exp(-self.beta * u))

    def wait_quantiles(self, qs: Sequence[float] = (0.1, 0.5, 0.9)) -> List[float]:
        """Квантили времени до ближайшего события, в токенах.

        Распределение точное: при экспоненциальном ядре интеграл интенсивности
        на интервале без событий берётся в замкнутой форме, квантиль находится
        обращением. Это интервал, а не точечная оценка — так и показываем.
        """
        out = []
        for q in qs:
            z = -math.log(max(1.0 - float(q), 1e-12))
            hi = z / max(self.mu_tot, 1e-9)
            lo = 0.0
            for _ in range(60):
                mid = 0.5 * (lo + hi)
                if self.compensator(mid) < z:
                    lo = mid
                else:
                    hi = mid
            out.append(0.5 * (lo + hi))
        return out


# ── голова В: эмпирическая таблица «до конца цикла» и фаза ───────────────────

class CycleTable:
    """Условные квантили остатка цикла и наиболее вероятная фаза."""

    LEVELS = ["m_ph_nev_age_cand", "m_ph_age_cand", "m_age_cand",
              "m_age", "m", "p_age", "p"]

    def __init__(self, pack: dict):
        self.q = pack.get("quantiles", {})
        self.phase = pack.get("phase", {})
        self.min_n = int(pack.get("min_n", 40))
        self.meta = pack.get("meta", {})
        # ступени берутся из набора параметров: ячейки считались под них, и
        # разъехавшиеся ступени молча увели бы поиск на грубый уровень
        self.age_edges = [float(x) for x in self.meta.get("age_edges", AGE_EDGES)]
        self.nev_edges = [int(x) for x in self.meta.get("nev_edges", NEV_EDGES)]

    @classmethod
    def load(cls, path: str) -> "CycleTable":
        return cls(json.load(open(path, encoding="utf-8")))

    @staticmethod
    def keys(model: str, phase_i: int, nev: int, age: int, cand: int) -> List[str]:
        m = model or "__pooled__"
        return [
            "%s|%d|%d|%d|%d" % (m, phase_i, nev, age, cand),
            "%s|%d|%d|%d" % (m, phase_i, age, cand),
            "%s|%d|%d" % (m, age, cand),
            "%s|%d" % (m, age),
            "%s" % m,
            "__pooled__|%d" % age,
            "__pooled__",
        ]

    def lookup(self, model: str, phase_i: int, nev: int, age: float,
               cand: bool) -> Tuple[List[float], str, int]:
        ab = age_bucket(age, self.age_edges)
        nb = nev_bucket(nev, self.nev_edges)
        for lvl, key in zip(self.LEVELS,
                            self.keys(model, phase_i, nb, ab, int(bool(cand)))):
            cell = self.q.get(lvl, {}).get(key)
            if cell and cell[3] >= self.min_n:
                return [float(cell[0]), float(cell[1]), float(cell[2])], lvl, int(cell[3])
        cell = self.q.get("p", {}).get("__pooled__")
        if cell:
            return [float(cell[0]), float(cell[1]), float(cell[2])], "p", int(cell[3])
        return [0.0, 0.0, 0.0], "none", 0

    PHASE_LEVELS = ["m_last_cand_pos", "m_last_cand", "m_last", "m", "p"]

    @staticmethod
    def phase_keys(model: str, last: int, cand: int, pos: int) -> List[str]:
        m = model or "__pooled__"
        return ["%s|%d|%d|%d" % (m, last, cand, pos),
                "%s|%d|%d" % (m, last, cand),
                "%s|%d" % (m, last),
                "%s" % m,
                "__pooled__"]

    def phase_of(self, model: str, last_type: Optional[int], cand: bool,
                 frac: Optional[float]) -> Tuple[int, float]:
        last = -1 if last_type is None else int(last_type)
        keys = self.phase_keys(model, last, int(bool(cand)), pos_bucket(frac))
        for lvl, key in zip(self.PHASE_LEVELS, keys):
            cell = self.phase.get(lvl, {}).get(key)
            if cell and cell[1] >= 20:
                probs = cell[0]
                i = int(np.argmax(probs))
                return i, float(probs[i])
        return 1, 0.0


# ── голова А: события из текста ──────────────────────────────────────────────

def _seg_regex_marks(seg_texts: Sequence[str], detect_events, domain="M"
                     ) -> List[Tuple[int, int]]:
    """Шаблонные срабатывания, привязанные к сегментам ровно так же, как в
    обучающем наборе ученика: склейка сегментов переводом строки, точное
    смещение в символах, двоичный поиск по началам сегментов."""
    offs, pos, parts = [], 0, []
    for s in seg_texts:
        offs.append(pos)
        parts.append(s)
        pos += len(s) + 1
    text = "\n".join(parts)
    out = []
    for e in detect_events(text, domain):
        ty = e.get("type")
        if ty not in TIDX:
            continue
        cp = int(e.get("char_pos", 0))
        seg = bisect.bisect_right(offs, cp) - 1
        seg = min(max(seg, 0), max(len(seg_texts) - 1, 0))
        out.append((seg, TIDX[ty]))
    return sorted(out)


class StudentHead:
    """Ученик судьи: вероятности событий на сегмент. Тяжёлые зависимости
    подгружаются здесь и только здесь."""

    def __init__(self, ckpt: str, emb_model: str = "sentence-transformers/all-MiniLM-L6-v2",
                 device: str = "cpu", model_tag: str = None, thresholds=None):
        import torch  # noqa: F401  (проверка наличия до тяжёлой загрузки)
        from transformers import AutoModel, AutoTokenizer
        import torch.nn as nn

        blob = torch.load(ckpt, map_location="cpu", weights_only=False)
        cfg = blob["config"]
        self.cfg = cfg
        self.tags = list(cfg["tags"])
        self.types = list(cfg["types"])
        self.torch = torch
        h = int(cfg["hidden"])

        class Tagger(nn.Module):
            def __init__(self, d_emb, d_aux, n_models, h, proj=128, d_model_emb=16):
                super().__init__()
                self.proj = nn.Linear(d_emb, proj)
                self.memb = nn.Embedding(n_models, d_model_emb)
                self.gru = nn.GRU(proj + d_aux + d_model_emb, h, batch_first=True,
                                  bidirectional=True)
                self.out = nn.Linear(2 * h, len(M_TYPES))
                self.mbias = nn.Embedding(n_models, len(M_TYPES))

            def forward(self, emb, aux, memb_vec, bias_vec):
                b, t, _ = emb.shape
                me = memb_vec.unsqueeze(0).unsqueeze(0).expand(b, t, -1)
                x = torch.cat([torch.relu(self.proj(emb)), aux, me], dim=-1)
                o, _ = self.gru(x)
                return self.out(o) + bias_vec.unsqueeze(0).unsqueeze(0)

        self.net = Tagger(int(cfg["d_emb"]), int(cfg["d_aux"]),
                          int(cfg["n_models"]), h)
        self.net.load_state_dict(blob["state_dict"], strict=True)
        self.net.eval()
        self.device = device
        self.tok = AutoTokenizer.from_pretrained(emb_model)
        self.emb = AutoModel.from_pretrained(emb_model).to(device).eval()
        self.set_model_tag(model_tag)
        self.thresholds = (np.asarray(thresholds, dtype=np.float64)
                           if thresholds is not None else None)
        self._cache: Dict[str, np.ndarray] = {}

    def set_model_tag(self, model_tag: Optional[str]) -> None:
        """Вектор условия генерации. Для модели вне обучающего списка берётся
        среднее по известным условиям, а поправка порога обнуляется: панель для
        незнакомой модели работает на общем режиме, не притворяясь настроенной."""
        W = self.net.memb.weight.detach()
        B = self.net.mbias.weight.detach()
        if model_tag in self.tags:
            i = self.tags.index(model_tag)
            self.memb_vec, self.bias_vec, self.tag_used = W[i], B[i], model_tag
        else:
            self.memb_vec, self.bias_vec = W.mean(0), B.mean(0) * 0.0
            self.tag_used = "__mean__"

    def _embed(self, texts: List[str]) -> np.ndarray:
        torch = self.torch
        need = [t for t in texts if t not in self._cache]
        for i in range(0, len(need), 64):
            batch = need[i:i + 64]
            enc = self.tok(batch, padding=True, truncation=True, max_length=128,
                           return_tensors="pt").to(self.device)
            with torch.no_grad():
                out = self.emb(**enc).last_hidden_state
            m = enc["attention_mask"].unsqueeze(-1).float()
            v = ((out * m).sum(1) / m.sum(1).clamp(min=1e-9)).cpu().numpy()
            for t, row in zip(batch, v):
                self._cache[t] = row.astype(np.float32)
        return np.stack([self._cache[t] for t in texts], axis=0)

    @staticmethod
    def aux_features(seg_texts: Sequence[str], regex_seg) -> np.ndarray:
        """13 признаков на сегмент, дословно как в обучении ученика."""
        n = len(seg_texts)
        imp = np.zeros((n, K), dtype=np.float32)
        for seg, k in regex_seg:
            if 0 <= seg < n:
                imp[seg, k] += 1.0
        dec = np.zeros_like(imp)
        run = np.zeros(K, dtype=np.float32)
        for t in range(n):
            run = run * 0.7 + imp[t]
            dec[t] = run
        lens = np.array([len(s) for s in seg_texts], dtype=np.float32)
        cum = np.cumsum(lens)
        tot = max(float(cum[-1]) if n else 1.0, 1.0)
        idx = np.arange(n, dtype=np.float32)
        struct = np.stack([
            idx / max(n - 1, 1),
            np.log1p(lens) / 5.0,
            np.full(n, math.log1p(n) / 5.0, dtype=np.float32),
            cum / tot,
            (idx == 0).astype(np.float32),
            (idx == n - 1).astype(np.float32),
        ], axis=1)
        return np.concatenate([dec, imp.sum(1, keepdims=True), struct],
                              axis=1).astype(np.float32)

    def probs(self, seg_texts: Sequence[str], regex_seg) -> np.ndarray:
        torch = self.torch
        if not seg_texts:
            return np.zeros((0, K))
        E = torch.from_numpy(self._embed(list(seg_texts))).unsqueeze(0)
        A = torch.from_numpy(self.aux_features(seg_texts, regex_seg)).unsqueeze(0)
        with torch.no_grad():
            logits = self.net(E, A, self.memb_vec, self.bias_vec)
        return torch.sigmoid(logits)[0].numpy().astype(np.float64)


class EventStream:
    """Поток событий по закрытым сегментам, без двойного счёта.

    Сегмент закрыт, если после него уже есть текст. На каждом пересчёте
    переразмечается весь закрытый префикс, но наружу уходят только события,
    которых раньше не было; последние `lag` сегментов префикса придерживаются,
    потому что у края префикса локальный максимум вероятности возникает как
    краевой эффект и без задержки счётчик раздувается.
    """

    def __init__(self, backend="student", ckpt=None, thresholds=None,
                 model_tag=None, lag=3, recompute_every=1, device="cpu",
                 domain="M"):
        from reasoning_event_detector.patterns import detect_events
        from reasoning_event_detector.segmentation import segment
        self._detect = detect_events
        self._segment = segment
        self.domain = domain
        self.lag = int(lag)
        self.recompute_every = max(int(recompute_every), 1)
        self.backend = backend
        self.thr = np.asarray(thresholds, dtype=np.float64) if thresholds is not None \
            else np.full(K, 0.5)
        self.student = None
        self.scorer = None
        if backend == "student":
            self.student = StudentHead(ckpt, model_tag=model_tag, device=device)
        elif backend == "detector":
            from reasoning_event_detector.scorer import EventScorer
            self.scorer = EventScorer.load().with_mapping("char")
        elif backend == "regex":
            pass
        else:
            raise ValueError("backend: student | detector | regex")
        self.reset()

    def reset(self) -> None:
        self.text = ""
        self._emitted = set()
        self.events: List[Tuple[float, int, int]] = []   # (токен, тип, сегмент)
        self.counts = np.zeros(K, dtype=np.int64)
        self._closed_seen = 0

    def feed(self, text: str, tok_now: float) -> List[Tuple[float, int, int]]:
        """Принять весь накопленный текст рассуждения и текущее число токенов."""
        self.text = text
        segs = self._segment(text)
        n_closed = max(len(segs) - 1, 0)
        if n_closed == 0 or n_closed - self._closed_seen < self.recompute_every:
            return []
        self._closed_seen = n_closed
        return self._decode(segs[:n_closed], tok_now, final=False)

    def finalize(self, tok_now: float) -> List[Tuple[float, int, int]]:
        segs = self._segment(self.text)
        return self._decode(segs, tok_now, final=True)

    def _decode(self, segs, tok_now, final) -> List[Tuple[float, int, int]]:
        n = len(segs)
        if n == 0:
            return []
        marks = _seg_regex_marks(segs, self._detect, self.domain)
        if self.backend == "student":
            P = self.student.probs(segs, marks)
            fired = self._peaks(P)
        elif self.backend == "detector":
            evs, _, _, _ = self.scorer.decode_text("\n".join(segs))
            fired = sorted({(e.seg_id, TIDX[e.type]) for e in evs
                            if e.type in TIDX})
        else:
            fired = marks
        boundary = n if final else max(n - self.lag, 0)
        lens = [max(len(s), 1) for s in segs]
        tot = float(sum(lens))
        cum, acc = [], 0.0
        for L in lens:
            acc += L
            cum.append(acc / tot)
        out = []
        for seg, k in fired:
            if seg >= boundary or (seg, k) in self._emitted:
                continue
            self._emitted.add((seg, k))
            tok = float(tok_now) * cum[seg]
            self.counts[k] += 1
            ev = (tok, int(k), int(seg))
            self.events.append(ev)
            out.append(ev)
        self.events.sort()
        return out

    def _peaks(self, P: np.ndarray, radius: int = 2) -> List[Tuple[int, int]]:
        """Локальные максимумы вероятности выше порога — по одному событию на
        вершину, чтобы плато не превращалось в очередь одинаковых событий."""
        n = P.shape[0]
        out = []
        for k in range(K):
            col = P[:, k]
            for i in range(n):
                lo, hi = max(0, i - radius), min(n, i + radius + 1)
                if col[i] >= self.thr[k] and col[i] >= col[lo:hi].max() - 1e-12:
                    out.append((i, k))
        return sorted(out)


# ── сборка состояния панели ──────────────────────────────────────────────────

class Panel:
    """Состояние панели: три головы, один словарь наружу."""

    SPARK_BINS = 24

    def __init__(self, hawkes_pack=None, cycle_pack=None, model=None,
                 events_backend="student", student_ckpt=None, thresholds=None,
                 model_tag=None, device="cpu", pack_dir=DEFAULT_PACK_DIR):
        hp = hawkes_pack or os.path.join(pack_dir, "hawkes_pack.json")
        cp = cycle_pack or os.path.join(pack_dir, "cycle_pack.json")
        self.hawkes_pack = json.load(open(hp, encoding="utf-8"))
        self.cycle = CycleTable.load(cp)
        self.model = model if model in self.hawkes_pack else "__pooled__"
        self.hawkes = HawkesHead.from_pack(self.hawkes_pack, self.model)
        self.table_model = model if self._has_table(model) else "__pooled__"
        thr = thresholds
        if thr is None:
            sp = os.path.join(pack_dir, "student_pack.json")
            table = json.load(open(sp, encoding="utf-8")) if os.path.exists(sp) else {}
            thr = (table.get("thresholds", {}).get(model_tag or "") or
                   table.get("thresholds", {}).get(model or "") or
                   table.get("thresholds", {}).get("__pooled__"))
        self.stream = EventStream(backend=events_backend, ckpt=student_ckpt,
                                  thresholds=thr, model_tag=model_tag,
                                  device=device)
        self.reset()

    def _has_table(self, model) -> bool:
        return bool(model) and ("%s" % model) in self.cycle.q.get("m", {})

    def reset(self) -> None:
        self.hawkes.reset()
        self.stream.reset()
        self.tok = 0.0
        self.cycle_start = 0.0
        self.cycle_events = 0
        self.last_event: Optional[Tuple[float, int]] = None
        self.cand_tok: Optional[float] = None
        self.spark = [0] * self.SPARK_BINS
        self._spark_tok = 0.0
        self.n_updates = 0

    # -- вход -----------------------------------------------------------------
    def feed(self, text: str, tok: float, final: bool = False) -> dict:
        self.tok = float(tok)
        new = self.stream.finalize(tok) if final else self.stream.feed(text, tok)
        for t, k, _seg in new:
            self.hawkes.observe(t, k)
            self.cycle_events += 1
            self.last_event = (t, k)
            self._bump_spark(t)
            if k == COMMIT:
                self.cycle_start = t
                self.cycle_events = 0
        if self.cand_tok is None:
            self.cand_tok = self._first_candidate(text, tok)
        self.hawkes.advance(self.tok)
        self.n_updates += 1
        return self.state()

    def _bump_spark(self, t: float) -> None:
        if self.tok <= 0:
            return
        i = min(int(self.SPARK_BINS * t / max(self.tok, 1.0)), self.SPARK_BINS - 1)
        self.spark[i] += 1

    @staticmethod
    def _first_candidate(text: str, tok: float) -> Optional[float]:
        from reasoning_event_detector.candidate import first_candidate
        c = first_candidate(text)
        if c is None or not text:
            return None
        return float(tok) * (c.char_pos / max(len(text), 1))

    # -- выход ----------------------------------------------------------------
    def state(self) -> dict:
        probs = self.hawkes.next_probs()
        lo, mid, hi = self.hawkes.wait_quantiles((0.1, 0.5, 0.9))
        age = max(self.tok - self.cycle_start, 0.0)
        last_k = None if self.last_event is None else self.last_event[1]
        ph_i, ph_p = self.cycle.phase_of(self.table_model, last_k,
                                         self.cand_tok is not None, None)
        qs, level, n_cell = self.cycle.lookup(
            self.table_model, ph_i, self.cycle_events, age,
            self.cand_tok is not None)
        order = list(np.argsort(-probs))
        st = {
            "tok": round(self.tok, 1),
            "phase": PHASE_WORDS[OPERATORS[ph_i]],
            "phase_conf": round(ph_p, 3),
            "last_event": (None if self.last_event is None else {
                "type": EVENT_WORDS[M_TYPES[self.last_event[1]]],
                "tok": round(self.last_event[0], 1)}),
            "next_probs": {EVENT_WORDS[M_TYPES[i]]: round(float(probs[i]), 4)
                           for i in order},
            "next_order": [EVENT_WORDS[M_TYPES[i]] for i in order],
            "tok_to_event": [round(lo, 1), round(hi, 1)],
            "tok_to_event_q50": round(mid, 1),
            "tok_to_cycle_end": [round(q, 1) for q in qs],
            "cycle": {"start_tok": round(self.cycle_start, 1),
                      "age_tok": round(age, 1),
                      "n_events": int(self.cycle_events),
                      "table_level": level, "table_n": n_cell},
            "post_candidate": (None if self.cand_tok is None else {
                "tok": round(self.tok - self.cand_tok, 1),
                "share": round((self.tok - self.cand_tok) / max(self.tok, 1.0), 3)}),
            "density_spark": list(self.spark),
            "counts": {EVENT_WORDS[M_TYPES[i]]: int(self.stream.counts[i])
                       for i in range(K)},
            "n_events": int(self.stream.counts.sum()),
            "model": self.model,
        }
        return st


def load_packs(pack_dir=DEFAULT_PACK_DIR):
    return (json.load(open(os.path.join(pack_dir, "hawkes_pack.json"), encoding="utf-8")),
            CycleTable.load(os.path.join(pack_dir, "cycle_pack.json")))
