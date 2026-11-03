#!/usr/bin/env python3
"""gate common v3 for the gpt-oss effort controller."""
from __future__ import annotations

import json
import math
import os
import re

import numpy as np

TYPES = ["backtrack", "verify", "branch", "subgoal_done", "failed_attempt",
         "commit"]
T_BACK, T_VER, T_BR, T_SUB, T_FAIL, T_COMMIT = range(6)

EFFORTS = ("normal", "low")
ARM_OF_EFFORT = {"normal": "none", "low": "effort_low"}


# ── прогноз стоимости следующего цикла ───────────────────────────────────────

class ForestForecaster:
    """Квантильный градиентный бустинг задания 12 в виде массивов.

    Обучение шло на log1p стоимости, поэтому наружу отдаётся expm1 предсказания.
    Квантили сортируются: у разных квантильных моделей пересечения возможны.
    """

    def __init__(self, npz_path: str, json_path: str):
        self.meta = json.load(open(json_path, encoding="utf-8"))
        z = np.load(npz_path)
        self.qs = [float(q) for q in self.meta["quantiles"]]
        self.features = list(self.meta["features"])
        self.models = list(self.meta["models"])
        self.arms = list(self.meta["arms"])
        self.packs = []
        for j in range(len(self.qs)):
            m = z["q%d_meta" % j]
            self.packs.append({
                "children_left": z["q%d_children_left" % j],
                "children_right": z["q%d_children_right" % j],
                "feature": z["q%d_feature" % j],
                "threshold": z["q%d_threshold" % j],
                "value": z["q%d_value" % j],
                "offsets": z["q%d_offsets" % j],
                "lr": float(m[0]), "base": float(m[1]),
            })

    @classmethod
    def load(cls, out_dir: str) -> "ForestForecaster":
        return cls(os.path.join(out_dir, "cnext_forest.npz"),
                   os.path.join(out_dir, "cnext_forest.json"))

    @staticmethod
    def _one(pack, x):
        cl, cr = pack["children_left"], pack["children_right"]
        fe, th, va = pack["feature"], pack["threshold"], pack["value"]
        off = pack["offsets"]
        acc = pack["base"]
        for t in range(len(off) - 1):
            o = int(off[t])
            node = 0
            while fe[o + node] >= 0:
                node = (cl[o + node] if x[fe[o + node]] <= th[o + node]
                        else cr[o + node])
            acc += pack["lr"] * va[o + node]
        return acc

    def ladder(self, x: np.ndarray) -> list[float]:
        """Лестница квантилей стоимости следующего цикла в токенах."""
        raw = [float(np.expm1(self._one(p, x))) for p in self.packs]
        return sorted(max(v, 0.0) for v in raw)

    def at(self, lad: list[float], q: float) -> float:
        return float(np.interp(q, self.qs, lad))


class VerifyTable:
    """Цена проверочного цикла и безусловная таблица стоимости цикла."""

    def __init__(self, path: str):
        blob = json.load(open(path, encoding="utf-8"))
        self.cv = blob["c_verify"]
        self.cu = blob["c_uncond"]

    def verify(self, model: str, q: float) -> float:
        lad = self.cv.get(model) or self.cv["__pooled__"]
        return float(np.interp(q, self.cv["qs"], lad))

    def uncond(self, model: str, q: float) -> float:
        lad = self.cu.get(model) or self.cu["__pooled__"]
        return float(np.interp(q, self.cu["qs"], lad))


# ── голова трудности вопроса ────────────────────────────────────────────────

DIFF_FEATURES = ["log_chars", "log_words", "digit_share", "dollar_count",
                 "newline_count", "mean_word_len", "has_choices",
                 "nonascii_share", "question_marks"]


def difficulty_x(text: str) -> np.ndarray:
    """Дешёвые признаки вопроса: только текст, никакой генерации."""
    t = text or ""
    n = max(len(t), 1)
    words = t.split()
    nw = max(len(words), 1)
    digits = sum(c.isdigit() for c in t)
    nonascii = sum(1 for c in t if ord(c) > 127)
    has_choices = 1.0 if re.search(r"\(\s*[A-D]\s*\)|\n\s*[A-D][\).]\s", t) else 0.0
    return np.asarray([
        math.log1p(len(t)), math.log1p(len(words)), digits / n,
        float(t.count("$")), float(t.count("\n")),
        sum(len(w) for w in words) / nw, has_choices,
        nonascii / n, float(t.count("?")),
    ], dtype=np.float64)


class DifficultyHead:
    """Множитель трудности вопроса к прогнозной стоимости.

    Наружу отдаётся m(x) = exp(mu(x) - mu_ref): во сколько раз этот вопрос
    дороже среднего вопроса обучающего источника. Абсолютные токены отсюда не
    берутся: масштаб даёт прогноз стоимости цикла, обученный в домене живого
    контура.
    """

    def __init__(self, path: str):
        d = json.load(open(path, encoding="utf-8"))
        self.w = np.asarray(d["w"], float)
        self.sources = list(d["sources"])
        self.ref_source = d["ref_source"]
        self.mu_ref = float(d["mu_ref"])
        self.nfeat = len(d["features"])
        self.meta = {k: v for k, v in d.items() if k != "w"}

    def mult(self, text: str) -> float:
        x = difficulty_x(text)
        mu = float(x @ self.w[:self.nfeat]
                   + self.w[self.nfeat + self.sources.index(self.ref_source)])
        return float(math.exp(mu - self.mu_ref))


# ── состояние цикла для признаков прогноза ───────────────────────────────────

FEATURE_ORDER = [
    "cycle_index", "t_start", "log_t_start", "cap_known", "cap_value",
    "frac_budget", "budget_left", "log_budget_left", "has_cand", "since_cand",
    "prev_len", "log_prev_len", "prev_len2", "mean_prev_len",
    "log_mean_prev_len", "max_prev_len", "min_prev_len", "std_prev_len",
    "n_ev_total", "n_ev_prev_cycle", "ev_rate_total", "ev_rate_prev_cycle",
    "n_ev_last200", "n_ev_last500", "n_seg_so_far", "mean_seg_len",
    "last_seg_len", "gap_last_ev", "mean_commit_gap", "last_ev_type",
    "prev_last_noncommit",
] + ["cnt_tot_%s" % t for t in TYPES] + ["cnt_prev_%s" % t for t in TYPES] \
  + ["model", "arm"]


def build_features(events, seg_tok, cand_tok, cap, cycle_bounds,
                   model_idx, arm_idx):
    """Признаки на старте цикла по правилу build_cycles.rows_of_trace.

    Момент оценки один и тот же для всех признаков — начало цикла, то есть
    последний элемент cycle_bounds.
    """
    i = max(len(cycle_bounds) - 1, 0)
    t = float(cycle_bounds[i]) if cycle_bounds else 0.0
    seen = [(float(s), int(k)) for (s, k) in events if s <= t]
    seen.sort()
    lo_prev = cycle_bounds[i - 1] if i > 0 else 0.0
    t_cycle_start = t

    cnt_tot = [0] * len(TYPES)
    for (_s, k) in seen:
        cnt_tot[k] += 1
    cnt_prev = [0] * len(TYPES)
    n_ev_prev = 0
    if i > 0:
        for (s, k) in seen:
            if lo_prev < s <= t_cycle_start:
                cnt_prev[k] += 1
                n_ev_prev += 1
    n200 = sum(1 for (s, _k) in seen if s > t - 200.0)
    n500 = sum(1 for (s, _k) in seen if s > t - 500.0)
    last_ev_type = seen[-1][1] if seen else -1
    prev_noncommit = -1
    for (_s, k) in reversed(seen):
        if k != T_COMMIT:
            prev_noncommit = k
            break
    gap_last_ev = (t - seen[-1][0]) if seen else t

    segs = [float(x) for x in seg_tok if 0.0 < float(x) <= t]
    ns = len(segs)
    mean_seg = (t / ns) if ns > 0 else -1.0
    last_seg = (segs[-1] - segs[-2]) if ns >= 2 else (segs[0] if ns == 1
                                                      else -1.0)

    lens = [cycle_bounds[j + 1] - cycle_bounds[j] for j in range(i)]
    prev_len = lens[-1] if lens else -1.0
    prev_len2 = lens[-2] if len(lens) >= 2 else -1.0
    mean_prev = float(np.mean(lens)) if lens else -1.0
    max_prev = float(np.max(lens)) if lens else -1.0
    min_prev = float(np.min(lens)) if lens else -1.0
    std_prev = float(np.std(lens)) if len(lens) >= 2 else -1.0

    commit_ts = [c for c in cycle_bounds[1:] if c <= t]
    mean_cg = (commit_ts[-1] / len(commit_ts)) if commit_ts else -1.0

    cap_known = 1.0 if cap else 0.0
    cap_v = float(cap) if cap else 0.0
    tc = float(t_cycle_start)
    frac_b = (tc / cap_v) if cap else -1.0
    left_b = (cap_v - tc) if cap else -1.0

    has_cand = 1.0 if (cand_tok is not None and float(cand_tok) <= tc) else 0.0
    since_cand = (tc - float(cand_tok)) if has_cand else -1.0

    f = [
        float(i), tc, float(np.log1p(tc)), cap_known, cap_v, frac_b,
        left_b, float(np.log1p(max(left_b, 0.0))) if cap else -1.0,
        has_cand, since_cand,
        prev_len, float(np.log1p(prev_len)) if prev_len >= 0 else -1.0,
        prev_len2, mean_prev,
        float(np.log1p(mean_prev)) if mean_prev >= 0 else -1.0,
        max_prev, min_prev, std_prev,
        float(len(seen)), float(n_ev_prev),
        float(len(seen)) / max(tc, 1.0),
        (float(n_ev_prev) / max(tc - lo_prev, 1.0)) if i > 0 else -1.0,
        float(n200), float(n500),
        float(ns), mean_seg, float(last_seg), gap_last_ev, mean_cg,
        float(last_ev_type), float(prev_noncommit),
    ]
    f += [float(x) for x in cnt_tot]
    f += [float(x) for x in cnt_prev]
    f += [float(model_idx), float(arm_idx)]
    assert len(f) == len(FEATURE_ORDER), (len(f), len(FEATURE_ORDER))
    return np.asarray(f, dtype=np.float64)


def verified_at(events, cand_tok, t) -> bool:
    """Проверенный кандидат: проверка после кандидата без последующей неудачи
    и без нового ветвления. Определение дословно из переигровки задания 12."""
    if cand_tok is None or cand_tok > t:
        return False
    last_ver = None
    for (s, k) in sorted((float(a), int(b)) for (a, b) in events):
        if s > t:
            break
        if s < cand_tok:
            continue
        if k == T_VER:
            last_ver = s
        elif k in (T_FAIL, T_BR):
            last_ver = None
    return last_ver is not None


# ── правило политики v2 и v2.1 (переносится дословно) ────────────────────────

VERSIONS = ("v2", "v2.1")


def policy_v2(brem: float, W: float, A: bool, V: bool, cnext: float,
              cverify: float, version: str = "v2.1") -> dict:
    assert version in VERSIONS, "неизвестная версия политики %r" % version
    if brem <= W:
        return {"act": "finalize", "rule": "4", "mode": "none",
                "allow": 0.0, "terminal": True}
    room = brem - W
    if not A:
        if room >= cnext:
            return {"act": "continue", "rule": "1a", "mode": "normal",
                    "allow": room, "terminal": False}
        return {"act": "continue", "rule": "1b", "mode": "constrained",
                "allow": room, "terminal": False}
    if not V:
        if room >= cnext + cverify:
            return {"act": "continue", "rule": "2a", "mode": "normal",
                    "allow": room, "terminal": False}
        if room >= cverify:
            return {"act": "continue", "rule": "2b", "mode": "verify",
                    "allow": min(room, math.ceil(cverify)), "terminal": True}
        return {"act": "finalize", "rule": "2c", "mode": "none",
                "allow": 0.0, "terminal": True}
    if version == "v2":
        return {"act": "finalize", "rule": "3", "mode": "none",
                "allow": 0.0, "terminal": True}
    if room >= cnext:
        return {"act": "continue", "rule": "3a", "mode": "improve",
                "allow": room, "terminal": False}
    return {"act": "finalize", "rule": "3b", "mode": "none",
            "allow": 0.0, "terminal": True}


# ── политика v3 ──────────────────────────────────────────────────────────────

# значения по умолчанию перекрываются предрегистрацией арма
V3_DEFAULTS = {
    "thr_branch": 0.1297,    # доля ветвления в ближайшем событии для ветки стопа
    "thr_commit": 0.2620,    # доля консолидации для ветки короткого хвоста
    "kappa_commit": 2.0,     # во сколько раз хвост длиннее квантили ожидания
    "min_tail": 32.0,        # минимальная длина короткого хвоста в токенах
    "kappa_start": 2.0,      # порог выбора усилия в стоимостях цикла
    "allow_switch": True,    # разрешена ли смена усилия на границе цикла
    "use_next_type": True,   # разрешены ли ветки по ведущему типу события
}


def tau_of(c0: float, mult: float, kappa_start: float) -> float:
    """Порог достаточности бюджета: kappa_start прогнозных стоимостей цикла на
    этом вопросе. Стоимость цикла берётся у прогнозной головы в состоянии
    начала трассы при данном бюджете и данном уровне усилия и умножается на
    множитель трудности вопроса.

    Эта величина живёт на масштабе корпуса (сотни токенов), в отличие от
    поточечного прогноза стоимости следующего цикла, который после появления
    кандидата отвечает на другой вопрос — сколько токенов до ближайшей
    консолидации — и в живом контуре имеет медиану около десяти токенов.
    """
    return float(kappa_start) * float(c0) * float(mult)


def start_effort(budget: float, W: float, c0: float, mult: float,
                 kappa_start: float) -> tuple[str, dict]:
    """Выбор уровня усилия до начала генерации.

    Сравниваются остаток бюджета за вычетом резерва довершения и порог
    достаточности при обычном усилии. Если бюджета не хватает, трасса
    стартует с низким усилием.
    """
    room = float(budget) - float(W)
    need = tau_of(c0, mult, kappa_start)
    eff = "low" if room < need else "normal"
    return eff, {"room": round(room, 1), "c0": round(float(c0), 1),
                 "mult": round(float(mult), 3), "need": round(need, 1)}


def policy_v3(brem: float, W: float, A: bool, V: bool, effort: str,
              cnext_cur: float, cverify: float, p_branch: float,
              p_commit: float, wait_q: float, tau_normal: float,
              tau_low: float, params: dict) -> dict:
    """Одно решение политики v3.

    Порядок веток заморожен предрегистрацией и здесь ему следует буквально.
    Возвращается действие, режим цикла, кап цикла, ветка правила, уровень
    усилия, с которым цикл пойдёт дальше, и признак срабатывания смены усилия.

    Смена усилия устроена не отдельной веткой, а поправкой перед выбором
    ветки: усилие снижается ровно тогда, когда остатка не хватает на работу
    при обычном усилии, но хватает при низком. Ниже этой полосы снижать
    усилие поздно, там работают ветки остановки. Полоса существует только
    потому, что у прогнозной головы есть ключ усилия: без него оба порога
    совпадают и смена не срабатывает никогда.
    """
    p = dict(V3_DEFAULTS)
    p.update(params or {})
    nxt = bool(p["use_next_type"])
    br = nxt and (p_branch >= p["thr_branch"])
    cm = nxt and (p_commit >= p["thr_commit"])
    tail = max(float(p["min_tail"]), float(p["kappa_commit"]) * float(wait_q))

    if brem <= W:
        return {"act": "finalize", "rule": "4", "mode": "none", "allow": 0.0,
                "terminal": True, "effort": effort, "switched": False,
                "tight": False}
    room = brem - W
    switched = bool(p["allow_switch"] and effort == "normal"
                    and tau_low <= room < tau_normal)
    eff_after = "low" if switched else effort
    tight = bool(room < tau_low)

    def cont(rule, mode, allow, terminal=False):
        return {"act": "continue", "rule": rule, "mode": mode,
                "allow": float(allow), "terminal": bool(terminal),
                "effort": eff_after, "switched": switched, "tight": tight}

    def stop(rule):
        return {"act": "finalize", "rule": rule, "mode": "none", "allow": 0.0,
                "terminal": True, "effort": effort, "switched": False,
                "tight": tight}

    if not A:
        # докандидатную зону не режем никогда: стопа в этой ветке нет
        if room >= cnext_cur:
            return cont("1a", "normal", room)
        return cont("1b", "constrained", room)

    if not V:
        if room >= cnext_cur + cverify:
            return cont("2a", "normal", room)
        if br and tight:
            return stop("2e_br")
        if room >= cverify:
            return cont("2b", "verify", min(room, math.ceil(cverify)),
                        terminal=True)
        if cm and room >= tail:
            return cont("2d_cm", "commit_tail", min(room, math.ceil(tail)),
                        terminal=True)
        return stop("2c")

    if br and tight:
        return stop("3e_br")
    if cm and tight and room >= tail:
        return cont("3d_cm", "commit_tail", min(room, math.ceil(tail)),
                    terminal=True)
    if room >= cnext_cur:
        return cont("3a", "improve", room)
    return stop("3b")


# какие ветки принадлежат какой голове: нужно разбору вкладов
RULES_NEXT_TYPE = ("2e_br", "3e_br", "2d_cm", "3d_cm")
RULES_LIMITING = ("1b", "2b", "2c", "2d_cm", "2e_br", "3", "3b", "3d_cm",
                  "3e_br", "4")
