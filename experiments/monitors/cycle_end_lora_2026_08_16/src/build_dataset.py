"""Этап 1: сборка датасета «префикс -> остаток до конца текущего цикла».

Реализует раздел 5 предрегистрации эксперимента cycle_end_lora_2026_08_16.
Читает пять корпусов reasoner-разметки, строит токенную ось, выбирает точки
наблюдения, считает цель, признаки события-истории для баров и хоуковский блок
для инъекции. Пишет по файлу на часть разбиения.

Запуск на боксе 192.168.2.48, системный python3, нужен только numpy.
"""

import argparse
import glob
import hashlib
import json
import math
import os
import random
import re
import sys

import numpy as np

M_TYPES = ["backtrack", "verify", "branch", "subgoal_done", "failed_attempt", "commit"]
TIDX = {t: i for i, t in enumerate(M_TYPES)}
OPERATORS = ["SETUP", "DERIVING", "EXPLORING", "VERIFYING", "REVISING", "CONCLUDING"]
OPIDX = {o: i for i, o in enumerate(OPERATORS)}
COMMIT = TIDX["commit"]

SEED = 20260816
H_HORIZON = 256.0


def draft_root():
    """Корень чернового рабочего репозитория.

    В записях происхождения разметки путь к исходному файлу записан абсолютным,
    и на каждой машине он свой. В файлах репозитория абсолютных путей машин не
    держат, поэтому корень берётся из переменной среды DRAFT_ROOT либо из файла
    draft_root в каталоге настроек, который лежит вне git. Если настройки нет,
    приставка считается пустой, и путь из записи используется как есть.
    """
    from_env = os.environ.get("DRAFT_ROOT", "")
    if from_env:
        return from_env
    settings = os.environ.get(
        "REASONING_DATA_SETTINGS",
        os.path.join(os.path.expanduser("~"), ".config", "reasoning-data"))
    path = os.path.join(settings, "draft_root")
    if os.path.isfile(path):
        return open(path, encoding="utf-8").read().strip()
    return ""


DRAFT_ROOT = draft_root()
BUCKET_EDGES = [0.0, 16.0, 32.0, 64.0, 128.0, 256.0, 512.0, 1024.0, 2048.0, 4096.0, 8192.0]
BUCKET_LABELS = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K"]
BETA = 1.0 / 140.0
MAX_POINTS_PER_TRACE = 3
N_CTX_SEG = 12
N_EV_HIST = 12
SEG_CHARS = 400
Q_CHARS = 400

LAYERS = [
    ("native_po", "native_po_2026_08_11/judge_labels_r1", 1),
    ("effort_repeats", "effort_repeats_2026_08_14/judge_labels_r1", 3),
    ("traces_30b", "traces_30b_2026_08_14/judge_labels_r1_30b", 1),
    ("distill_vs_self", "distill_vs_self_2026_08_13/judge_labels_r1", 1),
    ("event_span_dispersion", "event_span_dispersion_2026_08_14/labels_r1_bestofn", 0),
]

CAND_PATTERNS = [
    re.compile(r"\\boxed\{"),
    re.compile(r"(?<!to )(?<!To )\b(?:the\s+)?(?:final\s+)?(?:answer|result)\s*(?:is\b|:|=)",
               re.IGNORECASE),
    re.compile(r"\bответ\b", re.IGNORECASE),
    re.compile(r"=\s*(?:-?\d[\d,./]*|\(?[A-E]\)?)\s*(?:[.!?]|\n|$)"),
    re.compile(r"\banswer\s+\(?([A-E])\)?(?:[.!?\s]|$)"),
]


def split_of_question(qid, salt="sft_judge_distill", frac_test=0.30, frac_val=0.15):
    h = hashlib.sha1((salt + "::" + str(qid)).encode("utf-8")).hexdigest()
    u = int(h[:12], 16) / float(1 << 48)
    if u < frac_test:
        return "test"
    if u < frac_test + frac_val:
        return "val"
    return "train"


def token_axis(seg_texts, token_total):
    lens = [max(len(s), 1) for s in seg_texts]
    total_chars = float(sum(lens))
    tt = float(token_total or 0)
    if tt <= 0:
        tt = total_chars / 4.0
    ends, acc = [], 0.0
    for L in lens:
        acc += L
        ends.append(acc / total_chars * tt)
    return ends, tt


def candidate_token(seg_texts, ends):
    text = "\n".join(seg_texts)
    best = None
    for rx in CAND_PATTERNS:
        m = rx.search(text)
        if m and (best is None or m.start() < best):
            best = m.start()
    if best is None:
        return None
    pos, cp = 0, int(best)
    for i, s in enumerate(seg_texts):
        pos += len(s) + 1
        if pos > cp:
            return ends[i]
    return ends[-1]


def seg_operators(spans, n):
    ops = [-1] * n
    for sp in spans or []:
        try:
            a, b = int(sp["seg_start"]), int(sp["seg_end"])
            o = OPIDX.get(sp.get("operator"), -1)
        except Exception:
            continue
        for i in range(max(0, a), min(n, b)):
            ops[i] = o
    return ops


def norm_model(m):
    if not m:
        return "?"
    m = str(m)
    if "/" in m and m.startswith("/"):
        m = os.path.basename(m).replace("_", "/", 1)
    return m


def group_cut(model, effort, arm):
    """Ключ разреза матриц возбуждения по группе модель плюс усилие."""
    m = model.lower()
    e = (effort or "").lower()
    a = (arm or "").lower()
    if "gpt-oss-20b" in m:
        if e in ("low", "medium", "high"):
            return "v2_gptoss20_" + e, "group"
        if "effort_low" in a:
            return "r1_gptoss20_effort_low", "group"
        if "effort_high" in a:
            return "r1_gptoss20_effort_high", "group"
        return "r1_gptoss20", "group"
    if "qwen3.6-35b" in m:
        if e in ("low", "medium", "high"):
            return "v2_qwen36_35b_" + e, "group"
        return "v2_qwen36_35b_medium", "group"
    return "pooled", "pooled"


def load_matrices(path):
    d = json.load(open(path))
    lin = d["linear"]
    cuts = {}
    for k, v in lin.items():
        if "A" not in v or "mu" not in v:
            continue
        cuts[k] = {
            "A": np.asarray(v["A"], dtype=float),
            "mu": np.asarray(v["mu"], dtype=float),
            "rho": float(v.get("rho", float("nan"))),
            "sum_A": float(v.get("sum_A", float("nan"))),
            "n_events": float(v.get("n_events", 0.0)),
        }
    main = ["v2_gptoss20_low", "v2_gptoss20_medium", "v2_gptoss20_high",
            "v2_qwen36_35b_low", "v2_qwen36_35b_medium", "v2_qwen36_35b_high"]
    have = [c for c in main if c in cuts]
    w = np.asarray([cuts[c]["n_events"] for c in have], dtype=float)
    w = w / w.sum()
    A = sum(wi * cuts[c]["A"] for wi, c in zip(w, have))
    mu = sum(wi * cuts[c]["mu"] for wi, c in zip(w, have))
    cuts["pooled"] = {"A": A, "mu": mu,
                      "rho": float(np.max(np.abs(np.linalg.eigvals(A)))),
                      "sum_A": float(A.sum()), "n_events": float(w.sum())}
    return cuts, d.get("types", M_TYPES), float(d.get("frozen_decay_tokens", 140.0))


def load_layer(data_root, rel, depth):
    root = os.path.join(data_root, rel)
    pat = os.path.join(root, *(["*"] * depth), "*.json")
    out = []
    for f in sorted(glob.glob(pat)):
        if os.path.basename(f).startswith("_"):
            continue
        try:
            lab = json.load(open(f))
        except Exception:
            continue
        prov = lab.get("_provenance", {}) or {}
        sp = prov.get("source_payload")
        if not sp:
            continue
        sp = sp.replace(os.path.join(DRAFT_ROOT, "internal_signals_poc", ""),
                        os.path.join(data_root, ""))
        if not os.path.exists(sp):
            continue
        try:
            pay = json.load(open(sp))
        except Exception:
            continue
        out.append((lab, pay, f))
    return out


def make_record(lab, pay):
    segs = pay.get("segments") or []
    if len(segs) < 3:
        return None
    seg_texts = [s.get("text", "") for s in segs]
    n = len(seg_texts)
    ends, T = token_axis(seg_texts, pay.get("token_total"))
    spans = lab.get("spans") or []
    if not spans:
        return None
    ops = seg_operators(spans, n)
    evs = []
    for e in lab.get("events", []):
        t = e.get("type")
        if t not in TIDX:
            continue
        s = e.get("seg_id")
        try:
            s = int(s)
        except Exception:
            continue
        if not (0 <= s < n):
            continue
        evs.append((ends[s], TIDX[t], s))
    evs.sort()
    meta = lab.get("_meta", {}) or {}
    model = norm_model(meta.get("trace_model") or pay.get("model"))
    return {
        "qid": lab.get("question_id") or pay.get("question_id"),
        "model": model,
        "effort": meta.get("effort_level") or "",
        "arm": meta.get("arm") or pay.get("arm") or "",
        "tag": meta.get("tag") or pay.get("tag") or "",
        "benchmark": meta.get("benchmark") or pay.get("benchmark") or "",
        "cap": pay.get("cap"),
        "seed": meta.get("seed"),
        "n_seg": n,
        "T": T,
        "seg_ends": ends,
        "seg_texts": seg_texts,
        "seg_op": ops,
        "events": evs,
        "cand_tok": candidate_token(seg_texts, ends),
        "question": pay.get("question") or "",
    }


def bucket_of(rem):
    for i in range(len(BUCKET_EDGES) - 1, -1, -1):
        if rem >= BUCKET_EDGES[i]:
            return i
    return 0


def intensities(evs_before, t, A, mu):
    lam = mu.copy()
    for (te, ke, _s) in evs_before:
        lam += A[ke] * BETA * math.exp(-BETA * max(0.0, t - te))
    return lam


def features_at(rec, i_seg, evs_before, lam, rho):
    """Признаки состояния для событийных баров. Только прошлое."""
    ends = rec["seg_ends"]
    t = ends[i_seg]
    cap = rec["cap"] or 0
    cap_known = 1.0 if rec["cap"] else 0.0
    n_ev = len(evs_before)
    cnt = [0.0] * 6
    last_t = [-1.0] * 6
    for (te, ke, _s) in evs_before:
        cnt[ke] += 1.0
        last_t[ke] = te
    tsl = t - evs_before[-1][0] if n_ev else t
    since_type = [(t - last_t[k]) if last_t[k] >= 0 else -1.0 for k in range(6)]
    n200 = sum(1.0 for (te, _k, _s) in evs_before if t - te <= 200.0)
    n500 = sum(1.0 for (te, _k, _s) in evs_before if t - te <= 500.0)
    commits = [te for (te, ke, _s) in evs_before if ke == COMMIT]
    n_cyc = len(commits)
    since_commit = t - commits[-1] if commits else t
    if n_cyc >= 1:
        starts = [0.0] + commits[:-1]
        cyc_len = [c - s for c, s in zip(commits, starts)]
        mean_cyc = float(np.mean(cyc_len))
        last_cyc = float(cyc_len[-1])
    else:
        mean_cyc, last_cyc = -1.0, -1.0
    ev_in_cyc = sum(1.0 for (te, _k, _s) in evs_before if te > (commits[-1] if commits else -1))
    cand = rec["cand_tok"]
    has_cand = 1.0 if (cand is not None and cand <= t) else 0.0
    since_cand = (t - cand) if has_cand else -1.0
    seg_len = ends[i_seg] - (ends[i_seg - 1] if i_seg else 0.0)
    f = [t, math.log1p(t), cap_known, float(cap), (t / cap) if cap else -1.0,
         float(n_ev), n_ev / max(t, 1.0) * 1000.0, tsl, n200, n500,
         float(n_cyc), since_commit, mean_cyc, last_cyc, ev_in_cyc,
         has_cand, since_cand, seg_len, float(i_seg), float(rec["seg_op"][i_seg])]
    f += cnt
    f += since_type
    return f, list(lam), rho


FEAT_BASE = ["t_tok", "log_t", "cap_known", "cap_value", "frac_cap",
             "n_ev", "ev_rate_1k", "since_last_ev", "n_ev_200", "n_ev_500",
             "n_cycles", "since_commit", "mean_cycle_len", "last_cycle_len",
             "n_ev_in_cycle", "has_cand", "since_cand", "seg_len", "seg_index",
             "seg_operator"] + \
            ["cnt_" + t for t in M_TYPES] + ["since_" + t for t in M_TYPES]
FEAT_HAWKES = ["lam_" + t for t in M_TYPES] + ["rho_group"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True, help="каталог с корпусами, слепок isp")
    ap.add_argument("--matrices", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-points", type=int, default=MAX_POINTS_PER_TRACE)
    a = ap.parse_args()

    rng = random.Random(SEED)
    cuts, types, decay = load_matrices(a.matrices)
    assert list(types) == M_TYPES, "порядок типов в матрицах разошёлся с каноном"
    os.makedirs(a.out, exist_ok=True)

    fh = {s: open(os.path.join(a.out, "windows_%s.jsonl" % s), "w") for s in
          ("train", "val", "test")}
    stats = {"layers": {}, "splits": {"train": 0, "val": 0, "test": 0},
             "traces": 0, "traces_by_split": {"train": 0, "val": 0, "test": 0},
             "windows_by_bucket": {b: 0 for b in BUCKET_LABELS},
             "matrix_source": {"group": 0, "pooled": 0},
             "models": {}, "rem_quantiles": {}, "n_events_total": 0}
    rems = []

    for name, rel, depth in LAYERS:
        rows = load_layer(a.data_root, rel, depth)
        n_tr, n_w = 0, 0
        for lab, pay, _f in rows:
            rec = make_record(lab, pay)
            if rec is None:
                continue
            n_tr += 1
            stats["traces"] += 1
            stats["n_events_total"] += len(rec["events"])
            sp = split_of_question(rec["qid"])
            stats["traces_by_split"][sp] += 1
            stats["models"][rec["model"]] = stats["models"].get(rec["model"], 0) + 1
            cut, src = group_cut(rec["model"], rec["effort"], rec["arm"])
            cutd = cuts.get(cut) or cuts["pooled"]
            if cut not in cuts:
                src = "pooled"
            A, mu, rho = cutd["A"], cutd["mu"], cutd["rho"]
            ends = rec["seg_ends"]
            commits = [te for (te, ke, _s) in rec["events"] if ke == COMMIT]
            cand_idx = list(range(2, rec["n_seg"] - 1))
            if not cand_idx:
                continue
            rng.shuffle(cand_idx)
            take = cand_idx[:a.max_points]
            for i_seg in sorted(take):
                t = ends[i_seg]
                nxt = [c for c in commits if c > t]
                end = nxt[0] if nxt else rec["T"]
                rem = end - t
                if rem < 0:
                    continue
                evs_before = [e for e in rec["events"] if e[0] <= t]
                lam = intensities(evs_before, t, A, mu)
                fb, lamv, rhov = features_at(rec, i_seg, evs_before, lam, rho)
                b = bucket_of(rem)
                lo = max(0, i_seg - N_CTX_SEG + 1)
                ctx = [rec["seg_texts"][k][:SEG_CHARS] for k in range(lo, i_seg + 1)]
                hist = [[round(t - te, 1), M_TYPES[ke]] for (te, ke, _s) in evs_before[-N_EV_HIST:]]
                w = {
                    "uid": "%s::%s::%s::%d" % (name, rec["tag"] or rec["arm"], rec["qid"], i_seg),
                    "layer": name, "qid": rec["qid"], "split": sp,
                    "model": rec["model"], "effort": rec["effort"], "arm": rec["arm"],
                    "benchmark": rec["benchmark"], "cap": rec["cap"],
                    "trace_uid": "%s::%s::%s" % (name, rec["tag"] or rec["arm"], rec["qid"]),
                    "i_seg": i_seg, "t_tok": t, "rem": rem, "closed": bool(nxt),
                    "bucket": b, "label": BUCKET_LABELS[b],
                    "bin_H": int(rem < H_HORIZON),
                    "cut": cut, "cut_src": src, "rho": rhov,
                    "feat_base": fb, "feat_hawkes": lamv + [rhov],
                    "A": [[round(float(x), 3) for x in row] for row in A],
                    "mu1k": [round(float(x) * 1000.0, 3) for x in mu],
                    "question": rec["question"][:Q_CHARS],
                    "ctx_segments": ctx, "ev_hist": hist,
                }
                fh[sp].write(json.dumps(w, ensure_ascii=False) + "\n")
                stats["splits"][sp] += 1
                stats["windows_by_bucket"][BUCKET_LABELS[b]] += 1
                stats["matrix_source"][src] += 1
                rems.append(rem)
                n_w += 1
        stats["layers"][name] = {"traces": n_tr, "windows": n_w}
        print("слой %-24s трасс %5d окон %6d" % (name, n_tr, n_w), flush=True)

    for f in fh.values():
        f.close()
    r = np.asarray(rems, dtype=float)
    stats["rem_quantiles"] = {str(q): float(np.percentile(r, q))
                              for q in (5, 10, 25, 50, 75, 90, 95)}
    stats["rem_mean"] = float(r.mean())
    stats["base_rate_binH"] = float((r < H_HORIZON).mean())
    stats["bucket_edges"] = BUCKET_EDGES
    stats["horizon_H"] = H_HORIZON
    stats["decay_tokens"] = decay
    stats["seed"] = SEED
    with open(os.path.join(a.out, "ds_stats.json"), "w") as f:
        json.dump(stats, f, ensure_ascii=False, indent=1)
    print(json.dumps({k: stats[k] for k in
                      ("traces", "splits", "matrix_source", "base_rate_binH",
                       "rem_quantiles", "windows_by_bucket")},
                     ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
