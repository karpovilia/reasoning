"""Этап 0: потолок шума границ хоуковских циклов по тест-ретесту судьи.

Считает согласие границ циклов и производной от них цели между тремя независимыми
проходами разметки одного и того же входа. Реализует раздел 4 предрегистрации
internal_signals_poc/cycle_end_lora_2026_08_16/PREREGISTRATION.md

Запуск на боксе 192.168.2.48, системный python3, нужны только numpy и стандартная
библиотека.
"""

import argparse
import json
import math
import os
import sys

import numpy as np

M_TYPES = ["backtrack", "verify", "branch", "subgoal_done", "failed_attempt", "commit"]
H_HORIZON = 256.0
SEED = 20260816
N_BOOT = 2000
N_PERM = 200
TOL_TOK = [0.0, 64.0, 128.0, 256.0]
TOL_SEG = [0, 1, 2]


def token_axis(seg_texts, token_total):
    """Канон из hawkes_heads_2026_08_13/scripts/build_streams.py, строка 80."""
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


def match_boundaries(pred, true, tol):
    """Канон из judge_test_retest_2026_08_15/src/analyze.py, строка 130."""
    pred, true = sorted(pred), sorted(true)
    used = [False] * len(true)
    hits = 0
    for p in pred:
        best, bd = -1, None
        for j, t in enumerate(true):
            if used[j]:
                continue
            d = abs(p - t)
            if d <= tol and (bd is None or d < bd):
                best, bd = j, d
        if best >= 0:
            used[best] = True
            hits += 1
    return hits


def rem_series(commit_tok, seg_ends, T):
    """Остаток до ближайшей консолидации в каждой точке наблюдения.

    Точки наблюдения — концы сегментов, кроме последнего. Формула дословно из
    hawkes_declustering_2026_08_13/src/points.py, строки 166-172.
    """
    obs = np.asarray(seg_ends[:-1], dtype=float)
    cm = np.asarray(sorted(commit_tok), dtype=float)
    if cm.size:
        idx = np.searchsorted(cm, obs, side="right")
        end = np.where(idx < cm.size, cm[np.minimum(idx, cm.size - 1)], T)
    else:
        end = np.full(obs.shape, float(T))
    return obs, end - obs


def spearman(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.size < 3:
        return float("nan")
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean()
    rb -= rb.mean()
    d = math.sqrt(float((ra * ra).sum()) * float((rb * rb).sum()))
    if d <= 0:
        return float("nan")
    return float((ra * rb).sum() / d)


def cohen_kappa(a, b):
    a = np.asarray(a, dtype=int)
    b = np.asarray(b, dtype=int)
    n = a.size
    if n == 0:
        return float("nan")
    po = float((a == b).mean())
    pa1, pb1 = float(a.mean()), float(b.mean())
    pe = pa1 * pb1 + (1 - pa1) * (1 - pb1)
    if pe >= 1.0:
        return float("nan")
    return (po - pe) / (1 - pe)


def auc(labels, scores):
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    npos = int(labels.sum())
    nneg = int(labels.size - npos)
    if npos == 0 or nneg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(scores.size, dtype=float)
    ranks[order] = np.arange(1, scores.size + 1, dtype=float)
    # средние ранги на связках
    s_sorted = scores[order]
    i = 0
    while i < s_sorted.size:
        j = i
        while j + 1 < s_sorted.size and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    return float((ranks[labels == 1].sum() - npos * (npos + 1) / 2.0) / (npos * nneg))


def load_traces(root):
    lab_root = os.path.join(root, "labels", "t1")
    pay_root = os.path.join(root, "payloads")
    out = {}
    for key in sorted(os.listdir(lab_root)):
        d = os.path.join(lab_root, key)
        if not os.path.isdir(d):
            continue
        pfile = os.path.join(pay_root, key + ".json")
        if not os.path.exists(pfile):
            continue
        pay = json.load(open(pfile))
        seg_texts = [s["text"] for s in pay["segments"]]
        ends, T = token_axis(seg_texts, pay.get("token_total"))
        passes = []
        for pi in (1, 2, 3):
            f = os.path.join(d, "pass%d.json" % pi)
            if not os.path.exists(f):
                passes.append(None)
                continue
            lab = json.load(open(f))
            cm_seg = sorted(
                int(e["seg_id"]) for e in lab.get("events", [])
                if e.get("type") == "commit" and 0 <= int(e.get("seg_id", -1)) < len(ends)
            )
            passes.append({
                "commit_seg": cm_seg,
                "commit_tok": [ends[s] for s in cm_seg],
                "n_events": len(lab.get("events", [])),
            })
        if any(p is None for p in passes) or len(ends) < 3:
            continue
        out[key] = {
            "key": key,
            "model": pay.get("model", "?"),
            "n_seg": len(ends),
            "seg_ends": ends,
            "T": T,
            "passes": passes,
        }
    return out


def per_trace_pairs(tr):
    """Величины на трассу по всем шести упорядоченным парам проходов."""
    res = {
        "bnd_tok": {t: [0, 0, 0] for t in TOL_TOK},   # hits, n_pred, n_true
        "bnd_seg": {t: [0, 0, 0] for t in TOL_SEG},
        "rem_a": [], "rem_b": [],
        "bin_a": [], "bin_b": [],
    }
    ends, T = tr["seg_ends"], tr["T"]
    rem = []
    for p in tr["passes"]:
        _, r = rem_series(p["commit_tok"], ends, T)
        rem.append(r)
    for i in range(3):
        for j in range(3):
            if i == j:
                continue
            pi, pj = tr["passes"][i], tr["passes"][j]
            for t in TOL_TOK:
                h = match_boundaries(pi["commit_tok"], pj["commit_tok"], t)
                res["bnd_tok"][t][0] += h
                res["bnd_tok"][t][1] += len(pi["commit_tok"])
                res["bnd_tok"][t][2] += len(pj["commit_tok"])
            for t in TOL_SEG:
                h = match_boundaries(pi["commit_seg"], pj["commit_seg"], t)
                res["bnd_seg"][t][0] += h
                res["bnd_seg"][t][1] += len(pi["commit_seg"])
                res["bnd_seg"][t][2] += len(pj["commit_seg"])
            res["rem_a"].append(rem[i])
            res["rem_b"].append(rem[j])
            res["bin_a"].append((rem[i] < H_HORIZON).astype(int))
            res["bin_b"].append((rem[j] < H_HORIZON).astype(int))
    for k in ("rem_a", "rem_b", "bin_a", "bin_b"):
        res[k] = np.concatenate(res[k]) if res[k] else np.zeros(0)
    return res


def aggregate(per, keys):
    ra = np.concatenate([per[k]["rem_a"] for k in keys])
    rb = np.concatenate([per[k]["rem_b"] for k in keys])
    ba = np.concatenate([per[k]["bin_a"] for k in keys]).astype(int)
    bb = np.concatenate([per[k]["bin_b"] for k in keys]).astype(int)
    out = {
        "n_points": int(ra.size),
        "spearman_rem": spearman(ra, rb),
        "mae_log1p": float(np.abs(np.log1p(ra) - np.log1p(rb)).mean()),
        "median_abs_diff_tok": float(np.median(np.abs(ra - rb))),
        "kappa_binH": cohen_kappa(ba, bb),
        "agree_binH": float((ba == bb).mean()),
        "auc_binH": auc(ba, bb.astype(float)),
        "base_rate_binH": float(ba.mean()),
    }
    for name, tols in (("bnd_tok", TOL_TOK), ("bnd_seg", TOL_SEG)):
        for t in tols:
            h = sum(per[k][name][t][0] for k in keys)
            np_ = sum(per[k][name][t][1] for k in keys)
            nt = sum(per[k][name][t][2] for k in keys)
            prec = h / np_ if np_ else float("nan")
            rec = h / nt if nt else float("nan")
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else float("nan")
            out["f1_%s_%s" % (name, str(t))] = float(f1)
    return out


def boot_ci(per, keys, fields, n_boot, rng):
    ks = list(keys)
    acc = {f: [] for f in fields}
    for _ in range(n_boot):
        smp = [ks[i] for i in rng.integers(0, len(ks), len(ks))]
        a = aggregate(per, smp)
        for f in fields:
            acc[f].append(a[f])
    return {f: [float(np.nanpercentile(acc[f], 2.5)), float(np.nanpercentile(acc[f], 97.5))]
            for f in fields}


def permutation_level(traces, n_perm, rng):
    """Уровень случайного согласия: цель прохода 1 против цели чужой трассы."""
    keys = sorted(traces)
    sp, kp = [], []
    for _ in range(n_perm):
        ra, rb = [], []
        perm = rng.permutation(len(keys))
        for a_i, b_i in enumerate(perm):
            if a_i == b_i:
                continue
            ta, tb = traces[keys[a_i]], traces[keys[b_i]]
            _, r1 = rem_series(ta["passes"][0]["commit_tok"], ta["seg_ends"], ta["T"])
            _, r2 = rem_series(tb["passes"][1]["commit_tok"], tb["seg_ends"], tb["T"])
            n = min(r1.size, r2.size)
            if n < 2:
                continue
            ra.append(r1[:n])
            rb.append(r2[:n])
        ra = np.concatenate(ra)
        rb = np.concatenate(rb)
        sp.append(spearman(ra, rb))
        kp.append(cohen_kappa((ra < H_HORIZON).astype(int), (rb < H_HORIZON).astype(int)))
    return {
        "spearman_rem": {"mean": float(np.nanmean(sp)),
                         "p2.5": float(np.nanpercentile(sp, 2.5)),
                         "p97.5": float(np.nanpercentile(sp, 97.5))},
        "kappa_binH": {"mean": float(np.nanmean(kp)),
                       "p2.5": float(np.nanpercentile(kp, 2.5)),
                       "p97.5": float(np.nanpercentile(kp, 97.5))},
        "n_perm": int(n_perm),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--retest-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--n-perm", type=int, default=N_PERM)
    a = ap.parse_args()

    rng = np.random.default_rng(SEED)
    traces = load_traces(a.retest_root)
    if not traces:
        print("не найдено ни одной трассы", file=sys.stderr)
        sys.exit(2)
    per = {k: per_trace_pairs(v) for k, v in traces.items()}
    keys = sorted(traces)

    point = aggregate(per, keys)
    fields = ["spearman_rem", "mae_log1p", "median_abs_diff_tok", "kappa_binH",
              "agree_binH", "auc_binH"] + \
             ["f1_bnd_tok_%s" % str(t) for t in TOL_TOK] + \
             ["f1_bnd_seg_%s" % str(t) for t in TOL_SEG]
    ci = boot_ci(per, keys, fields, a.n_boot, rng)
    perm = permutation_level(traces, a.n_perm, rng)

    by_model = {}
    models = sorted({v["model"] for v in traces.values()})
    for m in models:
        ks = [k for k in keys if traces[k]["model"] == m]
        by_model[m] = dict(aggregate(per, ks), n_traces=len(ks))

    gate_sp = ci["spearman_rem"][0] > 0.30
    gate_kp = ci["kappa_binH"][0] > 0.20
    res = {
        "arm": "cycle_end_lora_2026_08_16",
        "stage": "0_noise_ceiling",
        "prereg": "PREREGISTRATION.md, раздел 4",
        "horizon_H_tokens": H_HORIZON,
        "seed": SEED,
        "n_traces": len(traces),
        "n_passes": 3,
        "n_commits_per_pass": {
            "mean": float(np.mean([len(p["commit_tok"]) for v in traces.values()
                                   for p in v["passes"]])),
            "total": int(sum(len(p["commit_tok"]) for v in traces.values()
                             for p in v["passes"])),
        },
        "point": point,
        "ci95": ci,
        "permutation_chance": perm,
        "by_model": by_model,
        "stop_gate": {
            "rule": "нижняя граница Спирмена > 0.30 И нижняя граница каппы > 0.20",
            "spearman_lo": ci["spearman_rem"][0],
            "kappa_lo": ci["kappa_binH"][0],
            "passed_spearman": bool(gate_sp),
            "passed_kappa": bool(gate_kp),
            "passed": bool(gate_sp and gate_kp),
        },
    }
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print(json.dumps({k: res[k] for k in ("n_traces", "point", "stop_gate")},
                     ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
