"""Этап 2: бары для прогноза остатка до конца текущего цикла.

Реализует раздел 8 предрегистрации эксперимента cycle_end_lora_2026_08_16.
Б0 константа, Б1 свободная точка, Б2 событийный градиентный бустинг,
Б3 тот же бустинг плюс хоуковские интенсивности и ветвящееся число группы.

Метрики раздела 9: М1 Спирмен, М2 средняя абсолютная ошибка на шкале log1p,
М3 площадь под кривой ошибок для двоичной цели, М4 оценка Уинклера при 0.80.

Запуск на боксе 192.168.2.48, системный python3 с numpy и scikit-learn.
"""

import argparse
import json
import math
import os

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier

H_HORIZON = 256.0
SEED = 20260816
N_BOOT = 2000
ALPHA = 0.20

FEAT_BASE = ["t_tok", "log_t", "cap_known", "cap_value", "frac_cap",
             "n_ev", "ev_rate_1k", "since_last_ev", "n_ev_200", "n_ev_500",
             "n_cycles", "since_commit", "mean_cycle_len", "last_cycle_len",
             "n_ev_in_cycle", "has_cand", "since_cand", "seg_len", "seg_index",
             "seg_operator", "cnt_backtrack", "cnt_verify", "cnt_branch",
             "cnt_subgoal_done", "cnt_failed_attempt", "cnt_commit",
             "since_backtrack", "since_verify", "since_branch",
             "since_subgoal_done", "since_failed_attempt", "since_commit_t"]
FEAT_HAWKES = ["lam_backtrack", "lam_verify", "lam_branch", "lam_subgoal_done",
               "lam_failed_attempt", "lam_commit", "rho_group"]
FEAT_CAT = ["model_code", "effort_code", "layer_code"]


def spearman(a, b):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    if a.size < 3:
        return float("nan")
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean()
    rb -= rb.mean()
    d = math.sqrt(float((ra * ra).sum()) * float((rb * rb).sum()))
    return float((ra * rb).sum() / d) if d > 0 else float("nan")


def auc(labels, scores):
    labels = np.asarray(labels, int)
    scores = np.asarray(scores, float)
    npos, nneg = int(labels.sum()), int(labels.size - labels.sum())
    if npos == 0 or nneg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(scores.size, float)
    ranks[order] = np.arange(1, scores.size + 1, dtype=float)
    s = scores[order]
    i = 0
    while i < s.size:
        j = i
        while j + 1 < s.size and s[j + 1] == s[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    return float((ranks[labels == 1].sum() - npos * (npos + 1) / 2.0) / (npos * nneg))


def winkler(lo, hi, y, alpha=ALPHA):
    lo = np.asarray(lo, float)
    hi = np.asarray(hi, float)
    y = np.asarray(y, float)
    w = hi - lo
    w = w + (2.0 / alpha) * np.maximum(0.0, lo - y)
    w = w + (2.0 / alpha) * np.maximum(0.0, y - hi)
    return w


def load(path):
    rows = []
    with open(path) as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def encode(rows, vocab=None):
    if vocab is None:
        vocab = {"model": {}, "effort": {}, "layer": {}}
        for r in rows:
            for k in ("model", "effort", "layer"):
                v = r.get(k) or ""
                if v not in vocab[k]:
                    vocab[k][v] = len(vocab[k])
    X_base = np.asarray([r["feat_base"] for r in rows], dtype=float)
    X_hawk = np.asarray([r["feat_hawkes"] for r in rows], dtype=float)
    X_cat = np.asarray([[vocab["model"].get(r.get("model") or "", -1),
                         vocab["effort"].get(r.get("effort") or "", -1),
                         vocab["layer"].get(r.get("layer") or "", -1)] for r in rows],
                       dtype=float)
    y = np.asarray([r["rem"] for r in rows], dtype=float)
    b = np.asarray([r["bin_H"] for r in rows], dtype=int)
    tr = np.asarray([r["trace_uid"] for r in rows])
    return X_base, X_hawk, X_cat, y, b, tr, vocab


def fit_gbr(Xtr, ytr, Xte, cat_idx):
    out = {}
    for q in (0.10, 0.50, 0.90):
        m = HistGradientBoostingRegressor(
            loss="quantile", quantile=q, max_iter=300, learning_rate=0.1,
            max_leaf_nodes=63, min_samples_leaf=100, l2_regularization=1.0,
            categorical_features=cat_idx, random_state=0)
        m.fit(Xtr, np.log1p(ytr))
        out[q] = np.expm1(m.predict(Xte))
    return out


def fit_clf(Xtr, btr, Xte, cat_idx):
    m = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.1, max_leaf_nodes=63, min_samples_leaf=100,
        l2_regularization=1.0, categorical_features=cat_idx, random_state=0)
    m.fit(Xtr, btr)
    return m.predict_proba(Xte)[:, 1]


def metrics(pred_med, pred_lo, pred_hi, p_h, y, b):
    return {
        "M1_spearman": spearman(pred_med, y),
        "M2_mae_log1p": float(np.abs(np.log1p(np.maximum(pred_med, 0.0)) - np.log1p(y)).mean()),
        "M3_auc": auc(b, p_h),
        "M4_winkler80": float(winkler(pred_lo, pred_hi, y).mean()),
        "coverage80": float(((y >= pred_lo) & (y <= pred_hi)).mean()),
        "median_width": float(np.median(pred_hi - pred_lo)),
        "n": int(y.size),
    }


def boot_diff(tr_ids, per_a, per_b, fn, n_boot=N_BOOT, seed=SEED):
    """Бутстреп по трассам для разности статистик двух прогнозов."""
    rng = np.random.default_rng(seed)
    uniq, inv = np.unique(tr_ids, return_inverse=True)
    idx_by = [np.where(inv == i)[0] for i in range(uniq.size)]
    d = []
    for _ in range(n_boot):
        pick = rng.integers(0, uniq.size, uniq.size)
        idx = np.concatenate([idx_by[p] for p in pick])
        d.append(fn(per_a, idx) - fn(per_b, idx))
    return [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)), float(np.mean(d))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    tr_rows = load(os.path.join(a.ds, "windows_train.jsonl"))
    va_rows = load(os.path.join(a.ds, "windows_val.jsonl"))
    te_rows = load(os.path.join(a.ds, "windows_test.jsonl"))
    Xb_tr, Xh_tr, Xc_tr, y_tr, b_tr, _u, vocab = encode(tr_rows + va_rows)
    Xb_te, Xh_te, Xc_te, y_te, b_te, u_te, _ = encode(te_rows, vocab)

    res = {"n_train": int(y_tr.size), "n_test": int(y_te.size),
           "base_rate_train": float(b_tr.mean()), "base_rate_test": float(b_te.mean()),
           "bars": {}}

    # Б0 константа
    med = float(np.median(y_tr))
    lo0, hi0 = float(np.percentile(y_tr, 10)), float(np.percentile(y_tr, 90))
    rate = float(b_tr.mean())
    pred0 = np.full(y_te.shape, med)
    res["bars"]["B0_const"] = metrics(pred0, np.full(y_te.shape, lo0),
                                      np.full(y_te.shape, hi0),
                                      np.full(y_te.shape, rate), y_te, b_te)
    res["bars"]["B0_const"]["train_median"] = med
    np.savez(os.path.join(a.out, "pred_B0_const.npz"), med=pred0,
             lo=np.full(y_te.shape, lo0), hi=np.full(y_te.shape, hi0),
             p_h=np.full(y_te.shape, rate))

    # Б1 свободная точка: модель + усилие + cap
    def key(r):
        return "%s|%s|%s" % (r.get("model"), r.get("effort"), r.get("cap"))
    tab = {}
    for r, yy, bb in zip(tr_rows + va_rows, y_tr, b_tr):
        tab.setdefault(key(r), []).append((yy, bb))
    p1m, p1l, p1h, p1p = [], [], [], []
    for r in te_rows:
        v = tab.get(key(r))
        if v is None or len(v) < 30:
            p1m.append(med); p1l.append(lo0); p1h.append(hi0); p1p.append(rate)
        else:
            ys = np.asarray([x[0] for x in v]); bs = np.asarray([x[1] for x in v])
            p1m.append(float(np.median(ys)))
            p1l.append(float(np.percentile(ys, 10)))
            p1h.append(float(np.percentile(ys, 90)))
            p1p.append(float(bs.mean()))
    res["bars"]["B1_free_point"] = metrics(np.asarray(p1m), np.asarray(p1l),
                                           np.asarray(p1h), np.asarray(p1p), y_te, b_te)
    np.savez(os.path.join(a.out, "pred_B1_free_point.npz"), med=np.asarray(p1m),
             lo=np.asarray(p1l), hi=np.asarray(p1h), p_h=np.asarray(p1p))

    # Б2 и Б3 бустинг
    for name, Xtr, Xte in (
        ("B2_event_gbr", np.hstack([Xb_tr, Xc_tr]), np.hstack([Xb_te, Xc_te])),
        ("B3_event_gbr_hawkes", np.hstack([Xb_tr, Xh_tr, Xc_tr]),
         np.hstack([Xb_te, Xh_te, Xc_te])),
    ):
        ncat = Xtr.shape[1] - 3
        cat_idx = [ncat, ncat + 1, ncat + 2]
        q = fit_gbr(Xtr, y_tr, Xte, cat_idx)
        p = fit_clf(Xtr, b_tr, Xte, cat_idx)
        res["bars"][name] = metrics(q[0.50], q[0.10], q[0.90], p, y_te, b_te)
        np.savez(os.path.join(a.out, "pred_%s.npz" % name),
                 med=q[0.50], lo=q[0.10], hi=q[0.90], p_h=p)

    # предсказания на отложенном срезе сохраняются целиком для сравнения с прогнозистом
    np.savez(os.path.join(a.out, "test_targets.npz"), y=y_te, b=b_te, uid=u_te,
             qid=np.asarray([r["uid"] for r in te_rows]))

    # разности лучшего бара против прочих, бутстреп по трассам
    def sp_of(p, idx):
        return spearman(p["med"][idx], y_te[idx])

    def auc_of(p, idx):
        return auc(b_te[idx], p["p_h"][idx])

    packs = {}
    for name in ("B2_event_gbr", "B3_event_gbr_hawkes"):
        z = np.load(os.path.join(a.out, "pred_%s.npz" % name))
        packs[name] = {"med": z["med"], "lo": z["lo"], "hi": z["hi"], "p_h": z["p_h"]}
    res["B3_minus_B2"] = {
        "M1_spearman": boot_diff(u_te, packs["B3_event_gbr_hawkes"], packs["B2_event_gbr"], sp_of),
        "M3_auc": boot_diff(u_te, packs["B3_event_gbr_hawkes"], packs["B2_event_gbr"], auc_of),
    }
    best = max(("B2_event_gbr", "B3_event_gbr_hawkes"),
               key=lambda n: res["bars"][n]["M3_auc"])
    res["gate_reference_bar"] = best
    res["feature_names"] = {"base": FEAT_BASE, "hawkes": FEAT_HAWKES, "cat": FEAT_CAT}
    res["horizon_H"] = H_HORIZON

    with open(os.path.join(a.out, "bars.json"), "w") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print(json.dumps({"bars": res["bars"], "gate_reference_bar": best,
                      "B3_minus_B2": res["B3_minus_B2"]}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
