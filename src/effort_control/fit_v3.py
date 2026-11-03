#!/usr/bin/env python3
"""fit v3 for the gpt-oss effort controller."""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import time
from collections import Counter, defaultdict

import numpy as np
from joblib import Parallel, delayed
from sklearn.ensemble import GradientBoostingRegressor

TYPES = ["backtrack", "verify", "branch", "subgoal_done", "failed_attempt",
         "commit"]
T_VER = TYPES.index("verify")
T_BR = TYPES.index("branch")
T_COMMIT = TYPES.index("commit")
K = len(TYPES)

Q_EXPORT = [0.10, 0.30, 0.50, 0.90]
DROP = ["phase"]
LIVE_MODEL = "openai/gpt-oss-20b"

# сетка бюджетов задания 13, геометрическая
BUDGET_GRID = [200, 287, 411, 589, 845, 1211, 1736, 2489]


# ── общие мелочи ─────────────────────────────────────────────────────────────

def norm_text(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip().lower()


def cycles_of(rec):
    T = float(rec["T"])
    commits = sorted(float(e[0]) for e in rec["events"] if int(e[1]) == T_COMMIT)
    commits = [c for c in commits if 0.0 < c <= T]
    out, prev = [], 0.0
    for c in commits:
        if c > prev:
            out.append((prev, c, True))
            prev = c
    if T > prev:
        out.append((prev, T, False))
    return out


def export_tree(tree):
    t = tree.tree_
    return (t.children_left.astype(np.int32), t.children_right.astype(np.int32),
            t.feature.astype(np.int32), t.threshold.astype(np.float64),
            t.value.reshape(-1).astype(np.float64))


def pack_forest(model):
    cl, cr, fe, th, va, off = [], [], [], [], [], [0]
    for est in model.estimators_[:, 0]:
        a, b, c, d, e = export_tree(est)
        cl.append(a)
        cr.append(b)
        fe.append(c)
        th.append(d)
        va.append(e)
        off.append(off[-1] + len(a))
    return {"children_left": np.concatenate(cl),
            "children_right": np.concatenate(cr),
            "feature": np.concatenate(fe),
            "threshold": np.concatenate(th),
            "value": np.concatenate(va),
            "offsets": np.asarray(off, dtype=np.int64)}


def forest_predict(pack, lr, base, X):
    cl, cr = pack["children_left"], pack["children_right"]
    fe, th, va, off = (pack["feature"], pack["threshold"], pack["value"],
                       pack["offsets"])
    n = X.shape[0]
    out = np.full(n, float(base), dtype=np.float64)
    for t in range(len(off) - 1):
        o = int(off[t])
        node = np.zeros(n, dtype=np.int64)
        while True:
            f = fe[o + node]
            leaf = f < 0
            if leaf.all():
                break
            go_left = np.zeros(n, dtype=bool)
            idx = np.where(~leaf)[0]
            go_left[idx] = X[idx, f[idx]] <= th[o + node[idx]]
            nxt = np.where(go_left, cl[o + node], cr[o + node])
            node = np.where(leaf, node, nxt)
        out += lr * va[o + node]
    return out


def pinball(y, p, q):
    d = y - p
    return float(np.mean(np.maximum(q * d, (q - 1.0) * d)))


def winkler(y, lo, hi, alpha=0.2):
    w = hi - lo
    pen = (2.0 / alpha) * (np.maximum(lo - y, 0.0) + np.maximum(y - hi, 0.0))
    return float(np.mean(w + pen))


# ── голова трудности вопроса ────────────────────────────────────────────────

DIFF_FEATURES = ["log_chars", "log_words", "digit_share", "dollar_count",
                 "newline_count", "mean_word_len", "has_choices",
                 "nonascii_share", "question_marks"]


def difficulty_x(text: str) -> np.ndarray:
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cycle-dir", required=True)
    ap.add_argument("--streams", required=True)
    ap.add_argument("--qtext", required=True,
                    help="json: uid -> {question, ...}, собран build_qtext.py")
    ap.add_argument("--diag", required=True, help="diag_v1.jsonl отложенной выборки")
    ap.add_argument("--packs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", default="commit")
    ap.add_argument("--jobs", type=int, default=1)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()
    report = {}

    # ── 1. корпус циклов и починка ключа усилия ─────────────────────────────
    X = np.load(os.path.join(args.cycle_dir, "cycles_X_%s.npz" % args.tag))["X"]
    meta = [json.loads(l) for l in
            open(os.path.join(args.cycle_dir, "cycles_meta_%s.jsonl" % args.tag),
                 encoding="utf-8")]
    feats = json.load(open(os.path.join(args.cycle_dir,
                                        "features_%s.json" % args.tag),
                           encoding="utf-8"))
    names = list(feats["features"])
    arms = list(feats["arms"])
    models = list(feats["models"])
    i_arm = names.index("arm")
    hp = json.load(open(os.path.join(args.cycle_dir,
                                     "c1_metrics_%s_nophase.json" % args.tag),
                        encoding="utf-8"))["hp"]

    tag_of_uid = {}
    for line in open(args.streams, encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        tag_of_uid[r["uid"]] = r.get("tag", "")

    def arm_from_tag(tag, old):
        if tag.endswith("_effortlow"):
            return "effort_low"
        if tag.endswith("_efforthigh"):
            return "effort_high"
        return old

    arm_idx = {a: i for i, a in enumerate(arms)}
    X3 = X.copy()
    changed = Counter()
    for j, m in enumerate(meta):
        tag = tag_of_uid.get(m["uid"], "")
        new = arm_from_tag(tag, m["arm"])
        if new != m["arm"]:
            changed[(m["model"], m["arm"], new)] += 1
            X3[j, i_arm] = arm_idx[new]
        m["arm_fixed"] = new
    report["effort_key_fix"] = {
        "rule": ("плечо восстановлено из тега трассы: суффикс _effortlow даёт "
                 "effort_low, суффикс _efforthigh даёт effort_high"),
        "rows_changed": int(sum(changed.values())),
        "by_model": {"%s|%s->%s" % k: v for k, v in sorted(changed.items())},
        "rows_total": int(X.shape[0])}
    print("[1] строк с починенным ключом усилия: %d из %d; %s"
          % (sum(changed.values()), X.shape[0],
             report["effort_key_fix"]["by_model"]), flush=True)

    # ── 2. переобучение прогноза стоимости цикла ───────────────────────────
    keep = [i for i, n in enumerate(names) if n not in DROP]
    kept_names = [names[i] for i in keep]
    split = np.asarray([m["split"] for m in meta])
    y = np.asarray([m["y_next"] for m in meta], float)
    ly = np.log1p(y)
    trva = (split == "train") | (split == "val")
    te = split == "test"
    Xk3 = X3[:, keep]
    Xk0 = X[:, keep]

    def fit_one(Xm, q):
        m = GradientBoostingRegressor(loss="quantile", alpha=q, random_state=0,
                                      subsample=1.0, **hp)
        m.fit(Xm[trva], ly[trva])
        return m

    models_new = Parallel(n_jobs=min(args.jobs, len(Q_EXPORT)))(
        delayed(fit_one)(Xk3, q) for q in Q_EXPORT)
    models_old = Parallel(n_jobs=min(args.jobs, len(Q_EXPORT)))(
        delayed(fit_one)(Xk0, q) for q in Q_EXPORT)
    print("[2] обучено 2 x %d моделей за %.0f с" % (len(Q_EXPORT),
                                                    time.time() - t0), flush=True)

    # качество на отложенной части, отдельно по строкам живой модели
    i_model = kept_names.index("model")
    mi_live = models.index(LIVE_MODEL)
    sel_live = te & (X[:, names.index("model")] == mi_live)
    qual = {}
    for lab, mm, Xm in (("v3_effort_key", models_new, Xk3),
                        ("v2_as_is", models_old, Xk0)):
        pr = {q: np.expm1(m.predict(Xm[te])) for q, m in zip(Q_EXPORT, mm)}
        prl = {q: np.expm1(m.predict(Xm[sel_live])) for q, m in zip(Q_EXPORT, mm)}
        qual[lab] = {
            "n_test": int(te.sum()), "n_test_live_model": int(sel_live.sum()),
            "winkler_80": round(winkler(y[te], pr[0.1], pr[0.9]), 1),
            "winkler_80_live_model": round(
                winkler(y[sel_live], prl[0.1], prl[0.9]), 1),
            "pinball_q50": round(pinball(y[te], pr[0.5], 0.5), 1),
            "pinball_q50_live_model": round(pinball(y[sel_live], prl[0.5], 0.5), 1),
        }
    report["forecast_quality"] = qual
    print("[2] качество на отложенной части: %s" % json.dumps(qual, ensure_ascii=False),
          flush=True)

    # выгрузка массивами и сверка обходчика
    blob = {"features": kept_names, "quantiles": Q_EXPORT, "hp": hp,
            "n_rows_fit": int(trva.sum()), "models": models, "arms": arms,
            "effort_key_fixed": True}
    arrays, checks = {}, []
    Xchk = Xk3[te][:2000]
    for j, (q, m) in enumerate(zip(Q_EXPORT, models_new)):
        pack = pack_forest(m)
        lr = float(m.learning_rate)
        raw = m.predict(Xchk)
        base0 = forest_predict(pack, lr, 0.0, Xchk)
        cand = raw - base0
        base = float(np.median(cand))
        spread = float(np.max(np.abs(cand - base)))
        err = float(np.max(np.abs(forest_predict(pack, lr, base, Xchk) - raw)))
        assert spread < 1e-8 and err < 1e-6, (spread, err)
        checks.append({"q": q, "base": base, "base_spread": spread,
                       "max_abs_err": err, "n_trees": len(pack["offsets"]) - 1})
        for k, v in pack.items():
            arrays["q%d_%s" % (j, k)] = v
        arrays["q%d_meta" % j] = np.asarray([lr, base], dtype=np.float64)
    blob["selfcheck"] = checks
    np.savez_compressed(os.path.join(args.out, "cnext_forest.npz"), **arrays)
    json.dump(blob, open(os.path.join(args.out, "cnext_forest.json"), "w"),
              ensure_ascii=False, indent=1)
    print("[2] лес выгружен, сверка обходчика max %.2e"
          % max(c["max_abs_err"] for c in checks), flush=True)

    # ── 3. чувствительность прогноза к ключу усилия ────────────────────────
    i_arm_k = kept_names.index("arm")
    Xs = Xk3[sel_live].copy()
    lad = {}
    for lab, code in (("none", arm_idx["none"]), ("effort_low", arm_idx["effort_low"]),
                      ("effort_high", arm_idx["effort_high"])):
        Xa = Xs.copy()
        Xa[:, i_arm_k] = code
        lad[lab] = {q: np.expm1(m.predict(Xa))
                    for q, m in zip(Q_EXPORT, models_new)}
    ratio = lad["effort_low"][0.5] / np.maximum(lad["none"][0.5], 1e-9)
    sens = {"n_states": int(Xs.shape[0]),
            "share_states_where_low_differs": round(
                float(np.mean(np.abs(ratio - 1.0) > 1e-9)), 4),
            "ratio_low_to_none_q50": {
                str(p): round(float(np.quantile(ratio, p)), 3)
                for p in (0.1, 0.25, 0.5, 0.75, 0.9)},
            "median_cnext_none_q50": round(float(np.median(lad["none"][0.5])), 1),
            "median_cnext_low_q50": round(float(np.median(lad["effort_low"][0.5])), 1)}
    # запасной путь: эмпирическое отношение длин циклов при низком усилии
    len_by_arm = defaultdict(list)
    for line in open(args.streams, encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        if r["split"] not in ("train", "val"):
            continue
        tag = r.get("tag", "")
        a = arm_from_tag(tag, r["arm"])
        if r["model"] != LIVE_MODEL:
            continue
        for (lo, hi, _c) in cycles_of(r):
            len_by_arm[a].append(hi - lo)
    emp = {a: {"n": len(v), "median": round(float(np.median(v)), 1),
               "mean": round(float(np.mean(v)), 1)}
           for a, v in sorted(len_by_arm.items()) if v}
    rho = None
    if "effort_low" in emp and "none" in emp:
        rho = round(emp["effort_low"]["median"] / emp["none"]["median"], 4)
    sens["empirical_cycle_len_by_arm_live_model"] = emp
    sens["rho_low_empirical"] = rho
    report["effort_sensitivity"] = sens
    print("[3] чувствительность к усилию: %s" % json.dumps(sens, ensure_ascii=False),
          flush=True)

    # ── 4. таблицы цены проверочного цикла и безусловной длины ─────────────
    cheap, all_cycles = {}, {}
    for line in open(args.streams, encoding="utf-8"):
        if not line.strip():
            continue
        rec = json.loads(line)
        if rec["split"] not in ("train", "val"):
            continue
        evs = sorted((float(e[0]), int(e[1])) for e in rec["events"])
        cand = rec.get("cand_tok")
        for (a, b, _c) in cycles_of(rec):
            all_cycles.setdefault(rec["model"], []).append(b - a)
            all_cycles.setdefault("__pooled__", []).append(b - a)
            if cand is None or cand > a:
                continue
            inside = [k for (s, k) in evs if a < s <= b]
            if T_VER in inside and T_BR not in inside:
                cheap.setdefault(rec["model"], []).append(b - a)
                cheap.setdefault("__pooled__", []).append(b - a)
    cv = {"qs": Q_EXPORT, "min_n": 40, "n": {}}
    for k, v in cheap.items():
        cv["n"][k] = len(v)
        if len(v) >= 40:
            cv[k] = [round(float(x), 1) for x in np.quantile(np.asarray(v, float),
                                                             Q_EXPORT)]
    uncond = {"qs": Q_EXPORT, "n": {k: len(v) for k, v in all_cycles.items()}}
    for k, v in all_cycles.items():
        if len(v) >= 40:
            uncond[k] = [round(float(x), 1) for x in
                         np.quantile(np.asarray(v, float), Q_EXPORT)]
    json.dump({"c_verify": cv, "c_uncond": uncond},
              open(os.path.join(args.out, "cverify_table.json"), "w"),
              ensure_ascii=False, indent=1)
    report["c_verify_live_model"] = cv.get(LIVE_MODEL)
    print("[4] C_verify живой модели: %s" % cv.get(LIVE_MODEL), flush=True)

    # ── 5. голова трудности вопроса ────────────────────────────────────────
    qtext = json.load(open(args.qtext, encoding="utf-8"))
    diag = [json.loads(l) for l in open(args.diag, encoding="utf-8") if l.strip()]
    ban = {norm_text(r["question"]) for r in diag}
    rows = []
    for line in open(args.streams, encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        if r["model"] not in (LIVE_MODEL, "gpt-oss-20b"):
            continue
        tag = r.get("tag", "")
        if arm_from_tag(tag, r["arm"]) not in ("none", "older"):
            continue                      # обучаем на обычном уровне усилия
        q = qtext.get(r["uid"])
        if not q:
            continue
        nt = norm_text(q)
        if nt in ban:
            continue
        rows.append({"uid": r["uid"], "src": r["source"], "text": q,
                     "T": float(r["T"]), "nt": nt})
    by_q = defaultdict(list)
    for r in rows:
        by_q[r["nt"]].append(r)
    keys = sorted(by_q)
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(keys))
    n_ho = max(int(0.2 * len(keys)), 1)
    ho = {keys[i] for i in perm[:n_ho]}
    srcs = sorted({r["src"] for r in rows})
    si = {s: i for i, s in enumerate(srcs)}

    def design(rs):
        F = np.asarray([difficulty_x(r["text"]) for r in rs])
        S = np.zeros((len(rs), len(srcs)))
        for j, r in enumerate(rs):
            S[j, si[r["src"]]] = 1.0
        return np.hstack([F, S])

    tr_rows = [r for r in rows if r["nt"] not in ho]
    ho_rows = [r for r in rows if r["nt"] in ho]
    A = design(tr_rows)
    b = np.log1p(np.asarray([r["T"] for r in tr_rows]))
    lam = 1.0
    W = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ b)
    pred_tr = A @ W
    pred_ho = design(ho_rows) @ W
    y_ho = np.log1p(np.asarray([r["T"] for r in ho_rows]))

    def spearman(a, bb):
        ra = np.argsort(np.argsort(a)).astype(float)
        rb = np.argsort(np.argsort(bb)).astype(float)
        ra -= ra.mean()
        rb -= rb.mean()
        d = float(np.sqrt((ra ** 2).sum() * (rb ** 2).sum()))
        return float((ra * rb).sum() / d) if d > 0 else 0.0

    # опорный уровень: среднее предсказание по обучающим вопросам живого источника
    ref_src = "native" if "native" in si else srcs[0]
    mu_ref = float(np.mean([p for p, r in zip(pred_tr, tr_rows)
                            if r["src"] == ref_src])) if any(
        r["src"] == ref_src for r in tr_rows) else float(np.mean(pred_tr))
    sigma = float(np.std(b - pred_tr))
    dh = {"features": DIFF_FEATURES, "sources": srcs, "ref_source": ref_src,
          "w": [float(x) for x in W], "mu_ref": mu_ref, "sigma_log": sigma,
          "ridge_lambda": lam,
          "n_train_rows": len(tr_rows), "n_holdout_rows": len(ho_rows),
          "n_train_questions": len(keys) - len(ho), "n_holdout_questions": len(ho),
          "spearman_holdout": round(spearman(pred_ho, y_ho), 4),
          "spearman_train": round(spearman(pred_tr, b), 4),
          "banned_questions": len(ban),
          "multiplier_meaning": ("m(x) = exp(mu(x) - mu_ref), где mu(x) — "
                                 "предсказание log1p стоимости трассы по тексту "
                                 "вопроса; абсолютный масштаб берётся у прогноза "
                                 "стоимости цикла")}
    # разброс множителя на отложенных вопросах diag_v1 — только входы, без исходов
    md = np.exp(np.asarray([difficulty_x(r["question"]) for r in diag]) @ W[:len(DIFF_FEATURES)]
                + W[len(DIFF_FEATURES) + si[ref_src]] - mu_ref)
    dh["multiplier_on_eval_questions"] = {
        str(p): round(float(np.quantile(md, p)), 3)
        for p in (0.05, 0.25, 0.5, 0.75, 0.95)}
    json.dump(dh, open(os.path.join(args.out, "difficulty_head.json"), "w"),
              ensure_ascii=False, indent=1)
    report["difficulty_head"] = {k: v for k, v in dh.items() if k != "w"}
    print("[5] голова трудности: обучающих вопросов %d, отложенных %d, "
          "Спирмен на отложенных %.3f, множитель на оценочных вопросах %s"
          % (dh["n_train_questions"], dh["n_holdout_questions"],
             dh["spearman_holdout"], dh["multiplier_on_eval_questions"]), flush=True)

    # ── 6. опорная лестница стоимости первого цикла по сетке бюджетов ──────
    # состояние на старте трассы: цикл нулевой, событий нет, кандидата нет
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from . import gate_common_v3 as GC
    fc = GC.ForestForecaster.load(args.out)
    assert list(fc.features) == list(GC.FEATURE_ORDER), "порядок признаков разошёлся"
    c0 = {}
    for B in BUDGET_GRID:
        row = {}
        for lab, code in (("none", arm_idx["none"]),
                          ("effort_low", arm_idx["effort_low"])):
            x = GC.build_features([], [], None, B, [0.0], mi_live, code)
            L = fc.ladder(x)
            row[lab] = {str(q): round(fc.at(L, q), 1) for q in (0.3, 0.5)}
        c0[str(B)] = row
    report["c0_by_budget"] = c0
    print("[6] стоимость первого цикла по сетке: %s"
          % json.dumps(c0, ensure_ascii=False), flush=True)

    # ── 7. ведущий тип ближайшего события по корпусу живой модели ──────────
    hpack = json.load(open(os.path.join(args.packs, "hawkes_pack.json"),
                           encoding="utf-8"))
    p = hpack.get(LIVE_MODEL) or hpack["__pooled__"]
    mu = np.asarray(p["mu"], float)
    Amat = np.asarray(p["A"], float)
    beta = float(p["beta"])
    argmax_cnt = Counter()
    pb, pc = [], []
    n_pts = 0
    for line in open(args.streams, encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        if r["model"] != LIVE_MODEL or r["split"] not in ("train", "val"):
            continue
        evs = sorted((float(e[0]), int(e[1])) for e in r["events"])
        cand = r.get("cand_tok")
        # точки решения политики: границы циклов после появления кандидата
        bounds = [s for (s, k) in evs if k == T_COMMIT]
        S = np.zeros(K)
        t = 0.0
        ei = 0
        for bnd in bounds:
            while ei < len(evs) and evs[ei][0] <= bnd:
                s, k = evs[ei]
                S *= math.exp(-beta * (s - t))
                t = s
                S[k] += beta
                ei += 1
            if cand is None or cand > bnd:
                continue
            S2 = S * math.exp(-beta * (bnd - t))
            lam_ = mu + Amat.T.dot(S2)
            pr = lam_ / max(lam_.sum(), 1e-12)
            argmax_cnt[TYPES[int(np.argmax(pr))]] += 1
            pb.append(float(pr[T_BR]))
            pc.append(float(pr[T_COMMIT]))
            n_pts += 1
    nxt = {"n_points": n_pts,
           "argmax_share": {k: round(v / max(n_pts, 1), 4)
                            for k, v in argmax_cnt.most_common()},
           "p_branch_quantiles": {str(q): round(float(np.quantile(pb, q)), 4)
                                  for q in (0.5, 0.75, 0.9)} if pb else {},
           "p_commit_quantiles": {str(q): round(float(np.quantile(pc, q)), 4)
                                  for q in (0.25, 0.5, 0.75, 0.9)} if pc else {}}
    report["next_type_head"] = nxt
    print("[7] ведущий тип ближайшего события: %s" % json.dumps(nxt, ensure_ascii=False),
          flush=True)

    report["built_at"] = time.strftime("%Y-%m-%d %H:%M:%S %z")
    report["wall_s"] = round(time.time() - t0, 1)
    json.dump(report, open(os.path.join(args.out, "fit_v3_report.json"), "w"),
              ensure_ascii=False, indent=1)
    print("готово за %.0f с, отчёт %s"
          % (time.time() - t0, os.path.join(args.out, "fit_v3_report.json")),
          flush=True)


if __name__ == "__main__":
    main()
