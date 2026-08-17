"""Метрики детекторов при пороге, выбранном по проверочной части, и разрез по семействам.

Обученные головы не переобучаются: веса читаются из файлов outputs/main/net_*.pt того же
прогона, что дал прогнозы отложенной части. Бары подгоняются заново с тем же зерном, и
совпадение их прогнозов с сохранёнными проверяется прямым сравнением.

Запуск на боксе tesla:
  /home/user/ki_sft_judge/.venv/bin/python -u src/detector_metrics.py \
      --ds data/ds --out outputs/main
"""

import argparse
import json
import os

import numpy as np
import torch

import common as C
import run_main as R

H = C.H_HORIZON


def best_f1(scores, labels):
    ths = np.quantile(scores, np.linspace(0.02, 0.98, 49))
    best = (0.0, 0.5)
    lab = labels > 0.5
    for th in ths:
        pr = scores >= th
        tp = float((pr & lab).sum())
        fp = float((pr & ~lab).sum())
        fn = float((~pr & lab).sum())
        f1 = 2 * tp / max(2 * tp + fp + fn, 1e-9)
        if f1 > best[0]:
            best = (f1, float(th))
    return best


def prf(scores, labels, th):
    lab = labels > 0.5
    pr = scores >= th
    tp = float((pr & lab).sum())
    fp = float((pr & ~lab).sum())
    fn = float((~pr & lab).sum())
    p_ = tp / max(tp + fp, 1e-9)
    r_ = tp / max(tp + fn, 1e-9)
    return {"порог": th, "точность": p_, "полнота": r_,
            "F1": 2 * p_ * r_ / max(p_ + r_, 1e-9)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    tr = C.load_traces(a.ds, "train")
    va = C.load_traces(a.ds, "val")
    te = C.load_traces(a.ds, "test")
    Dtr, Dva, Dte = C.build_matrix(tr), C.build_matrix(va), C.build_matrix(te)
    mu = Dtr["X"].mean(0)
    sd = Dtr["X"].std(0)
    sd[sd < 1e-6] = 1.0
    y1tr = np.log1p(Dtr["rem1"])
    d1tr = (Dtr["rem1"] < H).astype(np.float32)

    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import HistGradientBoostingClassifier

    def time_only(D):
        t = D["X"][:, C.FEATS.index("t_tok")]
        return np.stack([t / 1000.0, np.log1p(t)], 1)

    hp = dict(max_iter=300, learning_rate=0.1, max_leaf_nodes=63,
              min_samples_leaf=100, l2_regularization=1.0, random_state=R.SEED)
    scores = {}
    lc = LogisticRegression(max_iter=1000).fit(time_only(Dtr), d1tr)
    lb = LogisticRegression(max_iter=1000).fit(time_only(Dtr), Dtr["bnext"])
    scores["B1_time_linear"] = {
        "val": (lc.predict_proba(time_only(Dva))[:, 1],
                lb.predict_proba(time_only(Dva))[:, 1]),
        "test": (lc.predict_proba(time_only(Dte))[:, 1],
                 lb.predict_proba(time_only(Dte))[:, 1])}
    gc = HistGradientBoostingClassifier(**hp).fit(Dtr["X"], d1tr)
    gb = HistGradientBoostingClassifier(**hp).fit(Dtr["X"], Dtr["bnext"])
    scores["B2_event_gbr"] = {
        "val": (gc.predict_proba(Dva["X"])[:, 1], gb.predict_proba(Dva["X"])[:, 1]),
        "test": (gc.predict_proba(Dte["X"])[:, 1], gb.predict_proba(Dte["X"])[:, 1])}
    for kind, name in (("gru", "M_gru_heads"), ("mlp", "M0_mlp_heads")):
        net = (R.GruHeads(mu.shape[0]) if kind == "gru"
               else R.MlpHeads(mu.shape[0])).to(dev)
        net.load_state_dict(torch.load(os.path.join(a.out, "net_%s.pt" % kind),
                                       map_location=dev))
        pv = R.predict_net(net, Dva["seqs"], mu, sd, dev)
        pt = R.predict_net(net, Dte["seqs"], mu, sd, dev)
        scores[name] = {"val": (pv[2], pv[3]), "test": (pt[2], pt[3])}

    z = np.load(os.path.join(a.out, "preds_test.npz"))
    check = {nm: float(np.max(np.abs(scores[nm]["test"][0] - z["%s__p_h" % nm])))
             for nm in scores if ("%s__p_h" % nm) in z}
    labv_h = (Dva["rem1"] < H).astype(float)
    labt_h = (Dte["rem1"] < H).astype(float)
    out = {"сверка с сохранёнными прогнозами": check, "детекторы": {}}
    for nm, s in scores.items():
        _f, th_h = best_f1(s["val"][0], labv_h)
        _f, th_b = best_f1(s["val"][1], Dva["bnext"])
        out["детекторы"][nm] = {
            "Д1 конец текущего цикла в горизонте 256 токенов": dict(
                prf(s["test"][0], labt_h, th_h),
                AUC=C.auc(s["test"][0], labt_h),
                PR_AUC=C.pr_auc(s["test"][0], labt_h),
                доля_положительных=float(labt_h.mean())),
            "Д2 консолидация в следующем сегменте": dict(
                prf(s["test"][1], Dte["bnext"], th_b),
                AUC=C.auc(s["test"][1], Dte["bnext"]),
                PR_AUC=C.pr_auc(s["test"][1], Dte["bnext"]),
                доля_положительных=float(Dte["bnext"].mean())),
        }
    # постоянный бар: вероятность одна на всех
    out["детекторы"]["B0_const"] = {
        "Д1 конец текущего цикла в горизонте 256 токенов": {
            "порог": None, "точность": float(labt_h.mean()), "полнота": 1.0,
            "F1": float(2 * labt_h.mean() / (1 + labt_h.mean())), "AUC": 0.5,
            "PR_AUC": float(labt_h.mean()), "доля_положительных": float(labt_h.mean())},
        "Д2 консолидация в следующем сегменте": {
            "порог": None, "точность": float(Dte["bnext"].mean()), "полнота": 1.0,
            "F1": float(2 * Dte["bnext"].mean() / (1 + Dte["bnext"].mean())),
            "AUC": 0.5, "PR_AUC": float(Dte["bnext"].mean()),
            "доля_положительных": float(Dte["bnext"].mean())}}

    # разрез по семейству модели на сохранённых прогнозах
    meta = json.load(open(os.path.join(a.out, "test_traces_meta.json"),
                          encoding="utf-8"))
    fam_pt = np.array([meta[int(i)]["family"] for i in z["tid"]])
    by_fam = {}
    names = [nm for nm in ("M_gru_heads", "M0_mlp_heads", "B2_event_gbr",
                           "B1_time_linear") if ("%s__rem1" % nm) in z]
    for f in sorted(set(fam_pt.tolist())):
        m = fam_pt == f
        if m.sum() < 500:
            continue
        row = {"точек": int(m.sum()), "трасс": int(len(set(z["tid"][m].tolist()))),
               "доля окон с концом цикла в горизонте": float(labt_h[m].mean())}
        for nm in names:
            row[nm] = {
                "M1_rem1": C.spearman(np.expm1(z["%s__rem1" % nm][m]), z["rem1"][m]),
                "M1_rem2": C.spearman(np.expm1(z["%s__rem2" % nm][m]), z["rem2"][m]),
                "M3_auc_H": C.auc(z["%s__p_h" % nm][m], labt_h[m])}
        by_fam[f] = row
    out["по семействам"] = by_fam
    json.dump(out, open(os.path.join(a.out, "detector_metrics.json"), "w",
                        encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps(out, ensure_ascii=False, indent=1)[:3000])


if __name__ == "__main__":
    main()
