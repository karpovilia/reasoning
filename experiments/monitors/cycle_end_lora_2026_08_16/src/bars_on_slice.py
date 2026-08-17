"""Бары на том же срезе оценки, что достанется прогнозисту, плюс оценка разрешающей
способности сравнения.

Срез задаётся тем же правилом, что и порядок окон в предрегистрации: sha1 от строки
«20260816::идентификатор окна». Берутся первые 1500 отложенных окон.

Разрешающая способность меряется как половина ширины бутстреп-интервала разности двух
баров: меньшую разность это сравнение отличить от нуля не сможет.

Запуск на боксе 192.168.2.48.
"""

import argparse
import hashlib
import json
import math
import os

import numpy as np

import compare as C

BARS = ["B0_const", "B1_free_point", "B2_event_gbr", "B3_event_gbr_hawkes"]


def order_key(u):
    return hashlib.sha1(("20260816::" + u).encode()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", required=True)
    ap.add_argument("--bars-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-eval", type=int, default=1500)
    a = ap.parse_args()

    te = [json.loads(l) for l in open(os.path.join(a.ds, "windows_test.jsonl"))]
    te_sorted = sorted(te, key=lambda r: order_key(r["uid"]))[:a.n_eval]
    uids = [r["uid"] for r in te_sorted]
    y = np.asarray([r["rem"] for r in te_sorted], float)
    b = np.asarray([r["bin_H"] for r in te_sorted], int)
    tr = np.asarray([r["trace_uid"] for r in te_sorted])
    grp = np.asarray([r["cut_src"] == "group" for r in te_sorted], bool)

    tt = np.load(os.path.join(a.bars_dir, "test_targets.npz"), allow_pickle=True)
    pos = {str(u): i for i, u in enumerate(tt["qid"])}
    sel = np.asarray([pos[u] for u in uids], int)
    assert np.allclose(np.asarray(tt["y"], float)[sel], y, rtol=1e-6, atol=1e-6)

    packs = {}
    for name in BARS:
        z = np.load(os.path.join(a.bars_dir, "pred_%s.npz" % name))
        packs[name] = {k: np.asarray(z[k], float)[sel] for k in ("med", "lo", "hi", "p_h")}

    res = {"n_eval": int(y.size), "n_traces": int(np.unique(tr).size),
           "base_rate": float(b.mean()), "share_group_matrix": float(grp.mean()),
           "metrics": {k: C.all_metrics(v, y, b) for k, v in packs.items()}}
    d = C.boot(tr, packs["B3_event_gbr_hawkes"], packs["B2_event_gbr"], y, b)
    res["B3_minus_B2"] = d
    res["resolution"] = {
        "M1_spearman_half_width": (d["M1"]["hi95"] - d["M1"]["lo95"]) / 2.0,
        "M3_auc_half_width": (d["M3"]["hi95"] - d["M3"]["lo95"]) / 2.0,
        "смысл": "разность меньше этой величины сравнение на срезе оценки от нуля не отличит",
    }
    res["slice_uids_file"] = os.path.join(os.path.dirname(a.out), "eval_slice_uids.json")
    with open(res["slice_uids_file"], "w") as f:
        json.dump({"n": len(uids), "uids": uids}, f, ensure_ascii=False)
    with open(a.out, "w") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print(json.dumps({k: res[k] for k in ("n_eval", "n_traces", "base_rate",
                                          "share_group_matrix", "metrics",
                                          "resolution")},
                     ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
