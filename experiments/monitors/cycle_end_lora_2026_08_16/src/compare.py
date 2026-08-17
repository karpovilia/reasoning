"""Сведение: прогнозист против баров и плечо с инъекцией против плеча без неё.

Реализует раздел 9 предрегистрации эксперимента cycle_end_lora_2026_08_16. Считает метрики
М1-М4 на общем срезе оценки, разности с бутстрепом по трассам и вторичный срез окон,
получивших матрицу своей группы.

Запуск на боксе 192.168.2.48, системный python3 с numpy.
"""

import argparse
import json
import math
import os

import numpy as np

H_HORIZON = 256.0
ALPHA = 0.20
N_BOOT = 2000
SEED = 20260816
BARS = ["B0_const", "B1_free_point", "B2_event_gbr", "B3_event_gbr_hawkes"]


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
    w = np.asarray(hi, float) - np.asarray(lo, float)
    w = w + (2.0 / alpha) * np.maximum(0.0, np.asarray(lo, float) - y)
    w = w + (2.0 / alpha) * np.maximum(0.0, y - np.asarray(hi, float))
    return w


def stat(name, p, y, b, idx):
    if name == "M1":
        return spearman(p["med"][idx], y[idx])
    if name == "M2":
        return float(np.abs(np.log1p(np.maximum(p["med"][idx], 0.0))
                            - np.log1p(y[idx])).mean())
    if name == "M3":
        return auc(b[idx], p["p_h"][idx])
    if name == "M4":
        return float(winkler(p["lo"][idx], p["hi"][idx], y[idx]).mean())
    raise KeyError(name)


def all_metrics(p, y, b):
    idx = np.arange(y.size)
    return {
        "M1_spearman": stat("M1", p, y, b, idx),
        "M2_mae_log1p": stat("M2", p, y, b, idx),
        "M3_auc": stat("M3", p, y, b, idx),
        "M4_winkler80": stat("M4", p, y, b, idx),
        "coverage80": float(((y >= p["lo"]) & (y <= p["hi"])).mean()),
        "median_width": float(np.median(p["hi"] - p["lo"])),
        "n": int(y.size),
    }


def boot(tr_ids, pa, pb, y, b, n_boot=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    uniq, inv = np.unique(tr_ids, return_inverse=True)
    idx_by = [np.where(inv == i)[0] for i in range(uniq.size)]
    acc = {m: [] for m in ("M1", "M2", "M3", "M4")}
    for _ in range(n_boot):
        pick = rng.integers(0, uniq.size, uniq.size)
        idx = np.concatenate([idx_by[p] for p in pick])
        for m in acc:
            acc[m].append(stat(m, pa, y, b, idx) - stat(m, pb, y, b, idx))
    out = {}
    for m, v in acc.items():
        v = np.asarray(v, float)
        out[m] = {"mean": float(np.nanmean(v)),
                  "lo95": float(np.nanpercentile(v, 2.5)),
                  "hi95": float(np.nanpercentile(v, 97.5))}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bars-dir", required=True)
    ap.add_argument("--pred-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--arms", default="noinj,inj")
    a = ap.parse_args()

    arms = [s for s in a.arms.split(",") if s]
    preds = {}
    for arm in arms:
        f = os.path.join(a.pred_dir, "pred_%s.jsonl" % arm)
        if not os.path.exists(f):
            print("нет файла предсказаний плеча %s, пропускаю" % arm)
            continue
        preds[arm] = [json.loads(l) for l in open(f)]
    if not preds:
        raise SystemExit("не найдено ни одного файла предсказаний")

    ref = preds[arms[0]] if arms[0] in preds else list(preds.values())[0]
    uids = [r["uid"] for r in ref]
    uset = {u: i for i, u in enumerate(uids)}
    y = np.asarray([r["rem"] for r in ref], float)
    b = np.asarray([r["bin_H"] for r in ref], int)
    tr = np.asarray([r["trace_uid"] for r in ref])
    grp = np.asarray([r.get("cut_src") == "group" for r in ref], bool)

    packs = {}
    for arm, rows in preds.items():
        m = {r["uid"]: r for r in rows}
        miss = [u for u in uids if u not in m]
        if miss:
            raise SystemExit("плечо %s не покрывает срез оценки, нет %d окон" % (arm, len(miss)))
        packs["lora_" + arm] = {
            "med": np.asarray([m[u]["med"] for u in uids], float),
            "lo": np.asarray([m[u]["lo"] for u in uids], float),
            "hi": np.asarray([m[u]["hi"] for u in uids], float),
            "p_h": np.asarray([m[u]["p_H"] for u in uids], float),
        }

    tt = np.load(os.path.join(a.bars_dir, "test_targets.npz"), allow_pickle=True)
    bar_uid = [str(x) for x in tt["qid"]]
    pos = {u: i for i, u in enumerate(bar_uid)}
    sel = np.asarray([pos[u] for u in uids], int)
    y_bar = np.asarray(tt["y"], float)[sel]
    if not np.allclose(y_bar, y, rtol=1e-6, atol=1e-6):
        raise SystemExit("цель баров и цель прогнозиста разошлись на срезе оценки")
    for name in BARS:
        z = np.load(os.path.join(a.bars_dir, "pred_%s.npz" % name))
        packs[name] = {k: np.asarray(z[k], float)[sel] for k in ("med", "lo", "hi", "p_h")}

    res = {"n_eval": int(y.size), "base_rate": float(b.mean()),
           "n_traces": int(np.unique(tr).size),
           "share_group_matrix": float(grp.mean()),
           "metrics": {k: all_metrics(v, y, b) for k, v in packs.items()},
           "diffs": {}}

    lora = [k for k in packs if k.startswith("lora_")]
    ref_bar = max(("B2_event_gbr", "B3_event_gbr_hawkes"),
                  key=lambda n: res["metrics"][n]["M3_auc"])
    res["gate_reference_bar"] = ref_bar
    for k in lora:
        res["diffs"]["%s_minus_%s" % (k, ref_bar)] = boot(tr, packs[k], packs[ref_bar], y, b)
    if "lora_inj" in packs and "lora_noinj" in packs:
        res["diffs"]["lora_inj_minus_lora_noinj"] = boot(
            tr, packs["lora_inj"], packs["lora_noinj"], y, b)
        # вторичный срез: окна со своей матрицей группы
        if grp.any() and (~grp).any():
            for tag, mask in (("group_matrix", grp), ("pooled_matrix", ~grp)):
                sub = {k: {kk: vv[mask] for kk, vv in v.items()} for k, v in packs.items()}
                res.setdefault("secondary", {})[tag] = {
                    "n": int(mask.sum()),
                    "metrics": {k: all_metrics(v, y[mask], b[mask]) for k, v in sub.items()},
                    "inj_minus_noinj": boot(tr[mask], sub["lora_inj"], sub["lora_noinj"],
                                            y[mask], b[mask]),
                }

    gate = {}
    for k in lora:
        d = res["diffs"]["%s_minus_%s" % (k, ref_bar)]
        gate[k] = {
            "M1_lo95": d["M1"]["lo95"], "M3_lo95": d["M3"]["lo95"],
            "passed": bool(d["M1"]["lo95"] > 0 and d["M3"]["lo95"] > 0),
        }
    res["acceptance_gate"] = {"reference_bar": ref_bar, "arms": gate}
    if "lora_inj_minus_lora_noinj" in res["diffs"]:
        d = res["diffs"]["lora_inj_minus_lora_noinj"]
        worse = (d["M1"]["hi95"] < 0) or (d["M3"]["hi95"] < 0) or \
                (d["M2"]["lo95"] > 0) or (d["M4"]["lo95"] > 0)
        res["injection_verdict"] = {
            "M3_lo95": d["M3"]["lo95"], "M1_lo95": d["M1"]["lo95"],
            "any_significant_harm": bool(worse),
            "helped": bool(d["M3"]["lo95"] > 0 and not worse),
        }

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print(json.dumps({"metrics": res["metrics"],
                      "acceptance_gate": res.get("acceptance_gate"),
                      "injection_verdict": res.get("injection_verdict")},
                     ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
