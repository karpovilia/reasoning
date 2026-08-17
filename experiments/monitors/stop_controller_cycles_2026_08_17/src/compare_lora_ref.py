"""Сравнение лёгкой головы с принятым прогнозистом остатка на одних и тех же окнах.

Справочный бар Б3 предрегистрации. Окна те же, метки те же, разница только в
предсказателе: адаптер поверх базы на 35 миллиардов параметров против рекуррентной
головы на признаках событий.

Запуск на боксе tesla:
  /home/user/ki_sft_judge/.venv/bin/python -u src/compare_lora_ref.py \
      --pred outputs/main/preds_test.npz --meta outputs/main/test_traces_meta.json \
      --lora outputs/main/pred_lora_ref.jsonl --out outputs/main/lora_ref_compare.json
"""

import argparse
import json

import numpy as np

import common as C

H = C.H_HORIZON
SEED = 20260817


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--lora", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rep", type=int, default=2000)
    a = ap.parse_args()

    z = np.load(a.pred)
    meta = json.load(open(a.meta, encoding="utf-8"))
    tid, pos = z["tid"], z["pos"]
    key = {}
    for i in range(tid.size):
        key[(meta[int(tid[i])]["trace_uid"], int(pos[i]))] = i

    rows = [json.loads(l) for l in open(a.lora, encoding="utf-8")]
    idx, med, p_h, rem_l = [], [], [], []
    miss = 0
    for r in rows:
        tu, seg = r["uid"].rsplit("::", 1)
        k = key.get((tu, int(seg)))
        if k is None:
            miss += 1
            continue
        idx.append(k)
        med.append(r["med"])
        p_h.append(r["p_H"])
        rem_l.append(r["rem"])
    idx = np.asarray(idx)
    med = np.asarray(med, dtype=float)
    p_h = np.asarray(p_h, dtype=float)
    rem = z["rem1"][idx]
    lab = (rem < H).astype(float)
    dev = float(np.max(np.abs(np.asarray(rem_l) - rem))) if idx.size else float("nan")

    out = {"окон сопоставлено": int(idx.size), "окон не найдено": miss,
           "наибольшее расхождение цели": dev, "метрики": {}}
    out["метрики"]["B3_lora_noinj"] = {
        "M1_rem1": C.spearman(med, rem),
        "M2_rem1": C.mae_log1p(med, rem),
        "M3_auc_H": C.auc(p_h, lab)}
    names = [nm for nm in ("M_gru_heads", "M0_mlp_heads", "B2_event_gbr",
                         "B1_time_linear") if ("%s__rem1" % nm) in z]
    for nm in names:
        r1 = np.expm1(z["%s__rem1" % nm][idx])
        out["метрики"][nm] = {"M1_rem1": C.spearman(r1, rem),
                              "M2_rem1": C.mae_log1p(r1, rem),
                              "M3_auc_H": C.auc(z["%s__p_h" % nm][idx], lab)}

    # бутстреп по трассам на сопоставленных окнах
    tr_of = tid[idx]
    uniq = np.unique(tr_of)
    groups = [np.flatnonzero(tr_of == t) for t in uniq]
    rng = np.random.default_rng(SEED)
    acc = {nm: {"M1_rem1": [], "M3_auc_H": []} for nm in ["B3_lora_noinj"] + names}
    for _ in range(a.rep):
        pick = rng.integers(0, len(groups), len(groups))
        bi = np.concatenate([groups[i] for i in pick])
        acc["B3_lora_noinj"]["M1_rem1"].append(C.spearman(med[bi], rem[bi]))
        acc["B3_lora_noinj"]["M3_auc_H"].append(C.auc(p_h[bi], lab[bi]))
        for nm in names:
            r1 = np.expm1(z["%s__rem1" % nm][idx][bi])
            acc[nm]["M1_rem1"].append(C.spearman(r1, rem[bi]))
            acc[nm]["M3_auc_H"].append(C.auc(z["%s__p_h" % nm][idx][bi], lab[bi]))
    out["интервалы"] = {nm: {k: {"среднее": float(np.mean(v)),
                                 "низ": float(np.percentile(v, 2.5)),
                                 "верх": float(np.percentile(v, 97.5))}
                             for k, v in d.items()} for nm, d in acc.items()}
    out["разности"] = {}
    for nm in names:
        d = {}
        for k in ("M1_rem1", "M3_auc_H"):
            v = np.array(acc[nm][k]) - np.array(acc["B3_lora_noinj"][k])
            d[k] = {"среднее": float(np.mean(v)), "низ": float(np.percentile(v, 2.5)),
                    "верх": float(np.percentile(v, 97.5))}
        out["разности"]["%s минус B3_lora_noinj" % nm] = d
    json.dump(out, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps(out, ensure_ascii=False, indent=1)[:2500])


if __name__ == "__main__":
    main()
