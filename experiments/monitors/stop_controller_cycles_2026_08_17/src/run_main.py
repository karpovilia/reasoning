"""Бары, обучение лёгких голов и оценка на отложенной части.

Порядок: сборка признаков, три бара, двухголовая рекуррентная модель, абляция без
рекуррентности, оценка на отложенной части с бутстрепом по трассам, гейт приёмки.

Запуск на боксе tesla:
  /home/user/ki_sft_judge/.venv/bin/python -u src/run_main.py \
      --ds /home/user/ki_cycles/data/ds --out /home/user/ki_cycles/outputs
"""

import argparse
import hashlib
import json
import os
import platform
import socket
import subprocess
import time

import numpy as np
import torch
import torch.nn as nn

import common as C

SEED = 20260817
H = C.H_HORIZON


def provenance(extra=None):
    try:
        card = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            text=True).strip().replace("\n", "; ")
    except Exception:
        card = "нет данных"
    import sklearn
    p = {"хост": socket.gethostname(), "тип карты": card,
         "время": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
         "время_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
         "python": platform.python_version(), "numpy": np.__version__,
         "torch": torch.__version__, "sklearn": sklearn.__version__,
         "зерно": SEED}
    if extra:
        p.update(extra)
    return p


def sha1_file(path):
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()[:16]


# ------------------------------------------------------------------- модели
class GruHeads(nn.Module):
    """Однонаправленная рекуррентная сеть с воротами и тремя головами."""

    def __init__(self, nf, hid=192):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(nf, hid), nn.ReLU())
        self.rnn = nn.GRU(hid, hid, num_layers=1, batch_first=True)
        self.head_rem = nn.Linear(hid, 2)
        self.head_h = nn.Linear(hid, 1)
        self.head_b = nn.Linear(hid, 1)

    def forward(self, x):
        z, _ = self.rnn(self.inp(x))
        return self.head_rem(z), self.head_h(z).squeeze(-1), self.head_b(z).squeeze(-1)


class MlpHeads(nn.Module):
    """Абляция: те же головы без памяти о ходе трассы."""

    def __init__(self, nf, hid=192):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(nf, hid), nn.ReLU(),
                                  nn.Linear(hid, hid), nn.ReLU())
        self.head_rem = nn.Linear(hid, 2)
        self.head_h = nn.Linear(hid, 1)
        self.head_b = nn.Linear(hid, 1)

    def forward(self, x):
        z = self.body(x)
        return self.head_rem(z), self.head_h(z).squeeze(-1), self.head_b(z).squeeze(-1)


def pack_batch(seqs, idx, mu, sd, dev, max_len=1600):
    lens = [min(seqs[i]["X"].shape[0], max_len) for i in idx]
    L = max(lens)
    B = len(idx)
    nf = seqs[idx[0]]["X"].shape[1]
    x = np.zeros((B, L, nf), dtype=np.float32)
    y1 = np.zeros((B, L), dtype=np.float32)
    y2 = np.zeros((B, L), dtype=np.float32)
    d1 = np.zeros((B, L), dtype=np.float32)
    d2 = np.zeros((B, L), dtype=np.float32)
    m = np.zeros((B, L), dtype=np.float32)
    for b, i in enumerate(idx):
        a = seqs[i]
        n = lens[b]
        x[b, :n] = (a["X"][:n] - mu) / sd
        y1[b, :n] = np.log1p(a["rem1"][:n])
        y2[b, :n] = np.log1p(a["rem2"][:n])
        d1[b, :n] = (a["rem1"][:n] < H).astype(np.float32)
        d2[b, :n] = a["bnext"][:n]
        m[b, :n] = a["mask"][:n].astype(np.float32)
    t = lambda z: torch.from_numpy(z).to(dev)
    return t(x), t(y1), t(y2), t(d1), t(d2), t(m)


def train_net(kind, seqs_tr, seqs_va, mu, sd, dev, log, epochs=40, bs=64, lr=2e-3,
              init=None):
    torch.manual_seed(SEED)
    nf = mu.shape[0]
    net = (GruHeads(nf) if kind == "gru" else MlpHeads(nf)).to(dev)
    if init:
        # Смещения голов ставятся в средние значения обучающей части: это только
        # ускоряет сходимость и не добавляет свободных величин.
        with torch.no_grad():
            net.head_rem.bias.copy_(torch.tensor(init["rem"], dtype=torch.float32))
            net.head_h.bias.fill_(float(init["logit_h"]))
            net.head_b.bias.fill_(float(init["logit_b"]))
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    hub = nn.HuberLoss(reduction="none")
    bce = nn.BCEWithLogitsLoss(reduction="none")
    rng = np.random.default_rng(SEED)
    n = len(seqs_tr)
    best, best_state, bad = float("inf"), None, 0
    for ep in range(epochs):
        net.train()
        order = rng.permutation(n)
        tot = 0.0
        for s in range(0, n, bs):
            idx = order[s:s + bs]
            x, y1, y2, d1, d2, m = pack_batch(seqs_tr, idx, mu, sd, dev)
            pr, ph, pb = net(x)
            w = m.sum().clamp(min=1.0)
            loss = ((hub(pr[..., 0], y1) + hub(pr[..., 1], y2)) * m).sum() / w
            loss = loss + (bce(ph, d1) * m).sum() / w + 0.5 * (bce(pb, d2) * m).sum() / w
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            tot += float(loss) * len(idx)
        sch.step()
        net.eval()
        vl, vn = 0.0, 0
        with torch.no_grad():
            for s in range(0, len(seqs_va), bs):
                idx = np.arange(s, min(s + bs, len(seqs_va)))
                x, y1, y2, d1, d2, m = pack_batch(seqs_va, idx, mu, sd, dev)
                pr, ph, pb = net(x)
                w = m.sum().clamp(min=1.0)
                l = ((hub(pr[..., 0], y1) + hub(pr[..., 1], y2)) * m).sum() / w
                l = l + (bce(ph, d1) * m).sum() / w + 0.5 * (bce(pb, d2) * m).sum() / w
                vl += float(l) * len(idx)
                vn += len(idx)
        vl /= max(vn, 1)
        log.append({"модель": kind, "эпоха": ep, "потери обучения": tot / n,
                    "потери проверки": vl})
        print("  %s эпоха %2d потери обучения %.4f проверки %.4f"
              % (kind, ep, tot / n, vl), flush=True)
        if vl < best - 1e-4:
            best, bad = vl, 0
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= 5:
                print("  ранняя остановка на эпохе %d" % ep, flush=True)
                break
    if best_state:
        net.load_state_dict(best_state)
    return net, best


def predict_net(net, seqs, mu, sd, dev, bs=32):
    """Плотный прогноз во всех наблюдательных точках, в порядке build_matrix.

    Обрезки последовательности здесь нет: она есть только в обучении, где длинные
    трассы режутся ради памяти, а прогноз обязан покрыть все точки.
    """
    net.eval()
    o1, o2, oh, ob = [], [], [], []
    with torch.no_grad():
        for s in range(0, len(seqs), bs):
            idx = np.arange(s, min(s + bs, len(seqs)))
            x, _y1, _y2, _d1, _d2, m = pack_batch(seqs, idx, mu, sd, dev,
                                                  max_len=1 << 20)
            pr, ph, pb = net(x)
            pr = pr.cpu().numpy()
            ph = torch.sigmoid(ph).cpu().numpy()
            pb = torch.sigmoid(pb).cpu().numpy()
            mm = m.cpu().numpy() > 0.5
            for b, i in enumerate(idx):
                o1.append(pr[b, :, 0][mm[b]])
                o2.append(pr[b, :, 1][mm[b]])
                oh.append(ph[b][mm[b]])
                ob.append(pb[b][mm[b]])
    return (np.concatenate(o1), np.concatenate(o2), np.concatenate(oh),
            np.concatenate(ob))


def metrics_of(pred, D):
    """pred: словарь с ключами rem1, rem2, p_h, p_b на шкале log1p для остатков."""
    r1 = np.expm1(pred["rem1"])
    r2 = np.expm1(pred["rem2"])
    return {
        "M1_rem1": C.spearman(r1, D["rem1"]),
        "M1_rem2": C.spearman(r2, D["rem2"]),
        "M2_rem1": C.mae_log1p(r1, D["rem1"]),
        "M2_rem2": C.mae_log1p(r2, D["rem2"]),
        "M3_auc_H": C.auc(pred["p_h"], (D["rem1"] < H).astype(float)),
        "M4_prauc_bnext": C.pr_auc(pred["p_b"], D["bnext"]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--boot-cap", type=int, default=20)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--limit-traces", type=int, default=0, help="смоук: обрезать корпус")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()

    tr = C.load_traces(a.ds, "train")
    va = C.load_traces(a.ds, "val")
    te = C.load_traces(a.ds, "test")
    te2 = C.load_traces(a.ds, "test", waves=("wave2",))
    if a.limit_traces:
        tr, va, te, te2 = (tr[:a.limit_traces], va[:a.limit_traces],
                           te[:a.limit_traces], te2[:a.limit_traces])
    print("трасс поток: обучение %d проверка %d отложенная %d; перенос %d"
          % (len(tr), len(va), len(te), len(te2)), flush=True)

    qtr = set(x["qid"] for x in tr)
    qva = set(x["qid"] for x in va)
    qte = set(x["qid"] for x in te) | set(x["qid"] for x in te2)
    split_check = {"вопросов обучение": len(qtr), "вопросов проверка": len(qva),
                   "вопросов отложенная": len(qte),
                   "пересечение обучение и отложенная": len(qtr & qte),
                   "пересечение проверка и отложенная": len(qva & qte),
                   "пересечение обучение и проверка": len(qtr & qva)}
    json.dump(split_check, open(os.path.join(a.out, "split_check.json"), "w"),
              ensure_ascii=False, indent=1)
    print("проверка разбиения:", split_check, flush=True)
    assert split_check["пересечение обучение и отложенная"] == 0
    assert split_check["пересечение проверка и отложенная"] == 0

    Dtr = C.build_matrix(tr)
    Dva = C.build_matrix(va)
    Dte = C.build_matrix(te)
    Dte2 = C.build_matrix(te2)
    print("точек: обучение %d проверка %d отложенная %d перенос %d, секунд %.1f"
          % (len(Dtr["rem1"]), len(Dva["rem1"]), len(Dte["rem1"]), len(Dte2["rem1"]),
             time.time() - t0), flush=True)

    mu = Dtr["X"].mean(0)
    sd = Dtr["X"].std(0)
    sd[sd < 1e-6] = 1.0

    y1tr = np.log1p(Dtr["rem1"])
    y2tr = np.log1p(Dtr["rem2"])
    d1tr = (Dtr["rem1"] < H).astype(np.float32)

    preds = {}   # имя -> словарь прогнозов на отложенной части
    preds2 = {}  # то же на срезе переноса
    predsv = {}  # то же на проверочной части, только для выбора порогов

    # --------------------------------------------------------------- бары
    med1, med2 = float(np.median(Dtr["rem1"])), float(np.median(Dtr["rem2"]))
    base = float(d1tr.mean())
    for nm, D, store in (("test", Dte, preds), ("transfer", Dte2, preds2),
                         ("val", Dva, predsv)):
        n = len(D["rem1"])
        store["B0_const"] = {"rem1": np.full(n, np.log1p(med1), dtype=np.float32),
                            "rem2": np.full(n, np.log1p(med2), dtype=np.float32),
                            "p_h": np.full(n, base, dtype=np.float32),
                            "p_b": np.full(n, float(Dtr["bnext"].mean()),
                                           dtype=np.float32)}

    from sklearn.linear_model import LinearRegression, LogisticRegression
    from sklearn.ensemble import (HistGradientBoostingRegressor,
                                  HistGradientBoostingClassifier)

    def time_only(D):
        """Голое прошедшее время: сами токены и их логарифм.

        Токены поделены на тысячу только ради устойчивости решателя логистической
        регрессии; на класс моделей и на порядок признаков это не влияет.
        """
        t = D["X"][:, C.FEATS.index("t_tok")]
        return np.stack([t / 1000.0, np.log1p(t)], 1)

    print("бар Б1, линейная модель на прошедшем времени", flush=True)
    l1 = LinearRegression().fit(time_only(Dtr), y1tr)
    l2 = LinearRegression().fit(time_only(Dtr), y2tr)
    lc = LogisticRegression(max_iter=1000).fit(time_only(Dtr), d1tr)
    lb = LogisticRegression(max_iter=1000).fit(time_only(Dtr), Dtr["bnext"])
    for D, store in ((Dte, preds), (Dte2, preds2), (Dva, predsv)):
        Z = time_only(D)
        store["B1_time_linear"] = {"rem1": l1.predict(Z), "rem2": l2.predict(Z),
                                  "p_h": lc.predict_proba(Z)[:, 1],
                                  "p_b": lb.predict_proba(Z)[:, 1]}

    print("бар Б2, событийный градиентный бустинг", flush=True)
    hp = dict(max_iter=300, learning_rate=0.1, max_leaf_nodes=63,
              min_samples_leaf=100, l2_regularization=1.0, random_state=SEED)
    g1 = HistGradientBoostingRegressor(**hp).fit(Dtr["X"], y1tr)
    g2 = HistGradientBoostingRegressor(**hp).fit(Dtr["X"], y2tr)
    gc = HistGradientBoostingClassifier(**hp).fit(Dtr["X"], d1tr)
    gb = HistGradientBoostingClassifier(**hp).fit(Dtr["X"], Dtr["bnext"])
    for D, store in ((Dte, preds), (Dte2, preds2), (Dva, predsv)):
        store["B2_event_gbr"] = {"rem1": g1.predict(D["X"]), "rem2": g2.predict(D["X"]),
                                 "p_h": gc.predict_proba(D["X"])[:, 1],
                                 "p_b": gb.predict_proba(D["X"])[:, 1]}
    print("бары готовы, секунд %.1f" % (time.time() - t0), flush=True)

    # -------------------------------------------------------------- модели
    log = []
    for kind, name in (("gru", "M_gru_heads"), ("mlp", "M0_mlp_heads")):
        print("обучение %s" % name, flush=True)
        pb0 = float(np.clip(Dtr["bnext"].mean(), 1e-4, 1 - 1e-4))
        ph0 = float(np.clip(d1tr.mean(), 1e-4, 1 - 1e-4))
        init = {"rem": [float(y1tr.mean()), float(y2tr.mean())],
                "logit_h": float(np.log(ph0 / (1 - ph0))),
                "logit_b": float(np.log(pb0 / (1 - pb0)))}
        net, vbest = train_net(kind, Dtr["seqs"], Dva["seqs"], mu, sd, dev, log,
                               epochs=a.epochs, init=init)
        for D, store in ((Dte, preds), (Dte2, preds2), (Dva, predsv)):
            r1, r2, ph, pb = predict_net(net, D["seqs"], mu, sd, dev)
            store[name] = {"rem1": r1, "rem2": r2, "p_h": ph, "p_b": pb}
        torch.save(net.state_dict(), os.path.join(a.out, "net_%s.pt" % kind))
        print("%s обучена, лучшие потери проверки %.4f, секунд %.1f"
              % (name, vbest, time.time() - t0), flush=True)
    json.dump(log, open(os.path.join(a.out, "train_log.json"), "w"),
              ensure_ascii=False, indent=1)

    # -------------------------------------------------------------- оценка
    res = {"провенанс": provenance({"устройство": dev}),
           "разбиение": split_check,
           "объём": {"трасс обучение": len(tr), "трасс проверка": len(va),
                     "трасс отложенная": len(te), "трасс перенос": len(te2),
                     "точек обучение": int(len(Dtr["rem1"])),
                     "точек отложенная": int(len(Dte["rem1"])),
                     "точек перенос": int(len(Dte2["rem1"]))},
           "описание целей": {
               "доля закрытых цикл1": float(Dte["closed1"].mean()),
               "доля закрытых цикл2": float(Dte["closed2"].mean()),
               "доля окон с концом цикла в горизонте": float((Dte["rem1"] < H).mean()),
               "доля шагов с событием": float(Dte["has_ev"].mean()),
               "медиана остатка цикл1": float(np.median(Dte["rem1"])),
               "медиана остатка цикл2": float(np.median(Dte["rem2"]))},
           "метрики": {}, "метрики перенос": {}, "метрики закрытые": {}}

    for nm, p in preds.items():
        res["метрики"][nm] = metrics_of(p, Dte)
    for nm, p in preds2.items():
        res["метрики перенос"][nm] = metrics_of(p, Dte2)
    sel = (Dte["closed1"] > 0.5) & (Dte["closed2"] > 0.5)
    Dsel = {k: Dte[k][sel] for k in ("rem1", "rem2", "bnext")}
    for nm, p in preds.items():
        res["метрики закрытые"][nm] = metrics_of({k: v[sel] for k, v in p.items()}, Dsel)

    # пороги детекторов выбираются по проверочной части, на отложенной не подбираются
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

    labv_h = (Dva["rem1"] < H).astype(float)
    labt_h = (Dte["rem1"] < H).astype(float)
    det = {}
    for nm in names:
        pv = predsv[nm]
        _f1h, th_h = best_f1(pv["p_h"], labv_h)
        _f1b, th_b = best_f1(pv["p_b"], Dva["bnext"])
        det[nm] = {
            "Д1 конец цикла в горизонте 256": dict(
                prf(preds[nm]["p_h"], labt_h, th_h),
                AUC=C.auc(preds[nm]["p_h"], labt_h),
                PR_AUC=C.pr_auc(preds[nm]["p_h"], labt_h),
                доля_положительных=float(labt_h.mean())),
            "Д2 консолидация в следующем сегменте": dict(
                prf(preds[nm]["p_b"], Dte["bnext"], th_b),
                AUC=C.auc(preds[nm]["p_b"], Dte["bnext"]),
                PR_AUC=C.pr_auc(preds[nm]["p_b"], Dte["bnext"]),
                доля_положительных=float(Dte["bnext"].mean())),
        }
    res["детекторы"] = det

    # разрез по семейству модели
    fam_pt = np.array([te[int(i)]["family"] for i in Dte["tid"]])
    by_fam = {}
    for f in sorted(set(fam_pt.tolist())):
        m = fam_pt == f
        if m.sum() < 500:
            continue
        Dsub = {k: Dte[k][m] for k in ("rem1", "rem2", "bnext")}
        by_fam[f] = {"точек": int(m.sum()),
                     "трасс": int(len(set(Dte["tid"][m].tolist())))}
        for nm in names:
            by_fam[f][nm] = metrics_of({k: v[m] for k, v in preds[nm].items()}, Dsub)
    res["по семействам"] = by_fam

    # бутстреп по трассам, парные разности с барами
    rng = np.random.default_rng(SEED)
    names = list(preds.keys())
    acc = {nm: {k: [] for k in ("M1_rem1", "M1_rem2", "M3_auc_H")} for nm in names}
    lab = (Dte["rem1"] < H).astype(float)
    t1 = time.time()
    for bi in C.boot_indices(Dte["ranges"], a.boot, rng, cap_per_trace=a.boot_cap):
        for nm in names:
            p = preds[nm]
            acc[nm]["M1_rem1"].append(C.spearman(p["rem1"][bi], Dte["rem1"][bi]))
            acc[nm]["M1_rem2"].append(C.spearman(p["rem2"][bi], Dte["rem2"][bi]))
            acc[nm]["M3_auc_H"].append(C.auc(p["p_h"][bi], lab[bi]))
    print("бутстреп %d повторов, секунд %.1f" % (a.boot, time.time() - t1), flush=True)
    ci = {}
    for nm in names:
        ci[nm] = {k: {"среднее": float(np.mean(v)),
                      "низ": float(np.percentile(v, 2.5)),
                      "верх": float(np.percentile(v, 97.5))}
                  for k, v in acc[nm].items()}
    res["интервалы"] = ci
    res["разности"] = {}
    for nm in names:
        if nm.startswith("B"):
            continue
        for bar in ("B0_const", "B1_time_linear", "B2_event_gbr"):
            d = {}
            for k in ("M1_rem1", "M1_rem2", "M3_auc_H"):
                v = np.array(acc[nm][k]) - np.array(acc[bar][k])
                d[k] = {"среднее": float(np.mean(v)),
                        "низ": float(np.percentile(v, 2.5)),
                        "верх": float(np.percentile(v, 97.5))}
            res["разности"]["%s минус %s" % (nm, bar)] = d

    best_bar = max(("B1_time_linear", "B2_event_gbr"),
                   key=lambda b: res["метрики"][b]["M1_rem1"])
    gate = {}
    for k in ("M1_rem1", "M1_rem2", "M3_auc_H"):
        lo = res["разности"]["M_gru_heads минус %s" % best_bar][k]["низ"]
        gate[k] = {"лучший бар": best_bar, "низ разности": lo, "пройден": bool(lo > 0)}
    gate["итог"] = bool(all(gate[k]["пройден"] for k in
                            ("M1_rem1", "M1_rem2", "M3_auc_H")))
    res["гейт"] = gate

    np.savez_compressed(
        os.path.join(a.out, "preds_test.npz"),
        rem1=Dte["rem1"], rem2=Dte["rem2"], bnext=Dte["bnext"],
        has_ev=Dte["has_ev"], seg_tok=Dte["seg_tok"], ev_kind=Dte["ev_kind"],
        tid=Dte["tid"], pos=Dte["pos"], closed1=Dte["closed1"], closed2=Dte["closed2"],
        ranges=np.array(Dte["ranges"], dtype=np.int64),
        **{("%s__%s" % (nm, k)): v for nm, p in preds.items() for k, v in p.items()})
    json.dump([{k: t[k] for k in ("trace_uid", "qid", "family", "corpus", "tag",
                                  "benchmark", "n_seg", "T")} for t in te],
              open(os.path.join(a.out, "test_traces_meta.json"), "w"),
              ensure_ascii=False)
    json.dump(res, open(os.path.join(a.out, "main_metrics.json"), "w"),
              ensure_ascii=False, indent=1)
    print(json.dumps({"метрики": res["метрики"], "гейт": res["гейт"]},
                     ensure_ascii=False, indent=1))
    print("всего секунд %.1f" % (time.time() - t0))


if __name__ == "__main__":
    main()
