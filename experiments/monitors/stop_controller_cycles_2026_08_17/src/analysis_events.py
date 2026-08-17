"""Разбор пункта 1.4 задания и сравнение расписаний вызовов.

Первое: меняется ли прогноз именно в точках событий. Второе: годится ли событийное
расписание вызовов против равномерного при равном числе вызовов.

Запуск на боксе tesla:
  /home/user/ki_sft_judge/.venv/bin/python -u src/analysis_events.py \
      --pred outputs/main/preds_test.npz --meta outputs/main/test_traces_meta.json \
      --out outputs/main
"""

import argparse
import json
import os
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import common as C

SEED = 20260817
H = C.H_HORIZON
M_TYPES = C.M_TYPES


def per_trace_ranges(ranges):
    return [(int(a), int(b)) for a, b in ranges if b > a]


def updates(y, ranges):
    """Модуль обновления прогноза и отметка шага; первый шаг трассы исключён."""
    idx = []
    for a, b in ranges:
        if b - a >= 2:
            idx.append(np.arange(a + 1, b))
    idx = np.concatenate(idx)
    d = np.abs(y[idx] - y[idx - 1])
    return idx, d


def boot_diff(vals_ev, vals_no, tid_ev, tid_no, traces, n_rep, rng):
    """Бутстреп по трассам для разности средних модулей обновления."""
    by_ev = {}
    by_no = {}
    for t, v in zip(tid_ev, vals_ev):
        by_ev.setdefault(t, []).append(v)
    for t, v in zip(tid_no, vals_no):
        by_no.setdefault(t, []).append(v)
    keys = [t for t in traces if t in by_ev and t in by_no]
    ev = [np.asarray(by_ev[t]) for t in keys]
    no = [np.asarray(by_no[t]) for t in keys]
    n = len(keys)
    out = []
    for _ in range(n_rep):
        pick = rng.integers(0, n, n)
        a = np.concatenate([ev[i] for i in pick])
        b = np.concatenate([no[i] for i in pick])
        out.append(float(a.mean() - b.mean()))
    return np.asarray(out), n


def perm_test(d, is_ev, tid, n_rep, rng):
    """Перестановка событийных отметок внутри трассы с сохранением их числа.

    Перестановка делается сортировкой по паре из номера трассы и случайного ключа: внутри
    трассы порядок случаен, границы трасс не переезжают, число отметок сохраняется.
    """
    order = np.argsort(tid, kind="mergesort")
    d = d[order].astype(np.float64)
    lab0 = (is_ev[order] > 0.5)
    tid = tid[order]
    n = d.size
    tot = d.sum()
    k = int(lab0.sum())
    obs = float(d[lab0].mean() - d[~lab0].mean())
    ge = 0
    for _ in range(n_rep):
        keys = rng.random(n)
        o = np.lexsort((keys, tid))
        lab = lab0[o]
        s = d[lab].sum()
        val = s / k - (tot - s) / (n - k)
        if val >= obs:
            ge += 1
    return obs, (ge + 1.0) / (n_rep + 1.0)


def stratified_diff(d, is_ev, seg_tok, n_bins=10):
    """Разность средних внутри децилей длины сегмента, взвешенная по числу шагов."""
    qs = np.quantile(seg_tok, np.linspace(0, 1, n_bins + 1))
    qs[0] -= 1e-6
    qs[-1] += 1e-6
    rows, num, den = [], 0.0, 0.0
    for i in range(n_bins):
        m = (seg_tok > qs[i]) & (seg_tok <= qs[i + 1])
        a = d[m & (is_ev > 0.5)]
        b = d[m & (is_ev < 0.5)]
        if a.size < 10 or b.size < 10:
            continue
        w = float(m.sum())
        rows.append({"дециль": i, "низ длины": float(qs[i]), "верх длины": float(qs[i + 1]),
                     "шагов с событием": int(a.size), "шагов без события": int(b.size),
                     "среднее с событием": float(a.mean()),
                     "среднее без события": float(b.mean()),
                     "разность": float(a.mean() - b.mean())})
        num += w * (a.mean() - b.mean())
        den += w
    return rows, (num / den if den else float("nan"))


def schedule_series(y, ranges, calls_mask):
    """Удерживаемое значение прогноза: между вызовами держится последнее.

    Вызов в первой наблюдательной точке трассы делается всегда, иначе расписание не
    определено на её начале.
    """
    out = np.empty_like(y)
    for a, b in ranges:
        cm = calls_mask[a:b].copy()
        cm[0] = True
        vals = y[a:b]
        last = vals[0]
        for i in range(b - a):
            if cm[i]:
                last = vals[i]
            out[a + i] = last
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rep", type=int, default=2000)
    ap.add_argument("--only-plots", action="store_true",
                    help="перерисовать графики по готовым прогнозам, без статистики")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    z = np.load(a.pred)
    meta = json.load(open(a.meta, encoding="utf-8"))
    rng = np.random.default_rng(SEED)
    t0 = time.time()

    ranges = per_trace_ranges(z["ranges"])
    rem1, rem2 = z["rem1"], z["rem2"]
    has_ev, seg_tok, ev_kind = z["has_ev"], z["seg_tok"], z["ev_kind"]
    tid = z["tid"]
    lab_h = (rem1 < H).astype(float)
    res = {"модели": {}, "расписания": {}, "время": time.strftime("%Y-%m-%d %H:%M:%S %Z")}

    models = [m for m in ("M_gru_heads", "M0_mlp_heads", "B2_event_gbr",
                          "B1_time_linear") if ("%s__rem1" % m) in z]

    # ------------------------------- пункт 1.4: обновление прогноза и события
    for nm in ([] if a.only_plots else models):
        y = z["%s__rem1" % nm]
        p = z["%s__p_h" % nm]
        idx, d = updates(y, ranges)
        _idx2, dp = updates(p, ranges)
        ev = has_ev[idx]
        st = seg_tok[idx]
        tt = tid[idx]
        blk = {}
        for tag, dd in (("остаток", d), ("вероятность горизонта", dp)):
            a_ = dd[ev > 0.5]
            b_ = dd[ev < 0.5]
            boot, n_tr = boot_diff(a_, b_, tt[ev > 0.5], tt[ev < 0.5],
                                   sorted(set(tt.tolist())), a.rep,
                                   np.random.default_rng(SEED))
            obs, pval = perm_test(dd, ev, tt, a.rep, np.random.default_rng(SEED))
            rows, sdiff = stratified_diff(dd, ev, st)
            blk[tag] = {
                "шагов с событием": int(a_.size), "шагов без события": int(b_.size),
                "среднее с событием": float(a_.mean()),
                "среднее без события": float(b_.mean()),
                "медиана с событием": float(np.median(a_)),
                "медиана без события": float(np.median(b_)),
                "отношение медиан": float(np.median(a_) / max(np.median(b_), 1e-9)),
                "разность средних": float(a_.mean() - b_.mean()),
                "бутстреп низ": float(np.percentile(boot, 2.5)),
                "бутстреп верх": float(np.percentile(boot, 97.5)),
                "трасс в бутстрепе": n_tr,
                "перестановочный уровень": pval,
                "стратифицированная разность": sdiff,
                "по децилям длины": rows,
            }
        # разрез по типу события
        by_type = {}
        for k, tname in enumerate(M_TYPES):
            m = ev_kind[idx, k] > 0.5
            if m.sum() < 20:
                continue
            by_type[tname] = {"шагов": int(m.sum()), "среднее модуля": float(d[m].mean()),
                              "медиана модуля": float(np.median(d[m]))}
        blk["по типу события"] = by_type
        blk["среднее модуля без события"] = float(d[ev < 0.5].mean())
        res["модели"][nm] = blk
        print("1.4 разобран для %s, секунд %.1f" % (nm, time.time() - t0), flush=True)

    # --------------------------------------------- расписания вызовов
    for nm in ([] if a.only_plots else models):
        y = z["%s__rem1" % nm]
        p = z["%s__p_h" % nm]
        ev_mask = has_ev > 0.5
        n_pos = np.array([b - a_ for a_, b in ranges])
        n_calls_ev = np.array([max(1, int(ev_mask[a_:b].sum()) + (0 if ev_mask[a_] else 1))
                               for a_, b in ranges])
        unif = np.zeros(len(y), dtype=bool)
        for (a_, b), nc in zip(ranges, n_calls_ev):
            k = max(1, int(round((b - a_) / float(nc))))
            unif[a_:b][::k] = True
        y_all, p_all = y, p
        y_ev = schedule_series(y, ranges, ev_mask)
        p_ev = schedule_series(p, ranges, ev_mask)
        y_un = schedule_series(y, ranges, unif)
        p_un = schedule_series(p, ranges, unif)
        calls = {"Р_все": float(np.mean(n_pos)),
                 "Р_соб": float(np.mean(n_calls_ev)),
                 "Р_равн": float(np.mean([unif[a_:b].sum() + (0 if unif[a_] else 1)
                                          for a_, b in ranges]))}
        sch = {}
        for tag, yy, pp in (("Р_все", y_all, p_all), ("Р_соб", y_ev, p_ev),
                            ("Р_равн", y_un, p_un)):
            sch[tag] = {"M1_rem1": C.spearman(np.expm1(yy), rem1),
                        "M3_auc_H": C.auc(pp, lab_h),
                        "M2_rem1": C.mae_log1p(np.expm1(yy), rem1),
                        "вызовов на трассу": calls[tag]}
        # парный бутстреп по трассам для разности событийного и равномерного
        acc = {"M1_rem1": [], "M3_auc_H": []}
        for bi in C.boot_indices(z["ranges"], a.rep, np.random.default_rng(SEED),
                                 cap_per_trace=20):
            acc["M1_rem1"].append(C.spearman(np.expm1(y_ev[bi]), rem1[bi]) -
                                  C.spearman(np.expm1(y_un[bi]), rem1[bi]))
            acc["M3_auc_H"].append(C.auc(p_ev[bi], lab_h[bi]) -
                                   C.auc(p_un[bi], lab_h[bi]))
        sch["разность Р_соб минус Р_равн"] = {
            k: {"среднее": float(np.mean(v)), "низ": float(np.percentile(v, 2.5)),
                "верх": float(np.percentile(v, 97.5))} for k, v in acc.items()}
        sch["цена перехода Р_все в Р_соб"] = {
            "M1_rem1": sch["Р_соб"]["M1_rem1"] - sch["Р_все"]["M1_rem1"],
            "M3_auc_H": sch["Р_соб"]["M3_auc_H"] - sch["Р_все"]["M3_auc_H"],
            "доля сэкономленных вызовов": 1.0 - calls["Р_соб"] / calls["Р_все"]}
        res["расписания"][nm] = sch
        print("расписания разобраны для %s, секунд %.1f" % (nm, time.time() - t0),
              flush=True)

    if not a.only_plots:
        json.dump(res, open(os.path.join(a.out, "events_analysis.json"), "w",
                            encoding="utf-8"), ensure_ascii=False, indent=1,
                  default=float)

    # ------------------------------------------------------------- графики
    nm = "M_gru_heads" if "M_gru_heads__rem1" in z else models[0]
    y = z["%s__rem1" % nm]
    idx, d = updates(y, ranges)
    ev = has_ev[idx] > 0.5
    st = seg_tok[idx]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    ax = axes[0]
    bins = np.linspace(0, np.percentile(d, 99), 60)
    ax.hist(d[~ev], bins=bins, density=True, alpha=0.55, color="tab:blue",
            label="шаг без события")
    ax.hist(d[ev], bins=bins, density=True, alpha=0.55, color="tab:orange",
            label="шаг с событием")
    ax.set_xlabel("модуль обновления прогноза, шкала log1p")
    ax.set_ylabel("плотность")
    ax.set_title("распределение обновления прогноза остатка")
    ax.legend()
    ax = axes[1]
    xs = np.sort(d[~ev])
    ax.plot(xs, np.linspace(0, 1, xs.size), color="tab:blue", label="шаг без события")
    xs = np.sort(d[ev])
    ax.plot(xs, np.linspace(0, 1, xs.size), color="tab:orange", label="шаг с событием")
    ax.set_xlim(0, np.percentile(d, 99))
    ax.set_xlabel("модуль обновления прогноза")
    ax.set_ylabel("доля шагов не больше значения")
    ax.set_title("накопленные доли")
    ax.legend()
    ax = axes[2]
    rows, sdiff = stratified_diff(d, has_ev[idx], st)
    xs = [r["дециль"] + 1 for r in rows]
    ax.plot(xs, [r["среднее без события"] for r in rows], "s-", color="tab:blue",
            label="шаг без события")
    ax.plot(xs, [r["среднее с событием"] for r in rows], "o-", color="tab:orange",
            label="шаг с событием")
    ax.set_xlabel("дециль длины сегмента в токенах")
    ax.set_ylabel("среднее модуля обновления")
    ax.set_title("после контроля длины сегмента")
    ax.legend()
    fig.suptitle("Обновление прогноза остатка текущего цикла, модель %s, "
                 "отложенная часть" % nm)
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, "fig_update_by_event.png"), dpi=140)
    plt.close(fig)

    # примеры трасс
    cand = []
    for ti, (a_, b) in enumerate(ranges):
        n = b - a_
        if 25 <= n <= 90:
            ncom = int(np.sum(np.diff(rem1[a_:b]) > 0)) + 1
            if 3 <= ncom <= 10:
                cand.append((ti, a_, b, ncom))
    cand.sort(key=lambda x: (-x[3], x[0]))
    for j, (ti, a_, b, ncom) in enumerate(cand[:2]):
        pos_t = np.cumsum(seg_tok[a_:b])
        fig, ax = plt.subplots(2, 1, figsize=(11, 6.4), sharex=True,
                               gridspec_kw={"height_ratios": [2, 1]})
        xs = np.arange(b - a_)
        ax[0].plot(xs, rem1[a_:b], color="0.35", lw=2, label="факт остатка")
        ax[0].plot(xs, np.expm1(z["%s__rem1" % nm][a_:b]), color="tab:blue",
                   lw=1.8, label="прогноз %s" % nm)
        if "B2_event_gbr__rem1" in z:
            ax[0].plot(xs, np.expm1(z["B2_event_gbr__rem1"][a_:b]), color="tab:orange",
                       lw=1.2, alpha=0.8, label="бар Б2")
        bnd = np.flatnonzero(np.diff(rem1[a_:b]) > 0) + 1
        for x in bnd:
            ax[0].axvline(x - 0.5, color="tab:red", ls="--", lw=1.0, alpha=0.7)
        evx = np.flatnonzero(has_ev[a_:b] > 0.5)
        ax[0].plot(evx, np.zeros(evx.size), "|", color="tab:green", ms=12,
                   label="сегмент с событием")
        ax[0].set_ylabel("токенов до конца текущего цикла")
        ax[0].legend(loc="upper right", fontsize=8)
        ax[0].set_title("трасса %s, красным пунктиром границы циклов" %
                        meta[ti]["trace_uid"][:60])
        dy = np.abs(np.diff(np.log1p(np.maximum(np.expm1(z["%s__rem1" % nm][a_:b]), 0))))
        ax[1].bar(xs[1:], dy, color=["tab:green" if has_ev[a_ + i] > 0.5 else "0.7"
                                     for i in range(1, b - a_)])
        ax[1].set_ylabel("модуль обновления")
        ax[1].set_xlabel("номер наблюдательной точки трассы")
        fig.tight_layout()
        fig.savefig(os.path.join(a.out, "fig_trace_example_%d.png" % (j + 1)), dpi=140)
        plt.close(fig)
    print("графики записаны, секунд %.1f" % (time.time() - t0))
    print(json.dumps({k: res["модели"][k]["остаток"] for k in res["модели"]},
                     ensure_ascii=False, indent=1, default=float)[:2000])


if __name__ == "__main__":
    main()
