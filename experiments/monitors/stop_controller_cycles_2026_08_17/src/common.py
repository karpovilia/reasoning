"""Общие части эксперимента stop_controller_cycles_2026_08_17: признаки, цели, метрики.

Признаки считаются только по прошлому. Наблюдательная точка — конец сегмента, события
сегмента стоят ровно в его конце, поэтому события текущего сегмента входят в историю, а
цель считается по консолидациям строго правее точки. Это дословно то же правило, что в
замороженном определении эксперимента cycle_end_lora_2026_08_16.
"""

import bisect
import json
import math
import os

import numpy as np

M_TYPES = ["backtrack", "verify", "branch", "subgoal_done", "failed_attempt", "commit"]
COMMIT = 5
H_HORIZON = 256.0
FAMILIES = ["gpt_oss_20b", "qwen36_35b", "qwen3_32b", "qwen3_30b", "other"]
EFFORTS = ["low", "medium", "high", ""]

FEATS = (
    ["t_tok", "log_t", "seg_tok", "seg_index", "cap_known", "cap_value", "frac_cap"]
    + ["fam_" + f for f in FAMILIES]
    + ["eff_" + (e or "none") for e in EFFORTS]
    + ["kind_toolloop"]
    + ["n_ev", "ev_rate_1k", "since_last_ev", "n_ev_200", "n_ev_500"]
    + ["cnt_" + t for t in M_TYPES]
    + ["since_" + t for t in M_TYPES]
    + ["n_cycles", "since_commit", "mean_cycle_len", "last_cycle_len", "n_ev_in_cycle"]
    + ["has_cand", "since_cand"]
    + ["op_%d" % i for i in range(6)] + ["op_unknown"]
    + ["cur_ev_" + t for t in M_TYPES] + ["cur_n_ev"]
)
NF = len(FEATS)


def load_traces(ds_dir, split, waves=("wave1",)):
    out = []
    with open(os.path.join(ds_dir, "traces_%s.jsonl" % split), encoding="utf-8") as fh:
        for line in fh:
            d = json.loads(line)
            if d["wave"] in waves:
                out.append(d)
    return out


def trace_arrays(rec):
    """Признаки всех сегментов трассы и цели в наблюдательных точках.

    Возвращает X формы [n_seg, NF], маску наблюдательных точек и цели.
    """
    n = rec["n_seg"]
    ends = rec["seg_end"]
    tok = rec["seg_tok"]
    ops = rec["seg_op"]
    T = rec["T"]
    cap = rec.get("cap") or 0
    cap_known = 1.0 if rec.get("cap") else 0.0
    fam = rec.get("family") or "other"
    fam_i = FAMILIES.index(fam) if fam in FAMILIES else len(FAMILIES) - 1
    eff = rec.get("effort") or ""
    eff = eff if eff in EFFORTS else ""
    eff_i = EFFORTS.index(eff)
    kind_tool = 1.0 if rec.get("kind") != "stream" else 0.0
    cand_seg = rec.get("cand_seg", -1)

    by_seg = [[] for _ in range(n)]
    for s, k in rec["events"]:
        if 0 <= s < n:
            by_seg[s].append(k)
    commits = [ends[s] for s, k in rec["events"] if k == COMMIT and 0 <= s < n]
    commits.sort()

    X = np.zeros((n, NF), dtype=np.float32)
    cnt = [0.0] * 6
    last_t = [-1.0] * 6
    ev_times = []
    n_ev = 0
    cyc_lens = []
    prev_commit_t = 0.0
    last_commit_t = -1.0
    n_cyc = 0
    ev_in_cyc = 0

    for i in range(n):
        t = ends[i]
        cur = by_seg[i]
        for k in cur:
            cnt[k] += 1.0
            last_t[k] = t
            ev_times.append(t)
            n_ev += 1
            ev_in_cyc += 1
            if k == COMMIT:
                cyc_lens.append(t - prev_commit_t)
                prev_commit_t = t
                last_commit_t = t
                n_cyc += 1
                ev_in_cyc = 0
        j = 0
        while j < len(ev_times) and ev_times[j] < t - 500.0:
            j += 1
        if j:
            del ev_times[:j]
        n500 = float(len(ev_times))
        n200 = float(sum(1 for te in ev_times if te >= t - 200.0))
        tsl = (t - max(last_t)) if n_ev else t
        f = [t, math.log1p(t), tok[i], float(i), cap_known, float(cap),
             (t / cap) if cap else -1.0]
        f += [1.0 if q == fam_i else 0.0 for q in range(len(FAMILIES))]
        f += [1.0 if q == eff_i else 0.0 for q in range(len(EFFORTS))]
        f += [kind_tool]
        f += [float(n_ev), n_ev / max(t, 1.0) * 1000.0, tsl, n200, n500]
        f += list(cnt)
        f += [(t - last_t[k]) if last_t[k] >= 0 else -1.0 for k in range(6)]
        f += [float(n_cyc), (t - last_commit_t) if last_commit_t >= 0 else t,
              float(np.mean(cyc_lens)) if cyc_lens else -1.0,
              float(cyc_lens[-1]) if cyc_lens else -1.0, float(ev_in_cyc)]
        has_cand = 1.0 if (0 <= cand_seg <= i) else 0.0
        f += [has_cand, (t - ends[cand_seg]) if has_cand else -1.0]
        op = ops[i]
        f += [1.0 if op == q else 0.0 for q in range(6)] + [1.0 if op < 0 else 0.0]
        f += [1.0 if k in cur else 0.0 for k in range(6)] + [float(len(cur))]
        X[i] = f

    mask = np.zeros(n, dtype=bool)
    if n >= 3:
        mask[1:n - 1] = True
    rem1 = np.zeros(n, dtype=np.float32)
    rem2 = np.zeros(n, dtype=np.float32)
    cl1 = np.zeros(n, dtype=np.float32)
    cl2 = np.zeros(n, dtype=np.float32)
    bnext = np.zeros(n, dtype=np.float32)
    for i in range(n):
        t = ends[i]
        p = bisect.bisect_right(commits, t)
        c1 = commits[p] if p < len(commits) else T
        c2 = commits[p + 1] if p + 1 < len(commits) else T
        rem1[i] = max(0.0, c1 - t)
        rem2[i] = max(0.0, c2 - t)
        cl1[i] = 1.0 if p < len(commits) else 0.0
        cl2[i] = 1.0 if p + 1 < len(commits) else 0.0
        if i + 1 < n and COMMIT in by_seg[i + 1]:
            bnext[i] = 1.0
    has_ev = np.array([1.0 if by_seg[i] else 0.0 for i in range(n)], dtype=np.float32)
    ev_kind = np.zeros((n, 6), dtype=np.float32)
    for i in range(n):
        for k in by_seg[i]:
            ev_kind[i, k] = 1.0
    return dict(X=X, mask=mask, rem1=rem1, rem2=rem2, closed1=cl1, closed2=cl2,
                bnext=bnext, has_ev=has_ev, ev_kind=ev_kind,
                seg_tok=np.asarray(tok, dtype=np.float32))


def build_matrix(traces):
    """Плоская матрица наблюдательных точек всех трасс и таблица целей."""
    Xs, ys, tid, pos = [], [], [], []
    keys = ["rem1", "rem2", "closed1", "closed2", "bnext", "has_ev", "seg_tok"]
    cols = {k: [] for k in keys}
    evk = []
    ranges = []
    start = 0
    seqs = []
    for ti, rec in enumerate(traces):
        a = trace_arrays(rec)
        m = a["mask"]
        k = int(m.sum())
        if k == 0:
            seqs.append(a)
            ranges.append((start, start))
            continue
        Xs.append(a["X"][m])
        for c in keys:
            cols[c].append(a[c][m])
        evk.append(a["ev_kind"][m])
        tid.append(np.full(k, ti, dtype=np.int32))
        pos.append(np.nonzero(m)[0].astype(np.int32))
        ranges.append((start, start + k))
        start += k
        seqs.append(a)
    out = dict(X=np.concatenate(Xs).astype(np.float32),
               tid=np.concatenate(tid), pos=np.concatenate(pos),
               ev_kind=np.concatenate(evk), ranges=ranges, seqs=seqs)
    for c in keys:
        out[c] = np.concatenate(cols[c]).astype(np.float32)
    return out


# ------------------------------------------------------------------- метрики
try:
    from scipy.stats import rankdata as _sp_rankdata
except Exception:
    _sp_rankdata = None


def rankdata(a):
    """Средние ранги со связками. Берётся реализация scipy, если она есть: у
    бутстрепа в две тысячи повторов чистый python на сотнях тысяч точек не считается."""
    a = np.asarray(a, dtype=np.float64)
    if _sp_rankdata is not None:
        return _sp_rankdata(a).astype(np.float64)
    n = a.size
    order = np.argsort(a, kind="mergesort")
    sa = a[order]
    ranks = np.empty(n, dtype=np.float64)
    start = 0
    brk = np.flatnonzero(np.diff(sa)) + 1
    for end in list(brk) + [n]:
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def spearman(x, y):
    if x.size < 3:
        return float("nan")
    rx, ry = rankdata(x), rankdata(y)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    d = math.sqrt(float((rx * rx).sum()) * float((ry * ry).sum()))
    return float((rx * ry).sum() / d) if d > 0 else float("nan")


def mae_log1p(pred, true):
    return float(np.mean(np.abs(np.log1p(np.maximum(pred, 0.0)) -
                                np.log1p(np.maximum(true, 0.0)))))


def auc(score, label):
    label = np.asarray(label)
    npos = float((label > 0.5).sum())
    nneg = float((label <= 0.5).sum())
    if npos == 0 or nneg == 0:
        return float("nan")
    r = rankdata(score)
    return float((r[label > 0.5].sum() - npos * (npos + 1) / 2.0) / (npos * nneg))


def pr_auc(score, label):
    label = np.asarray(label) > 0.5
    if label.sum() == 0:
        return float("nan")
    o = np.argsort(-np.asarray(score), kind="mergesort")
    lab = label[o]
    tp = np.cumsum(lab)
    fp = np.cumsum(~lab)
    prec = tp / np.maximum(tp + fp, 1)
    rec = tp / lab.sum()
    return float(np.sum(np.diff(np.concatenate([[0.0], rec])) * prec))


def boot_indices(ranges, n_rep, rng, cap_per_trace=None, base_rng=None):
    """Индексы бутстрепа по трассам: список массивов позиций."""
    tr = [np.arange(a, b) for (a, b) in ranges if b > a]
    if cap_per_trace:
        r2 = np.random.default_rng(20260817)
        tr = [t if t.size <= cap_per_trace
              else np.sort(r2.choice(t, cap_per_trace, replace=False)) for t in tr]
    n = len(tr)
    for _ in range(n_rep):
        pick = rng.integers(0, n, n)
        yield np.concatenate([tr[i] for i in pick])
