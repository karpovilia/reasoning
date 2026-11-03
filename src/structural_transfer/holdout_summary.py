"""HELD_OUT_FAMILY_BENCHMARK_TRANSFER_20261013: the fixed simple control and the comparison tables (CPU, saved exports only).

Control (fixed before any result, no search): state = (type of the current span, floor(log2(age of the span in
segments)), floor(log2(number of the segment))), fitted only on the training part of the regime.  Hierarchy of cells
(type, age bin, number bin) -> (type, age bin) -> type -> all.  Position weights: unit sum per question inside the
training part.  Step STAY / SWITCH / END: weighted class frequencies, every level smoothed by one unit of weight of the
distribution of its parent, the top level by a uniform prior of total mass 1.  Type of the new span at SWITCH: the same
over the seven types among observed switches.  Remainder (segments, span starts, characters): weighted median over
positions with a known natural remainder in the most detailed cell that holds at least 20 different training questions
with such a target, else its parent, else absent.
Evaluation sets come from the roster of the regime; every metric applies the mask of its own target.  Step losses use
the averaging of the E/S/E+S reports (positions of a question pooled, questions equal); remainder and lower-bound
metrics use the rule of TAIL_RISK_METRICS (indicator at every prefix, then prefix -> trace -> question).
Usage: python holdout_summary.py CORPUS_DIR TARGETS_NPZ REGIMES_DIR EVAL_DIR OUT_JSON [--control-only]"""
import json
import sys
from pathlib import Path

import numpy as np

BOOT, RNG_SEED, MIN_Q = 2000, 20261013, 20
SEEDS = (17, 29)
REGIMES = ["family_gpt-oss", "family_Qwen", "family_DeepSeek", "family_Gemma", "benchmark_math500", "benchmark_hle", "benchmark_bbh", "benchmark_gpqa_diamond"]
UNITS = ("segments", "span_starts", "characters")


def interval(v):
    v = np.asarray(v, float)
    if len(v) < 2:
        return None
    rng = np.random.default_rng(RNG_SEED)
    m = v[rng.integers(0, len(v), (BOOT, len(v)))].mean(1)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def grouped(values, group):
    names, inv = np.unique(group, return_inverse=True)
    return names, np.bincount(inv, weights=values, minlength=len(names)) / np.bincount(inv, minlength=len(names))


def wmedian(v, w):
    o = np.argsort(v, kind="stable")
    c = np.cumsum(w[o])
    return float(v[o][min(np.searchsorted(c, 0.5 * c[-1] - 1e-12), len(v) - 1)])


class Control:
    def __init__(self, D, train):
        q = D["question"][train]
        _, inv, cnt = np.unique(q, return_inverse=True, return_counts=True)
        self.w = np.zeros(D["N"])
        self.w[train] = 1.0 / cnt[inv]                                  # unit sum per question inside the training part
        self.D = D
        k3, k2, k1 = D["k3"], D["k2"], D["k1"]
        self.gate = self._dist(train[D["gate"][train] >= 0], D["gate"], 3)
        self.dest = self._dist(train[D["gate"][train] == 1], D["dest"], D["K"])
        nat = train[D["rem_mask"][train]]
        self.rem = {}
        for name, key in (("k3", k3), ("k2", k2), ("k1", k1), ("all", np.zeros(D["N"], np.int64))):
            o = nat[np.argsort(key[nat], kind="stable")]
            ks, st = np.unique(key[o], return_index=True)
            t = {}
            for kk, a, b in zip(ks, st, list(st[1:]) + [len(o)]):
                idx = o[a:b]
                if len(np.unique(D["question"][idx])) >= MIN_Q:
                    t[int(kk)] = [wmedian(D["rem"][idx, u], self.w[idx]) for u in range(3)]
            self.rem[name] = t

    def _dist(self, pos, target, C):
        D, w = self.D, self.w
        y = target[pos]
        top = (np.bincount(y, weights=w[pos], minlength=C) + 1.0 / C) / (w[pos].sum() + 1.0)
        tables = {"all": top}
        parent_of = lambda name, kk: top if name == "k1" else tables["k1"].get(kk // 100 if name == "k2" else None, top) if name == "k2" else tables["k2"].get(kk // 100, tables["k1"].get(kk // 10000, top))
        for name in ("k1", "k2", "k3"):
            key = D[name][pos]
            ks, inv = np.unique(key, return_inverse=True)
            c = np.zeros((len(ks), C))
            np.add.at(c, (inv, y), w[pos])
            t = {}
            for j, kk in enumerate(ks):
                par = parent_of(name, int(kk))
                t[int(kk)] = (c[j] + par) / (c[j].sum() + 1.0)
            tables[name] = t
        return tables

    def predict(self, pos):
        D = self.D
        out = {}
        for what, tables, C in (("gate", self.gate, 3), ("dest", self.dest, D["K"])):
            p = np.tile(tables["all"], (len(pos), 1))
            for name in ("k1", "k2", "k3"):                              # the most detailed fitted cell wins
                key = D[name][pos]
                for kk in np.unique(key):
                    if int(kk) in tables[name]:
                        p[key == kk] = tables[name][int(kk)]
            out[what] = p
        r = np.full((len(pos), 3), np.nan)
        for name, key in (("all", np.zeros(D["N"], np.int64)), ("k1", D["k1"]), ("k2", D["k2"]), ("k3", D["k3"])):
            kv = key[pos]
            for kk in np.unique(kv):
                if int(kk) in self.rem[name]:
                    r[kv == kk] = self.rem[name][int(kk)]
        out["rem"] = r
        return out


def metrics(D, pos, pred):
    """Per-question values of every metric on the positions pos (corpus indices) for forecasts pred (aligned with pos)."""
    q, y = D["question"][pos], D["gate"][pos]
    out, sup = {}, {}
    m = y >= 0
    gn = -np.log(np.clip(pred["gate"][m, :3], 1e-8, 1)[np.arange(m.sum()), y[m]])
    dn = -np.log(np.clip(pred["dest"][m], 1e-8, 1)[np.arange(m.sum()), np.maximum(D["dest"][pos][m], 0)])
    joint = gn + (y[m] == 1) * dn
    for name, val, mask in (("joint_nll", joint, np.ones(m.sum(), bool)), ("step_nll", gn, np.ones(m.sum(), bool)), ("stay_nll", gn, y[m] == 0), ("switch_nll", gn, y[m] == 1), ("end_nll", gn, y[m] == 2),
                            ("type_at_switch_nll", dn, y[m] == 1)):
        if mask.any():
            out[name] = dict(zip(*[x.tolist() for x in grouped(val[mask], q[m][mask])]))
            sup[name] = int(mask.sum())
    live = D["rem_mask"][pos] & (D["rem"][pos, 0] > 0) & ~np.isnan(pred["rem"][:, 0])
    if live.any():
        tr = D["trace_of"][pos][live]
        tnames, first = np.unique(tr, return_index=True)
        tq = q[live][first]
        for u, unit in enumerate(UNITS):
            err = pred["rem"][live, u] - D["rem"][pos, u][live]
            for name, val in ((f"remainder_abs_error_{unit}", np.abs(err)), (f"remainder_share_underestimated_{unit}", (err < 0).astype(float)), (f"remainder_mean_shortfall_{unit}", np.clip(-err, 0, None))):
                per_trace = grouped(val, tr)[1]
                out[name] = dict(zip(*[x.tolist() for x in grouped(per_trace, tq)]))
        sup["remainder_positions"], sup["remainder_traces"] = int(live.sum()), int(len(tnames))
    cut = ~D["rem_mask"][pos] & (D["left"][pos] > 0) & ~np.isnan(pred["rem"][:, 0])
    if cut.any():
        tr = D["trace_of"][pos][cut]
        tnames, first = np.unique(tr, return_index=True)
        per_trace = grouped((pred["rem"][cut, 0] < D["left"][pos][cut]).astype(float), tr)[1]
        out["share_below_the_provable_lower_bound_segments"] = dict(zip(*[x.tolist() for x in grouped(per_trace, q[cut][first])]))
        sup["lower_bound_traces"] = int(len(tnames))
    return out, sup


def contrast(a, b):
    out = {}
    for k in a:
        if k in b:
            qs = sorted(set(a[k]) & set(b[k]))
            if qs:
                d = np.array([a[k][x] - b[k][x] for x in qs])
                out[k] = dict(questions=len(qs), first=float(np.mean([a[k][x] for x in qs])), second=float(np.mean([b[k][x] for x in qs])), difference=float(d.mean()), interval95=interval(d),
                              share_of_questions_below_zero=float((d < 0).mean()))
    return out


def mean_seeds(ms):
    out = {}
    for k in ms[0]:
        if all(k in m for m in ms):
            qs = set.intersection(*[set(m[k]) for m in ms])
            out[k] = {x: float(np.mean([m[k][x] for m in ms])) for x in qs}
    return out


def main(corpus, targets, regimes_dir, eval_dir, out, *flags):
    corpus, regimes_dir, eval_dir = Path(corpus), Path(regimes_dir), Path(eval_dir)
    meta = json.loads((corpus / "metadata.json").read_text())
    with np.load(corpus / "sequences.npz") as z:
        mode, span_start, dest, question = z["mode"].astype(np.int64), z["span_start"], z["dest"].astype(np.int64), z["question"].astype(np.int64)
    with np.load(targets) as z:
        gate, rem, rem_mask = z["gate"].astype(np.int64), z["rem"].astype(np.float64), z["rem_mask"]
    N = len(mode)
    trace_of, t_of, age, left = np.zeros(N, np.int64), np.zeros(N, np.int64), np.zeros(N, np.int64), np.zeros(N, np.int64)
    for i, t in enumerate(meta["traces"]):
        a, n = t["offset"], t["n_segments"]
        trace_of[a:a + n], t_of[a:a + n], left[a:a + n] = i, np.arange(1, n + 1), n - 1 - np.arange(n)
        st = np.flatnonzero(span_start[a:a + n])
        age[a:a + n] = np.arange(n) - st[np.searchsorted(st, np.arange(n), side="right") - 1] + 1
    ab, pb = np.floor(np.log2(np.maximum(1, age))).astype(np.int64), np.floor(np.log2(np.maximum(1, t_of))).astype(np.int64)
    D = dict(N=N, K=len(meta["operators"]), question=question, gate=gate, dest=dest, rem=rem, rem_mask=rem_mask, trace_of=trace_of, left=left, k1=mode, k2=mode * 100 + ab, k3=(mode * 100 + ab) * 100 + pb)
    off = np.array([t["offset"] for t in meta["traces"]] + [N])
    res = dict(task_id="HELD_OUT_FAMILY_BENCHMARK_TRANSFER_20261013", control=__doc__.split("Control")[1].split("Evaluation sets")[0].strip(), bootstrap=f"{BOOT} question resamples, numpy default_rng({RNG_SEED}) anew for every interval",
               seeds=list(SEEDS), regimes={})

    def positions(roster, role):
        tr = [i for i, r in enumerate(roster) if r["role"] == role]
        return np.concatenate([np.arange(off[i], off[i + 1]) for i in tr]) if tr else np.zeros(0, np.int64)

    def load_export(fit, name):
        with np.load(eval_dir / fit / f"export_{name}.npz") as z:
            return dict(index=z["index"], gate=z["gate"], dest=z["dest"], rem=z["rem"].astype(np.float64))

    def take(e, pos):
        j = np.searchsorted(e["index"], pos)
        assert np.array_equal(e["index"][j], pos)
        return dict(gate=e["gate"][j], dest=e["dest"][j], rem=e["rem"][j])

    ref_roster = [json.loads(l) for l in open(regimes_dir / "roster_reference.jsonl")]
    ref_control = Control(D, positions(ref_roster, "train"))
    for r in REGIMES:
        roster = [json.loads(l) for l in open(regimes_dir / f"roster_{r}.jsonl")]
        train = positions(roster, "train")
        ctl = Control(D, train)
        R = dict(train_traces=sum(1 for x in roster if x["role"] == "train"), train_positions=int(len(train)), control_cells=dict(step={k: len(v) if isinstance(v, dict) else 1 for k, v in ctl.gate.items()},
                                                                                                                              remainder={k: len(v) for k, v in ctl.rem.items()}), sets={})
        for role in ("evaluation_main", "evaluation_familiar_question", "evaluation_unfamiliar_question"):
            pos = positions(roster, role)
            if not len(pos):
                continue
            rows = [x for x in roster if x["role"] == role]
            S = dict(traces=len(rows), questions=len({x["question_sha256"] for x in rows}), positions=int(len(pos)), end_status={k: sum(1 for x in rows if x["end_status"] == k) for k in ("natural", "confirmed_truncated", "unknown")},
                     teacher={k: sum(1 for x in rows if x["teacher"] == k) for k in sorted({x["teacher"] for x in rows})})
            mc, sup = metrics(D, pos, ctl.predict(pos))
            S["support"] = sup
            S["control"] = {k: float(np.mean(list(v.values()))) for k, v in mc.items()}
            if "--control-only" not in flags:
                name = "test_part" if role == "evaluation_main" else "held_out_train_part"
                mh = [metrics(D, pos, take(load_export(f"{r}__seed{s}", name), pos))[0] for s in SEEDS]
                S["holdout_by_seed"] = [{k: float(np.mean(list(v.values()))) for k, v in m.items()} for m in mh]
                S["holdout_minus_control"] = contrast(mean_seeds(mh), mc)
                S["holdout_minus_control_by_seed"] = [{k: v["difference"] for k, v in contrast(m, mc).items()} for m in mh]
                if role == "evaluation_main":                           # the reference saw the group; shown next to the main contrast with that caveat
                    mr = [metrics(D, pos, take(load_export(f"reference__seed{s}", "test_part"), pos))[0] for s in SEEDS]
                    mrc = metrics(D, pos, ref_control.predict(pos))[0]
                    S["reference_by_seed"] = [{k: float(np.mean(list(v.values()))) for k, v in m.items()} for m in mr]
                    S["holdout_minus_reference"] = contrast(mean_seeds(mh), mean_seeds(mr))
                    S["reference_control"] = {k: float(np.mean(list(v.values()))) for k, v in mrc.items()}
                    S["control_of_regime_minus_control_of_reference"] = contrast(mc, mrc)
            R["sets"][role] = S
            c = S.get("holdout_minus_control", {}).get("joint_nll")
            print(r, role, "traces", S["traces"], "q", S["questions"], "control joint %.4f" % S["control"]["joint_nll"], ("holdout-control %+.4f %s" % (c["difference"], c["interval95"])) if c else "")
        res["regimes"][r] = R
    Path(out).write_text(json.dumps(res, indent=1, ensure_ascii=False) + "\n")
    print("HOLDOUT SUMMARY DONE")


if __name__ == "__main__":
    main(*sys.argv[1:])
