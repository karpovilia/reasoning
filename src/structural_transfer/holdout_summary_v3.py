"""HELD_OUT_FAMILY_BENCHMARK_TRANSFER_20261013: comparison tables, version 3.  Version 2 (review E2 of the launch package) changed
the metrics of the remainder; version 3 changes no formula and adds the support of the comparison with the reference on
its own carrier (positions, traces, questions by metric and END status) and says whether the two carriers coincide.

The simple control is the class Control of holdout_summary.py, imported unchanged: its formula and its fitting are
those fixed before any number was read.  Only the metrics of the remainder changed, for four reasons shown by
synthetic counterexamples of the review and repeated here as regressions:
  1  a trace with a confirmed cut enters with EVERY stored prefix, the last one included; the provable lower bound of
     the remainder after a prefix is what is stored after it (segments, span starts, characters), 0 after the last;
  2  the main remainder error is taken on all prefixes of traces with a natural end, the last one (remainder 0)
     included; prefixes with a non-zero remainder are shown apart;
  3  the END status of the accepted table V3 is used: natural, confirmed_truncated and unknown are never mixed; the
     row of unknown is descriptive (only the stored part is known);
  4  a contrast is computed on the common carrier: the prefixes where both forecasts exist are fixed BEFORE averaging
     prefix -> trace -> question, and the support is reported for every comparison.
Step losses keep the averaging of the E/S/E+S reports (positions of a question pooled, questions equal).  The point
head of the remainder gives no probability of exceeding a budget; none is derived.
Usage: python holdout_summary_v3.py CORPUS_DIR TARGETS_NPZ REGIMES_DIR EVAL_DIR OUT_JSON [--selftest-only]"""
import json
import sys
from pathlib import Path

import numpy as np

from .holdout_summary import BOOT, REGIMES, RNG_SEED, SEEDS, UNITS, Control, grouped, interval

STATUS = {"natural": 0, "confirmed_truncated": 1, "unknown": 2}


def metrics(D, pos, pred, avail=None):
    q, y, tr, st = D["question"][pos], D["gate"][pos], D["trace_of"][pos], D["status"][pos]
    out, sup = {}, {}
    m = y >= 0
    if m.any():
        gn = -np.log(np.clip(pred["gate"][m, :3], 1e-8, 1)[np.arange(m.sum()), y[m]])
        dn = -np.log(np.clip(pred["dest"][m], 1e-8, 1)[np.arange(m.sum()), np.maximum(D["dest"][pos][m], 0)])
        for name, val, mask in (("joint_nll", gn + (y[m] == 1) * dn, np.ones(m.sum(), bool)), ("step_nll", gn, np.ones(m.sum(), bool)), ("stay_nll", gn, y[m] == 0), ("switch_nll", gn, y[m] == 1),
                                ("end_nll", gn, y[m] == 2), ("type_at_switch_nll", dn, y[m] == 1)):
            if mask.any():
                out[name] = dict(zip(*[x.tolist() for x in grouped(val[mask], q[m][mask])]))
                sup[name] = dict(positions=int(mask.sum()), questions=int(len(np.unique(q[m][mask]))))
    ok = ~np.isnan(pred["rem"]).any(1)
    if avail is not None:
        ok &= avail

    def ptq(val, mm):                                                   # prefix -> trace -> question
        tn, first = np.unique(tr[mm], return_index=True)
        return dict(zip(*[x.tolist() for x in grouped(grouped(val, tr[mm])[1], q[mm][first])])), int(len(tn))

    nat = (st == 0) & D["rem_mask"][pos] & ok
    for label, mm in (("natural_all_prefixes", nat), ("natural_nonzero_remainder", nat & (D["rem"][pos, 0] > 0))):
        if mm.any():
            for u, unit in enumerate(UNITS):
                err = pred["rem"][mm, u] - D["rem"][pos, u][mm]
                for name, val in (("abs_error", np.abs(err)), ("share_underestimated", (err < 0).astype(float)), ("mean_shortfall", np.clip(-err, 0, None))):
                    out[f"{label}|{name}_{unit}"], ntr = ptq(val, mm)
            sup[label] = dict(positions=int(mm.sum()), traces=ntr, questions=int(len(np.unique(q[mm]))))
    for label, code in (("confirmed_truncated", 1), ("unknown_descriptive", 2)):
        mm = (st == code) & ok
        if mm.any():
            for u, unit in enumerate(UNITS):
                out[f"{label}|share_below_the_stored_remainder_{unit}"], ntr = ptq((pred["rem"][mm, u] < D["stored"][pos, u][mm]).astype(float), mm)
            sup[label] = dict(positions=int(mm.sum()), traces=ntr, questions=int(len(np.unique(q[mm]))))
    return out, sup


def carrier(*preds):
    ok = np.ones(len(preds[0]["rem"]), bool)
    for p in preds:
        ok &= ~np.isnan(p["rem"]).any(1)
    return ok


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


def selftest():
    def toy(n_pos, trace, question, status, rem, stored, rem_mask):
        return dict(question=np.array(question), gate=np.full(n_pos, -1), dest=np.zeros(n_pos, np.int64), trace_of=np.array(trace), status=np.array(status), rem=np.array(rem, float), stored=np.array(stored, float),
                    rem_mask=np.array(rem_mask))
    z3 = lambda v: [[x, x, x] for x in v]
    P = lambda v: dict(gate=np.full((len(v), 3), 1 / 3), dest=np.full((len(v), 7), 1 / 7), rem=np.array(z3(v), float))
    # 1: one cut trace, stored remainder [1, 0], forecast [0, 0]: both prefixes count, the share is 0.5
    D = toy(2, [0, 0], [0, 0], [1, 1], z3([0, 0]), z3([1, 0]), [False, False])
    assert metrics(D, np.arange(2), P([0, 0]))[0]["confirmed_truncated|share_below_the_stored_remainder_segments"] == {0: 0.5}
    # 2: natural trace, truth [1, 0], forecast [1, 9]: the error over all natural prefixes is 4.5
    D = toy(2, [0, 0], [0, 0], [0, 0], z3([1, 0]), z3([1, 0]), [True, True])
    o = metrics(D, np.arange(2), P([1, 9]))[0]
    assert o["natural_all_prefixes|abs_error_segments"] == {0: 4.5} and o["natural_nonzero_remainder|abs_error_segments"] == {0: 0.0}
    # 3: one unknown and one confirmed trace: the confirmed row is 1, the unknown row is separate
    D = toy(2, [0, 1], [0, 0], [2, 1], z3([0, 0]), z3([3, 3]), [False, False])
    o = metrics(D, np.arange(2), P([5, 0]))[0]
    assert o["confirmed_truncated|share_below_the_stored_remainder_segments"] == {0: 1.0} and o["unknown_descriptive|share_below_the_stored_remainder_segments"] == {0: 0.0}
    # 4: a question with two natural traces; the control has no forecast on the second; on the common carrier the difference of the error is 0, not 5
    D = toy(2, [0, 1], [0, 0], [0, 0], z3([4, 4]), z3([4, 4]), [True, True])
    a, b = P([4, 14]), P([4, np.nan])
    ok = carrier(a, b)
    c = contrast(metrics(D, np.arange(2), a, ok)[0], metrics(D, np.arange(2), b, ok)[0])["natural_all_prefixes|abs_error_segments"]
    assert c["difference"] == 0.0 and metrics(D, np.arange(2), a)[0]["natural_all_prefixes|abs_error_segments"] == {0: 5.0}
    print("SELFTEST PASSED: four regressions of the review")


def main(corpus, targets, regimes_dir, eval_dir, out, *flags):
    selftest()
    if "--selftest-only" in flags:
        return
    corpus, regimes_dir, eval_dir = Path(corpus), Path(regimes_dir), Path(eval_dir)
    meta = json.loads((corpus / "metadata.json").read_text())
    with np.load(corpus / "sequences.npz") as z:
        mode, span_start, dest, question, chars = z["mode"].astype(np.int64), z["span_start"], z["dest"].astype(np.int64), z["question"].astype(np.int64), np.rint(np.expm1(z["x"][:, 2].astype(np.float64) * 10))
    with np.load(targets) as z:
        gate, rem, rem_mask = z["gate"].astype(np.int64), z["rem"].astype(np.float64), z["rem_mask"]
    N = len(mode)
    trace_of, t_of, age, stored = np.zeros(N, np.int64), np.zeros(N, np.int64), np.zeros(N, np.int64), np.zeros((N, 3))
    for i, t in enumerate(meta["traces"]):
        a, n = t["offset"], t["n_segments"]
        trace_of[a:a + n], t_of[a:a + n] = i, np.arange(1, n + 1)
        st = np.flatnonzero(span_start[a:a + n])
        age[a:a + n] = np.arange(n) - st[np.searchsorted(st, np.arange(n), side="right") - 1] + 1
        ss, cc = span_start[a:a + n].astype(np.float64), chars[a:a + n]
        stored[a:a + n, 0], stored[a:a + n, 1], stored[a:a + n, 2] = n - 1 - np.arange(n), ss.sum() - ss.cumsum(), cc.sum() - cc.cumsum()
    assert np.allclose(stored[rem_mask], rem[rem_mask]), "what is stored after a prefix of a natural trace must equal its remainder target"
    ab, pb = np.floor(np.log2(np.maximum(1, age))).astype(np.int64), np.floor(np.log2(np.maximum(1, t_of))).astype(np.int64)
    off = np.array([t["offset"] for t in meta["traces"]] + [N])
    lens = np.diff(off)
    D = dict(N=N, K=len(meta["operators"]), question=question, gate=gate, dest=dest, rem=rem, rem_mask=rem_mask, trace_of=trace_of, stored=stored, left=stored[:, 0].astype(np.int64), k1=mode, k2=mode * 100 + ab,
             k3=(mode * 100 + ab) * 100 + pb)
    res = dict(task_id="HELD_OUT_FAMILY_BENCHMARK_TRANSFER_20261013", version=3,
               carrier="a comparison uses the prefixes where EVERY forecast of the comparison exists: both seeds of the hold-out fit and the control; with the reference also both seeds of the reference and its control", bootstrap=f"{BOOT} question resamples, numpy default_rng({RNG_SEED}) anew for every interval; marginal exploratory intervals",
               note="the evaluation part was already seen for the control before the 18 weights were fixed (control-only run of 05:31); the experiment is exploratory, not a fully blind new test",
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

    means = lambda mm: {k: float(np.mean(list(v.values()))) for k, v in mm.items()}
    ref_roster = [json.loads(l) for l in open(regimes_dir / "roster_reference.jsonl")]
    D["status"] = np.repeat(np.array([STATUS[r["end_status"]] for r in ref_roster]), lens)
    assert np.array_equal(D["status"] == 0, rem_mask), "the natural status of the roster must coincide with the known remainder target"
    ref_control = Control(D, positions(ref_roster, "train"))
    for r in REGIMES:
        roster = [json.loads(l) for l in open(regimes_dir / f"roster_{r}.jsonl")]
        ctl = Control(D, positions(roster, "train"))
        R = dict(train_traces=sum(1 for x in roster if x["role"] == "train"), sets={})
        for role in ("evaluation_main", "evaluation_familiar_question", "evaluation_unfamiliar_question"):
            pos = positions(roster, role)
            if not len(pos):
                continue
            rows = [x for x in roster if x["role"] == role]
            name = "test_part" if role == "evaluation_main" else "held_out_train_part"
            pc = ctl.predict(pos)
            ph = [take(load_export(f"{r}__seed{s}", name), pos) for s in SEEDS]
            ok = carrier(pc, *ph)                                       # fixed before any averaging
            mc, sup = metrics(D, pos, pc, ok)
            mh = [metrics(D, pos, p, ok)[0] for p in ph]
            S = dict(traces=len(rows), questions=len({x["question_sha256"] for x in rows}), positions=int(len(pos)), end_status={k: sum(1 for x in rows if x["end_status"] == k) for k in STATUS},
                     teacher={k: sum(1 for x in rows if x["teacher"] == k) for k in sorted({x["teacher"] for x in rows})}, positions_on_the_common_carrier=int(ok.sum()), support=sup, control=means(mc),
                     holdout_by_seed=[means(m) for m in mh], holdout_minus_control=contrast(mean_seeds(mh), mc), holdout_minus_control_by_seed=[{k: v["difference"] for k, v in contrast(m, mc).items()} for m in mh])
            if role == "evaluation_main":                               # the reference saw the group: change after the group is taken out of training and selection, not a pure effect of novelty
                pr = [take(load_export(f"reference__seed{s}", "test_part"), pos) for s in SEEDS]
                prc = ref_control.predict(pos)
                ok2 = carrier(pc, prc, *ph, *pr)
                mh2, mr2 = [metrics(D, pos, p, ok2)[0] for p in ph], [metrics(D, pos, p, ok2)[0] for p in pr]
                S.update(support_on_the_carrier_with_the_reference=metrics(D, pos, ph[0], ok2)[1], carrier_with_the_reference_equals_the_carrier_with_the_control=bool(np.array_equal(ok, ok2)))
                S.update(positions_on_the_carrier_with_the_reference=int(ok2.sum()), reference_by_seed=[means(m) for m in mr2], holdout_minus_reference=contrast(mean_seeds(mh2), mean_seeds(mr2)),
                         holdout_minus_reference_by_seed=[{k: v["difference"] for k, v in contrast(a, b).items()} for a, b in zip(mh2, mr2)],
                         control_of_regime_minus_control_of_reference=contrast(metrics(D, pos, pc, ok2)[0], metrics(D, pos, prc, ok2)[0]))
            R["sets"][role] = S
            c = S["holdout_minus_control"].get("joint_nll")
            print(r, role, "traces", S["traces"], "q", S["questions"], "holdout-control joint %+.4f %s" % (c["difference"], c["interval95"]) if c else "")
        res["regimes"][r] = R
    Path(out).write_text(json.dumps(res, indent=1, ensure_ascii=False) + "\n")
    print("HOLDOUT SUMMARY V3 DONE")


if __name__ == "__main__":
    main(*sys.argv[1:])
