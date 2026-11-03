#!/usr/bin/env python3
"""gate analyze v3 for the gpt-oss effort controller."""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict

import numpy as np

SPEND_KEY = "total_tokens"
CAP_KINDS = ("cap", "capl", "capdup")


def cond_parts(name):
    """cap_512 -> ('cap', None, 512); v3_q30_1024 -> ('v3', 0.30, 1024)."""
    p = name.split("_")
    if p[0] in CAP_KINDS:
        return p[0], None, int(p[1])
    return p[0], int(p[1][1:]) / 100.0, int(p[2])


def per_question_table(recs):
    tab = defaultdict(dict)
    for r in recs:
        tab[r["question_id"]][r["condition"]] = r
    conds = sorted({r["condition"] for r in recs})
    full = {q: d for q, d in tab.items() if len(d) == len(conds)}
    return full, conds


def means(tab, qids, conds, fields):
    out = {}
    for c in conds:
        acc = {f: 0.0 for f in fields}
        for q in qids:
            r = tab[q][c]
            for f in fields:
                acc[f] += float(r[f])
        out[c] = {f: acc[f] / max(len(qids), 1) for f in fields}
    return out


def interp(points, x):
    """Точность опоры при расходе x; вне отрезка возвращается None."""
    if len(points) < 2:
        return None
    xs = np.asarray([p[0] for p in points], float)
    ys = np.asarray([p[1] for p in points], float)
    o = np.argsort(xs)
    xs, ys = xs[o], ys[o]
    if x < xs[0] or x > xs[-1]:
        return None
    return float(np.interp(x, xs, ys))


def envelope(curves):
    """Верхняя огибающая нескольких кривых на объединении их точек расхода."""
    xs = sorted({x for c in curves for (x, _y) in c})
    out = []
    for x in xs:
        best = None
        for c in curves:
            v = interp(c, x)
            if v is not None and (best is None or v > best):
                best = v
        if best is not None:
            out.append((x, best))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--decisions", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    recs = [json.loads(l) for l in open(args.dump, encoding="utf-8") if l.strip()]
    decs = ([json.loads(l) for l in open(args.decisions, encoding="utf-8")
             if l.strip()] if os.path.exists(args.decisions) else [])
    tab, conds = per_question_table(recs)
    qids = sorted(tab)
    print("записей %d, вопросов с полным набором условий %d, условий %d"
          % (len(recs), len(qids), len(conds)), flush=True)

    fields = ["correct", "reasoning_tokens", "final_tokens", SPEND_KEY,
              "V_at_stop", "A_at_stop", "hanging", "wasted_in_cycle",
              "n_decisions", "valid", "n_commit"]
    for q in qids:
        for c in conds:
            r = tab[q][c]
            r["leftover_pos"] = max(float(r["budget"]) - float(r[SPEND_KEY]), 0.0)
            r["low_start"] = 1.0 if r.get("effort_start") == "low" else 0.0
            r["switched"] = 1.0 if (r.get("n_effort_switch") or 0) > 0 else 0.0
            r["diff_head_flipped_start"] = 1.0 if (
                r.get("effort_start") != r.get("effort_start_if_no_difficulty")) else 0.0
            for k in ("correct", "V_at_stop", "A_at_stop", "hanging", "valid"):
                r[k] = float(r[k])
    fields += ["leftover_pos", "low_start", "switched", "diff_head_flipped_start"]

    m = means(tab, qids, conds, fields)
    summary = {}
    for c in conds:
        kind, qv, B = cond_parts(c)
        summary[c] = {"kind": kind, "q": qv, "budget": B, "n": len(qids),
                      **{k: round(v, 4) for k, v in m[c].items()}}
        sp = np.asarray([float(tab[q][c][SPEND_KEY]) for q in qids])
        summary[c]["spend_median"] = round(float(np.median(sp)), 1)
        summary[c]["spend_p90"] = round(float(np.quantile(sp, 0.9)), 1)
        cr = defaultdict(int)
        for q in qids:
            cr[tab[q][c]["closed"]] += 1
        summary[c]["closed_kinds"] = dict(cr)
        summary[c]["natural_share"] = round(cr.get("natural", 0) / len(qids), 4)

    cap_conds = [c for c in conds if cond_parts(c)[0] == "cap"]
    capl_conds = [c for c in conds if cond_parts(c)[0] == "capl"]
    gate_conds = [c for c in conds if cond_parts(c)[0] not in CAP_KINDS]
    v21_conds = [c for c in conds if cond_parts(c)[0] == "v21"]
    v3_conds = [c for c in conds if cond_parts(c)[0] == "v3"]

    def curve(mm, cs, field="correct"):
        return sorted([(mm[c][SPEND_KEY], mm[c][field]) for c in cs])

    def refs_of(mm):
        cap = curve(mm, cap_conds)
        capl = curve(mm, capl_conds) if capl_conds else []
        front = envelope([cap, capl]) if capl else cap
        v21 = {}
        for qv in sorted({cond_parts(c)[1] for c in v21_conds}):
            v21[qv] = curve(mm, [c for c in v21_conds if cond_parts(c)[1] == qv])
        return {"cap": cap, "front": front, "v21": v21}

    base = refs_of(m)
    comp = {}
    for c in gate_conds:
        kind, qv, B = cond_parts(c)
        x = m[c][SPEND_KEY]
        a = m[c]["correct"]
        e = {"kind": kind, "q": qv, "budget": B, "spend": round(x, 1),
             "acc": round(a, 4)}
        for ref in ("cap", "front"):
            v = interp(base[ref], x)
            e["acc_%s_at_same_spend" % ref] = None if v is None else round(v, 4)
            e["acc_diff_%s" % ref] = None if v is None else round(a - v, 4)
        if kind == "v3" and qv in base["v21"]:
            v = interp(base["v21"][qv], x)
            e["acc_v21_at_same_spend"] = None if v is None else round(v, 4)
            e["acc_diff_v21"] = None if v is None else round(a - v, 4)
        vc = interp(curve(m, cap_conds, "V_at_stop"), x)
        e["verified"] = round(m[c]["V_at_stop"], 4)
        e["verified_cap_at_same_spend"] = None if vc is None else round(vc, 4)
        e["verified_diff"] = None if vc is None else round(m[c]["V_at_stop"] - vc, 4)
        e["hanging"] = round(m[c]["hanging"], 4)
        e["leftover"] = round(m[c]["leftover_pos"], 1)
        e["low_start_share"] = round(m[c]["low_start"], 4)
        e["switched_share"] = round(m[c]["switched"], 4)
        e["in_range_cap"] = e["acc_diff_cap"] is not None
        comp[c] = e

    # ── бутстрап по вопросам ─────────────────────────────────────────────
    rng = np.random.default_rng(args.seed)
    nq = len(qids)
    arr = {c: {f: np.asarray([float(tab[q][c][f]) for q in qids])
               for f in fields} for c in conds}
    keys = ["cap", "front", "v21"]
    draws = {c: {k: [] for k in keys + ["verified"]} for c in gate_conds}
    for _ in range(args.boot):
        pick = rng.integers(0, nq, nq)
        mm = {c: {f: float(arr[c][f][pick].mean()) for f in fields}
              for c in conds}
        rf = refs_of(mm)
        vcap = curve(mm, cap_conds, "V_at_stop")
        for c in gate_conds:
            kind, qv, _B = cond_parts(c)
            x = mm[c][SPEND_KEY]
            a = mm[c]["correct"]
            for ref in ("cap", "front"):
                v = interp(rf[ref], x)
                draws[c][ref].append(np.nan if v is None else a - v)
            v = (interp(rf["v21"][qv], x)
                 if (kind == "v3" and qv in rf["v21"]) else None)
            draws[c]["v21"].append(np.nan if v is None else a - v)
            vv = interp(vcap, x)
            draws[c]["verified"].append(
                np.nan if vv is None else mm[c]["V_at_stop"] - vv)
    for c in gate_conds:
        for k in keys + ["verified"]:
            v = np.asarray(draws[c][k], float)
            ok = np.isfinite(v)
            if ok.sum() < 50:
                comp[c]["ci_" + k] = None
                continue
            lo, hi = np.quantile(v[ok], [0.025, 0.975])
            comp[c]["ci_" + k] = [round(float(lo), 4), round(float(hi), 4)]

    # усреднение по бюджетам внутри одного плеча и одной осторожности
    by_q = {}
    for arm, qv in sorted({(cond_parts(c)[0], cond_parts(c)[1])
                           for c in gate_conds}):
        key = "%s|q%.2f" % (arm, qv)
        row = {"arm": arm, "q": qv}
        for k in keys + ["verified"]:
            sel = [c for c in gate_conds
                   if cond_parts(c)[0] == arm and cond_parts(c)[1] == qv
                   and comp[c].get(("acc_diff_%s" % k) if k in keys
                                   else "verified_diff") is not None]
            if not sel:
                row[k] = None
                continue
            fld = ("acc_diff_%s" % k) if k in keys else "verified_diff"
            pt = float(np.mean([comp[c][fld] for c in sel]))
            md = np.nanmean(np.column_stack(
                [np.asarray(draws[c][k], float) for c in sel]), axis=1)
            ok = np.isfinite(md)
            row[k] = {"n_points": len(sel), "mean": round(pt, 4),
                      "ci": ([round(float(np.quantile(md[ok], 0.025)), 4),
                              round(float(np.quantile(md[ok], 0.975)), 4)]
                             if ok.sum() > 50 else None),
                      "points": sel}
        row["spend_mean"] = round(float(np.mean(
            [comp[c]["spend"] for c in gate_conds
             if cond_parts(c)[0] == arm and cond_parts(c)[1] == qv])), 1)
        by_q[key] = row

    # ── разложение вклада голов по решениям ──────────────────────────────
    keep_q = set(qids)
    d2 = [d for d in decs if d["question_id"] in keep_q]
    dstat = {"n_all": len(decs), "n_in_table": len(d2)}
    for ver in ("v3", "v2.1"):
        dv = [d for d in d2 if d.get("policy_version") == ver]
        if not dv:
            continue
        n = max(len(dv), 1)
        rules = defaultdict(int)
        for d in dv:
            rules[d["rule"]] += 1
        blk = {"n_decisions": len(dv), "rules": dict(sorted(rules.items())),
               "share_cnext_evaluated": round(
                   sum(int(d["cnext_evaluated"]) for d in dv) / n, 4),
               "share_cnext_pivotal": round(
                   sum(int(d["cnext_pivotal"]) for d in dv) / n, 4),
               "share_cnext_binding": round(
                   sum(int(d["rule"] != d["rule_if_cnext_zero"]) for d in dv) / n, 4),
               "share_uncond_flip": round(
                   sum(int(d["cnext_uncond_flip"]) for d in dv) / n, 4)}
        if ver == "v3":
            blk.update({
                "share_switched_here": round(
                    sum(int(d.get("switched_here", False)) for d in dv) / n, 4),
                "share_tight": round(
                    sum(int(d.get("tight", False)) for d in dv) / n, 4),
                "share_switch_binding": round(
                    sum(int(d.get("switch_binding", False)) for d in dv) / n, 4),
                "share_next_type_binding": round(
                    sum(int(d.get("next_type_binding", False)) for d in dv) / n, 4),
                "share_effort_key_binding": round(
                    sum(int(d.get("effort_key_binding", False)) for d in dv) / n, 4),
                "share_differs_from_v21": round(
                    sum(int(d.get("differs_from_v21", False)) for d in dv) / n, 4),
                "next_argmax_share": {
                    k: round(v / n, 4) for k, v in sorted(
                        defaultdict(int, {a: sum(1 for d in dv
                                                 if d.get("next_argmax") == a)
                                          for a in {d.get("next_argmax")
                                                    for d in dv}}).items(),
                        key=lambda kv: -kv[1])},
                "p_branch_quantiles": {
                    str(p): round(float(np.quantile(
                        [d.get("p_branch", 0.0) for d in dv], p)), 4)
                    for p in (0.5, 0.85, 0.9)},
                "p_commit_quantiles": {
                    str(p): round(float(np.quantile(
                        [d.get("p_commit", 0.0) for d in dv], p)), 4)
                    for p in (0.25, 0.5, 0.75)},
                "wait_q_median": round(float(np.median(
                    [d.get("wait_q", 0.0) for d in dv])), 1),
                "share_decisions_at_low_effort": round(
                    sum(int(d.get("effort") == "low") for d in dv) / n, 4),
            })
            # трассы, где голова хоть раз связала решение
            tr = defaultdict(set)
            for d in dv:
                key = (d["question_id"], d["condition"])
                tr["any"].add(key)
                if d["rule"] != d["rule_if_cnext_zero"]:
                    tr["cnext"].add(key)
                if d.get("switch_binding"):
                    tr["switch"].add(key)
                if d.get("next_type_binding"):
                    tr["next_type"].add(key)
                if d.get("effort_key_binding"):
                    tr["effort_key"].add(key)
                if d.get("differs_from_v21"):
                    tr["differs_v21"].add(key)
            blk["traces_with_binding"] = {
                k: len(v) for k, v in sorted(tr.items())}
            blk["share_traces_with_binding"] = {
                k: round(len(v) / max(len(tr["any"]), 1), 4)
                for k, v in sorted(tr.items()) if k != "any"}
        dstat[ver] = blk

    # стартовый выбор усилия по плечам v3
    start = {}
    for c in v3_conds:
        rs = [tab[q][c] for q in qids]
        start[c] = {
            "low_start_share": round(np.mean([r["low_start"] for r in rs]), 4),
            "diff_head_flipped_share": round(
                np.mean([r["diff_head_flipped_start"] for r in rs]), 4),
            "switched_share": round(np.mean([r["switched"] for r in rs]), 4),
            "switch_tok_median": (round(float(np.median(
                [r["effort_switch_tok"] for r in rs
                 if r.get("effort_switch_tok") is not None])), 1)
                if any(r.get("effort_switch_tok") is not None for r in rs)
                else None)}
    dstat["start_effort_by_condition"] = start

    # ── инварианты ───────────────────────────────────────────────────────
    inv = {}
    bad = [[d["question_id"], d["condition"], d["rule"]] for d in decs
           if d["act"] == "finalize" and not d["A"] and d["rule"] != "4"]
    inv["policy_stops_before_candidate_not_rule4"] = len(bad)
    inv["examples"] = bad[:5]
    inv["records_without_final"] = sum(1 for r in recs if not r["has_final"])
    inv["records_without_answer"] = sum(1 for r in recs if not r["valid"])
    inv["cap_arm_decisions"] = sum(
        1 for q in qids for c in cap_conds + capl_conds
        if tab[q][c]["n_decisions"] > 0)
    inv["cap_arm_switches"] = sum(
        1 for q in qids for c in cap_conds + capl_conds
        if (tab[q][c].get("n_effort_switch") or 0) > 0)
    inv["v21_switches"] = sum(1 for q in qids for c in v21_conds
                              if (tab[q][c].get("n_effort_switch") or 0) > 0)
    inv["v21_low_starts"] = sum(1 for q in qids for c in v21_conds
                                if tab[q][c].get("effort_start") == "low")

    # ── вердикт ──────────────────────────────────────────────────────────
    verdict = {"detail": {}}
    for name, cond in (("A", None), ("B", None), ("C", None), ("D", None)):
        verdict[name] = False
    for k, v in by_q.items():
        if not k.startswith("v3"):
            continue
        cap = v.get("cap")
        if cap and cap["ci"]:
            ok = (cap["mean"] >= -0.02 and cap["ci"][0] >= -0.05
                  and (v.get("verified") or {}).get("mean", -1) >= 0.05)
            verdict["detail"]["A_" + k] = {
                "acc_diff_cap_mean": cap["mean"], "ci": cap["ci"],
                "verified_diff_mean": (v.get("verified") or {}).get("mean"),
                "passed": bool(ok)}
            verdict["A"] = verdict["A"] or bool(ok)
        v21 = v.get("v21")
        if v21 and v21["ci"]:
            ok = v21["mean"] > 0 and v21["ci"][0] > 0
            verdict["detail"]["C_" + k] = {"acc_diff_v21_mean": v21["mean"],
                                           "ci": v21["ci"], "passed": bool(ok)}
            verdict["C"] = verdict["C"] or bool(ok)
        fr = v.get("front")
        if fr:
            ok = fr["mean"] >= -0.02
            verdict["detail"]["D_" + k] = {"acc_diff_front_mean": fr["mean"],
                                           "ci": fr["ci"], "passed": bool(ok)}
            verdict["D"] = verdict["D"] or bool(ok)
    for c in v3_conds:
        e = comp[c]
        if e.get("acc_diff_cap") is None or e.get("ci_cap") is None:
            continue
        ok = e["acc_diff_cap"] > 0 and e["ci_cap"][0] > 0
        verdict["detail"]["B_" + c] = {"acc_diff_cap": e["acc_diff_cap"],
                                       "ci": e["ci_cap"], "passed": bool(ok)}
        verdict["B"] = verdict["B"] or bool(ok)
    verdict["vs_fixed_cap"] = bool(verdict["A"] or verdict["B"])
    verdict["vs_v21"] = bool(verdict["C"])
    verdict["vs_config_front"] = bool(verdict["D"])

    out = {"dump": os.path.abspath(args.dump),
           "decisions": os.path.abspath(args.decisions),
           "n_records": len(recs), "n_questions_full": len(qids),
           "conditions": conds, "summary": summary,
           "reference_curves": {
               "cap": [[round(a, 1), round(b, 4)] for a, b in base["cap"]],
               "front": [[round(a, 1), round(b, 4)] for a, b in base["front"]],
               "v21": {str(k): [[round(a, 1), round(b, 4)] for a, b in v]
                       for k, v in base["v21"].items()}},
           "comparison": comp, "by_q": by_q, "decision_stats": dstat,
           "invariants": inv, "verdict": verdict, "n_boot": args.boot}
    json.dump(out, open(args.out, "w"), ensure_ascii=False, indent=1)

    print("\nопора cap: %s" % out["reference_curves"]["cap"])
    print("опора front: %s" % out["reference_curves"]["front"])
    print("\n%-16s %8s %7s %9s %9s %9s %7s %7s"
          % ("условие", "расход", "точн", "разн_cap", "разн_фронт", "разн_v21",
             "низкий", "смен"))
    for c in conds:
        s = summary[c]
        e = comp.get(c, {})
        def f(k):
            v = e.get(k)
            return ("%+.3f" % v) if v is not None else "-"
        print("%-16s %8.0f %7.3f %9s %9s %9s %7.2f %7.2f"
              % (c, s[SPEND_KEY], s["correct"], f("acc_diff_cap"),
                 f("acc_diff_front"), f("acc_diff_v21"),
                 s.get("low_start", 0.0), s.get("switched", 0.0)))
    print("\nсводка по осторожности:")
    for k, v in by_q.items():
        print("  %-12s cap %s  фронт %s  v21 %s"
              % (k, (v.get("cap") or {}).get("mean"),
                 (v.get("front") or {}).get("mean"),
                 (v.get("v21") or {}).get("mean")))
    print("\nвклад голов v3: %s" % json.dumps(
        {k: v for k, v in (dstat.get("v3") or {}).items()
         if k.startswith("share_") or k == "rules"}, ensure_ascii=False))
    print("\nвердикт: %s" % json.dumps(
        {k: v for k, v in verdict.items() if k != "detail"}, ensure_ascii=False))
    print("записано %s" % args.out)


if __name__ == "__main__":
    main()
