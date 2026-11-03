#!/usr/bin/env python3
"""calibrate v3 for the gpt-oss effort controller."""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter

import numpy as np

TYPES = ["backtrack", "verify", "branch", "subgoal_done", "failed_attempt",
         "commit"]
T_BR, T_COMMIT = 2, 5
K = 6
LIVE_MODEL = "openai/gpt-oss-20b"

Q_BRANCH = 0.85     # порог ветвления: верхние 15 процентов доли ветвления
Q_COMMIT = 0.50     # порог консолидации: медиана доли консолидации


def wait_quantile(mu_tot, G, beta, q):
    z = -math.log(max(1.0 - q, 1e-12))
    hi = z / max(mu_tot, 1e-9)
    lo = 0.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        comp = mu_tot * mid + (G / beta) * (1.0 - math.exp(-beta * mid))
        if comp < z:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--streams", required=True)
    ap.add_argument("--packs", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    pack = json.load(open("%s/hawkes_pack.json" % args.packs, encoding="utf-8"))
    p = pack.get(LIVE_MODEL) or pack["__pooled__"]
    mu = np.asarray(p["mu"], float)
    A = np.asarray(p["A"], float)
    beta = float(p["beta"])
    rowsum = A.sum(axis=1)
    mu_tot = float(mu.sum())

    pb, pc, w3, w5 = [], [], [], []
    argmax_cnt = Counter()
    for line in open(args.streams, encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        if r["model"] != LIVE_MODEL or r["split"] not in ("train", "val"):
            continue
        evs = sorted((float(e[0]), int(e[1])) for e in r["events"])
        cand = r.get("cand_tok")
        bounds = [s for (s, k) in evs if k == T_COMMIT]
        S = np.zeros(K)
        t = 0.0
        ei = 0
        for bnd in bounds:
            while ei < len(evs) and evs[ei][0] <= bnd:
                s, k = evs[ei]
                S *= math.exp(-beta * (s - t))
                t = s
                S[k] += beta
                ei += 1
            if cand is None or cand > bnd:
                continue
            S2 = S * math.exp(-beta * (bnd - t))
            lam = mu + A.T.dot(S2)
            pr = lam / max(lam.sum(), 1e-12)
            argmax_cnt[TYPES[int(np.argmax(pr))]] += 1
            pb.append(float(pr[T_BR]))
            pc.append(float(pr[T_COMMIT]))
            G = float(rowsum.dot(S2))
            w3.append(wait_quantile(mu_tot, G, beta, 0.3))
            w5.append(wait_quantile(mu_tot, G, beta, 0.5))

    pb = np.asarray(pb)
    pc = np.asarray(pc)
    qs = [0.1, 0.25, 0.5, 0.75, 0.85, 0.9, 0.95]
    out = {
        "model": LIVE_MODEL,
        "n_points": int(len(pb)),
        "points_meaning": ("границы циклов после появления кандидата у трасс "
                           "живой модели, обучающая и проверочная части корпуса "
                           "задания 12"),
        "argmax_share": {k: round(v / max(len(pb), 1), 4)
                         for k, v in argmax_cnt.most_common()},
        "p_branch_quantiles": {str(q): round(float(np.quantile(pb, q)), 4)
                               for q in qs},
        "p_commit_quantiles": {str(q): round(float(np.quantile(pc, q)), 4)
                               for q in qs},
        "wait_q30_quantiles": {str(q): round(float(np.quantile(w3, q)), 1)
                               for q in qs},
        "wait_q50_quantiles": {str(q): round(float(np.quantile(w5, q)), 1)
                               for q in qs},
        "rule": ("порог ветвления = квантиль %.2f корпусного распределения доли "
                 "ветвления; порог консолидации = квантиль %.2f доли "
                 "консолидации" % (Q_BRANCH, Q_COMMIT)),
        "thr_branch": round(float(np.quantile(pb, Q_BRANCH)), 4),
        "thr_commit": round(float(np.quantile(pc, Q_COMMIT)), 4),
        "share_points_above_thr_branch": round(float(np.mean(
            pb >= np.quantile(pb, Q_BRANCH))), 4),
        "share_points_above_thr_commit": round(float(np.mean(
            pc >= np.quantile(pc, Q_COMMIT))), 4),
    }
    json.dump(out, open(args.out, "w"), ensure_ascii=False, indent=1)
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
