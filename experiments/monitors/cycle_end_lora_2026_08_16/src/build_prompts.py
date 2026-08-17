"""Сборка текстов запросов для двух плеч прогнозиста: с инъекцией и без.

Реализует разделы 6 и 7 предрегистрации эксперимента cycle_end_lora_2026_08_16.
Плечи отличаются РОВНО одним блоком текста: хоуковским. Всё остальное совпадает
посимвольно, включая порядок окон.

Чистый python, никаких зависимостей. Запуск на боксе 192.168.2.48.
"""

import argparse
import json
import os

M_TYPES = ["backtrack", "verify", "branch", "subgoal_done", "failed_attempt", "commit"]
BUCKET_EDGES = [0, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
BUCKET_LABELS = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K"]
N_CTX_SEG = 8
SEG_CHARS = 320
Q_CHARS = 300
N_EV_HIST = 12

LEGEND = ", ".join(
    "%s=%d..%s" % (BUCKET_LABELS[i], BUCKET_EDGES[i],
                   str(BUCKET_EDGES[i + 1]) if i + 1 < len(BUCKET_EDGES) else "inf")
    for i in range(len(BUCKET_LABELS)))

SYSTEM = (
    "You forecast the end of the current reasoning cycle.\n"
    "A reasoning cycle is a burst of self-exciting reasoning events that is closed by a "
    "consolidation event of type commit, where the model fixes an intermediate or final "
    "result. You are given a prefix of a reasoning trace produced by some model, together "
    "with the reasoning events that a labeler already marked on that prefix.\n"
    "Predict how many more tokens the model will emit before the next commit event. "
    "Answer with exactly one capital letter naming the bucket that contains that number "
    "of tokens:\n" + LEGEND + "\n"
    "Event types: " + ", ".join(M_TYPES) + ".\n"
    "Output the single letter and nothing else."
)


def hawkes_block(w):
    A = w["A"]
    mu = w["mu1k"]
    lam = w["feat_hawkes"][:6]
    rho = w["feat_hawkes"][6]
    base = w["feat_base"]
    since = base[26:32]
    lines = ["[hawkes structure of this model and effort group%s]" %
             ("" if w.get("cut_src") == "group" else ", pooled fallback")]
    lines.append("excitation matrix A, rows = source type, columns = child type, "
                 "order " + " ".join(M_TYPES))
    for i, t in enumerate(M_TYPES):
        lines.append("  %-15s %s" % (t, " ".join("%.2f" % float(x) for x in A[i])))
    lines.append("branching number rho %.3f, sum A %.2f" %
                 (rho, sum(sum(r) for r in A)))
    lines.append("background mu per 1000 tokens: " +
                 " ".join("%s %.2f" % (t, float(m)) for t, m in zip(M_TYPES, mu)))
    lines.append("current intensity per 1000 tokens: " +
                 " ".join("%s %.2f" % (t, float(x) * 1000.0) for t, x in zip(M_TYPES, lam)))
    lines.append("tokens since last event of each type: " +
                 " ".join("%s %s" % (t, ("%.0f" % s) if s >= 0 else "never")
                          for t, s in zip(M_TYPES, since)))
    return "\n".join(lines)


def user_msg(w, inject):
    parts = []
    parts.append("[conditions]\nmodel %s\neffort %s\nlength cap %s\ntokens produced so far %.0f"
                 % (w.get("model"), w.get("effort") or "unspecified",
                    w.get("cap") if w.get("cap") else "unknown", w["t_tok"]))
    q = (w.get("question") or "")[:Q_CHARS].replace("\n", " ")
    parts.append("[question]\n" + q)
    ctx = w.get("ctx_segments") or []
    ctx = ctx[-N_CTX_SEG:]
    first = w["i_seg"] - len(ctx) + 1
    seg_lines = []
    for k, s in enumerate(ctx):
        seg_lines.append("(%d) %s" % (first + k, s[:SEG_CHARS].replace("\n", " ")))
    parts.append("[last reasoning segments, the last one ends exactly at the current "
                 "position]\n" + "\n".join(seg_lines))
    hist = (w.get("ev_hist") or [])[-N_EV_HIST:]
    if hist:
        ev_lines = ["%.0f tokens ago  %s" % (float(d), t) for d, t in hist]
    else:
        ev_lines = ["no events yet"]
    parts.append("[events on the prefix, most recent last]\n" + "\n".join(ev_lines))
    if inject:
        parts.append(hawkes_block(w))
    parts.append("[task]\nHow many more tokens until the next commit? "
                 "Answer with one letter.")
    return "\n\n".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    stats = {}
    for split in ("train", "val", "test"):
        src = os.path.join(a.ds, "windows_%s.jsonl" % split)
        fh = {arm: open(os.path.join(a.out, "prompts_%s_%s.jsonl" % (arm, split)), "w")
              for arm in ("noinj", "inj")}
        n = 0
        clen = {"noinj": 0, "inj": 0}
        with open(src) as f:
            for line in f:
                w = json.loads(line)
                for arm, inj in (("noinj", False), ("inj", True)):
                    u = user_msg(w, inj)
                    rec = {"uid": w["uid"], "trace_uid": w["trace_uid"], "split": split,
                           "arm": arm, "system": SYSTEM, "user": u,
                           "target": w["label"], "rem": w["rem"], "bin_H": w["bin_H"],
                           "bucket": w["bucket"], "model": w["model"],
                           "effort": w["effort"], "layer": w["layer"],
                           "cut": w["cut"], "cut_src": w["cut_src"]}
                    fh[arm].write(json.dumps(rec, ensure_ascii=False) + "\n")
                    clen[arm] += len(u)
                n += 1
        for f_ in fh.values():
            f_.close()
        stats[split] = {"n": n,
                        "mean_user_chars_noinj": clen["noinj"] / max(n, 1),
                        "mean_user_chars_inj": clen["inj"] / max(n, 1)}
        print(split, json.dumps(stats[split], ensure_ascii=False))
    stats["system_chars"] = len(SYSTEM)
    stats["legend"] = LEGEND
    with open(os.path.join(a.out, "prompt_stats.json"), "w") as f:
        json.dump(stats, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
