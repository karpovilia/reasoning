"""Выгрузка окон отложенной части в формате эксперимента cycle_end_lora для справочного бара Б3.

Принятый прогнозист остатка текущего цикла ждёт окно с текстом последних сегментов,
поэтому тексты приходится восстановить тем же проходом по корпусам, что и в сборке
датасета. Отбираются только окна отложенной части первичного корпуса; те же окна потом
сравниваются с прогнозом лёгкой головы точка в точку.

Запуск на боксе tesla:
  /home/user/ki_sft_judge/.venv/bin/python -u src/export_lora_windows.py \
      --root /home/user/ki_cycles --ds data/ds --out data/lora_windows --n 1000
"""

import argparse
import json
import os
import random

import build_dataset as B

BUCKET_EDGES = [0.0, 16.0, 32.0, 64.0, 128.0, 256.0, 512.0, 1024.0, 2048.0, 4096.0, 8192.0]
BUCKET_LABELS = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K"]
N_CTX_SEG = 8
SEG_CHARS = 320
N_EV_HIST = 12
H = 256.0


def bucket_of(rem):
    for i in range(len(BUCKET_EDGES) - 1, -1, -1):
        if rem >= BUCKET_EDGES[i]:
            return i
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--ds", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--per-trace", type=int, default=3)
    ap.add_argument("--seed", type=int, default=20260817)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    recs = {}
    with open(os.path.join(a.ds, "traces_test.jsonl"), encoding="utf-8") as fh:
        for line in fh:
            d = json.loads(line)
            if d["wave"] == "wave1":
                recs[d["trace_uid"]] = d
    rng = random.Random(a.seed)
    uids = sorted(recs)
    rng.shuffle(uids)
    want = {}
    total = 0
    for u in uids:
        d = recs[u]
        cand = list(range(1, d["n_seg"] - 1))
        if not cand:
            continue
        rng.shuffle(cand)
        take = sorted(cand[:a.per_trace])
        want[u] = take
        total += len(take)
        if total >= a.n:
            break
    print("отобрано трасс %d, окон %d" % (len(want), total), flush=True)

    lab_dir = os.path.join(a.root, "data", "labels")
    tr_root = os.path.join(a.root, "data", "traces")
    entries = json.load(open(os.path.join(lab_dir, "corpora_wave1.json"), encoding="utf-8"))
    out_path = os.path.join(a.out, "windows_test.jsonl")
    n_out = 0
    with open(out_path, "w", encoding="utf-8") as fo:
        for e in entries:
            path = B.local_path(e["path"], tr_root)
            tag, kind = e["tag"], e.get("kind", "stream")
            if not os.path.exists(path):
                continue
            for line in open(path, encoding="utf-8"):
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                txt = B.stream_text(r) if kind == "stream" else B.toolloop_text(r)
                if len(txt.strip()) < B.MIN_CHARS:
                    continue
                segs = B.segment(txt)
                n_seg = len(segs)
                if n_seg < B.MIN_SEGMENTS:
                    continue
                rule = e.get("arm_rule")
                if rule == "level_ersatz":
                    arm_name = "level_%s_ersatz%s" % (r.get("effort_level"), r.get("cap"))
                else:
                    arm_name = e.get("arm") or r.get("arm") or tag
                import hashlib
                thash = hashlib.sha1(("\n".join(segs)).encode("utf-8")).hexdigest()[:8]
                uid_arm = "%s|" % arm_name if rule else ""
                trace_uid = "%s|%s%s|%s" % (tag, uid_arm, r["question_id"], thash)
                if trace_uid not in want:
                    continue
                d = recs[trace_uid]
                ends = d["seg_end"]
                commits = sorted(ends[s] for s, k in d["events"] if k == 5)
                events = sorted(d["events"])
                for i_seg in want[trace_uid]:
                    t = ends[i_seg]
                    nxt = [c for c in commits if c > t]
                    rem = (nxt[0] if nxt else d["T"]) - t
                    b = bucket_of(rem)
                    lo = max(0, i_seg - N_CTX_SEG + 1)
                    ctx = [segs[k][:SEG_CHARS] for k in range(lo, i_seg + 1)]
                    hist = [[round(t - ends[s], 1), B.M_TYPES[k]]
                            for s, k in events if ends[s] <= t][-N_EV_HIST:]
                    fo.write(json.dumps({
                        "uid": "%s::%d" % (trace_uid, i_seg),
                        "trace_uid": trace_uid, "split": "test",
                        "layer": d["corpus"], "model": d["model"],
                        "effort": d["effort"], "cap": d["cap"],
                        "i_seg": i_seg, "t_tok": t, "rem": rem,
                        "bin_H": int(rem < H), "bucket": b, "label": BUCKET_LABELS[b],
                        "question": (r.get("question") or "")[:400],
                        "ctx_segments": ctx, "ev_hist": hist,
                        "cut": "n/a", "cut_src": "n/a",
                    }, ensure_ascii=False) + "\n")
                    n_out += 1
    print("записано окон %d в %s" % (n_out, out_path))


if __name__ == "__main__":
    main()
