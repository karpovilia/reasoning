"""Токенизация пар запрос-ответ для прогнозиста конца цикла.

Читает файлы src/build_prompts.py и пишет записи вида {prompt_ids, target_ids},
которые понимает готовый обучающий цикл
internal_signals_poc/lora_judge_2026_08_15/src/train_lora.py

Запускается на боксе tesla в обучающем окружении, только процессор.
"""

import argparse
import json
import os

from transformers import AutoTokenizer

BUCKET_LABELS = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-len", type=int, default=4096)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--label-check", default=None)
    a = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)

    lab_ids = {}
    bad = []
    for ch in BUCKET_LABELS:
        ids = tok.encode(ch, add_special_tokens=False)
        lab_ids[ch] = ids
        if len(ids) != 1:
            bad.append(ch)
    if a.label_check:
        with open(a.label_check, "w") as f:
            json.dump({"labels": BUCKET_LABELS,
                       "ids": {k: v for k, v in lab_ids.items()},
                       "multitoken": bad,
                       "eos": tok.eos_token, "eos_id": tok.eos_token_id,
                       "tokenizer": type(tok).__name__, "vocab": len(tok)},
                      f, ensure_ascii=False, indent=1)
    if bad:
        raise SystemExit("многотокенные метки: %s" % bad)

    eos = tok.eos_token_id
    n, drop, tot_p = 0, 0, 0
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.inp) as fi, open(a.out, "w") as fo:
        for line in fi:
            r = json.loads(line)
            msgs = [{"role": "system", "content": r["system"]},
                    {"role": "user", "content": r["user"]}]
            pid = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                          enable_thinking=False, tokenize=True)
            if not isinstance(pid, list):
                pid = pid["input_ids"]
            if pid and isinstance(pid[0], list):
                pid = pid[0]
            pid = [int(x) for x in pid]
            tid = lab_ids[r["target"]] + ([eos] if eos is not None else [])
            if len(pid) + len(tid) > a.max_len:
                drop += 1
                continue
            fo.write(json.dumps({"uid": r["uid"], "trace_uid": r["trace_uid"],
                                 "arm": r["arm"], "split": r["split"],
                                 "target": r["target"], "rem": r["rem"],
                                 "bin_H": r["bin_H"], "bucket": r["bucket"],
                                 "model": r["model"], "effort": r["effort"],
                                 "layer": r["layer"], "cut_src": r["cut_src"],
                                 "prompt_ids": pid, "target_ids": tid},
                                ensure_ascii=False) + "\n")
            tot_p += len(pid)
            n += 1
            if a.limit and n >= a.limit:
                break
    print(json.dumps({"out": a.out, "n": n, "dropped_too_long": drop,
                      "mean_prompt_tokens": tot_p / max(n, 1)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
