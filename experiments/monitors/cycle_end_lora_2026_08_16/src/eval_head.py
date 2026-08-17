"""Оценка прогнозиста конца цикла: один прямой проход на окно.

Реализует раздел 10 предрегистрации эксперимента cycle_end_lora_2026_08_16: адаптер
накладывается библиотекой peft без слияния весов, читаются логиты последней позиции
запроса, из них берутся только позиции одиннадцати букв-меток, по ним считается
softmax. Из распределения по корзинам получаются точечная оценка остатка, границы
интервала и вероятность конца цикла в горизонте.

Запуск на боксе tesla, две карты A6000.
"""

import argparse
import json
import math
import os
import time

import torch

BUCKET_EDGES = [0.0, 16.0, 32.0, 64.0, 128.0, 256.0, 512.0, 1024.0, 2048.0, 4096.0,
                8192.0, 16384.0]
BUCKET_LABELS = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K"]
H_BUCKET = 5  # корзины A..E целиком лежат ниже горизонта 256


def quantile_from_bins(p, q):
    """Квантиль кусочно-логравномерного распределения по корзинам."""
    acc = 0.0
    for i, pi in enumerate(p):
        if acc + pi >= q or i == len(p) - 1:
            lo = max(BUCKET_EDGES[i], 1.0)
            hi = max(BUCKET_EDGES[i + 1], lo + 1.0)
            frac = (q - acc) / pi if pi > 0 else 0.5
            frac = min(max(frac, 0.0), 1.0)
            return float(math.exp(math.log(lo) + frac * (math.log(hi) - math.log(lo))))
        acc += pi
    return float(BUCKET_EDGES[-1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-memory", default='{"0": "36GiB", "1": "40GiB"}')
    ap.add_argument("--label-ids", default=None,
                    help="файл outputs/label_tokens.json")
    ap.add_argument("--log-every", type=int, default=200)
    a = ap.parse_args()

    from transformers import AutoModelForCausalLM

    rows = [json.loads(l) for l in open(a.data)]
    if a.limit:
        rows = rows[:a.limit]

    if a.label_ids:
        lt = json.load(open(a.label_ids))
        ids = [lt["ids"][c][0] for c in BUCKET_LABELS]
    else:
        ids = list(range(32, 43))

    mm = {int(k): v for k, v in json.loads(a.max_memory).items()}
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        a.model, dtype=torch.bfloat16, device_map="balanced", max_memory=mm,
        trust_remote_code=True, attn_implementation="sdpa")
    if a.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, a.adapter)
    model.eval()
    t_load = time.time() - t0
    print("модель загружена за %.1f с" % t_load, flush=True)

    lab_idx = torch.tensor(ids, dtype=torch.long)
    in_dev = next(model.parameters()).device
    out = []
    t1 = time.time()
    with torch.no_grad():
        for n, r in enumerate(rows):
            x = torch.tensor([r["prompt_ids"]], dtype=torch.long, device=in_dev)
            try:
                o = model(input_ids=x, logits_to_keep=1)
            except TypeError:
                o = model(input_ids=x)
            lg = o.logits[0, -1, :].float().cpu()
            sel = lg[lab_idx]
            p = torch.softmax(sel, dim=-1).tolist()
            med = quantile_from_bins(p, 0.5)
            lo = quantile_from_bins(p, 0.1)
            hi = quantile_from_bins(p, 0.9)
            out.append({"uid": r["uid"], "trace_uid": r["trace_uid"],
                        "rem": r["rem"], "bin_H": r["bin_H"], "bucket": r["bucket"],
                        "model": r.get("model"), "effort": r.get("effort"),
                        "layer": r.get("layer"), "cut_src": r.get("cut_src"),
                        "p": [round(v, 5) for v in p],
                        "med": med, "lo": lo, "hi": hi,
                        "p_H": float(sum(p[:H_BUCKET])),
                        "argmax": BUCKET_LABELS[int(max(range(11), key=lambda i: p[i]))]})
            if (n + 1) % a.log_every == 0:
                el = time.time() - t1
                print("окон %d из %d, %.2f с/окно, осталось %.1f мин"
                      % (n + 1, len(rows), el / (n + 1),
                         (len(rows) - n - 1) * el / (n + 1) / 60.0), flush=True)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    meta = {"n": len(out), "load_s": t_load, "gen_s": time.time() - t1,
            "s_per_window": (time.time() - t1) / max(len(out), 1),
            "adapter": a.adapter, "data": a.data}
    with open(a.out + ".meta.json", "w") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)
    print(json.dumps(meta, ensure_ascii=False))


if __name__ == "__main__":
    main()
