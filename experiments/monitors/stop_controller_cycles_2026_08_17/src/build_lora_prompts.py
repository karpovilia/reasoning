"""Запросы для справочного бара Б3 в точности тем же кодом, что у принятого прогнозиста.

Берётся модуль build_prompts эксперимента cycle_end_lora_2026_08_16 и его функции SYSTEM и
user_msg. Пишется только плечо без инъекции: именно его адаптер принят.

Запуск на боксе tesla:
  /home/user/ki_sft_judge/.venv/bin/python -u src/build_lora_prompts.py \
      --src /home/user/ki_cycle_end/src --inp data/lora_windows/windows_test.jsonl \
      --out data/lora_windows/prompts_noinj_test.jsonl
"""

import argparse
import json
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--inp", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    sys.path.insert(0, a.src)
    import build_prompts as BP

    n, chars = 0, 0
    with open(a.inp, encoding="utf-8") as fi, open(a.out, "w", encoding="utf-8") as fo:
        for line in fi:
            w = json.loads(line)
            u = BP.user_msg(w, False)
            fo.write(json.dumps({"uid": w["uid"], "trace_uid": w["trace_uid"],
                                 "split": "test", "arm": "noinj",
                                 "system": BP.SYSTEM, "user": u,
                                 "target": w["label"], "rem": w["rem"],
                                 "bin_H": w["bin_H"], "bucket": w["bucket"],
                                 "model": w["model"], "effort": w["effort"],
                                 "layer": w["layer"], "cut_src": w["cut_src"]},
                                ensure_ascii=False) + "\n")
            n += 1
            chars += len(u)
    print(json.dumps({"окон": n, "средняя длина запроса в знаках": chars / max(n, 1)},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
