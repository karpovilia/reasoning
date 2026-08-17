#!/usr/bin/env python3
"""Проверка, что предел длины хода действительно дошёл до движка.

По дампам трасс считает длину каждого хода модели в токенах (поле
usage.completion_tokens сохранённого ответа) и причину завершения хода (поле
choices[0].finish_reason). Если предел работает, ни один ход не длиннее предела,
а упершиеся в него ходы имеют причину length.

Читать строки только через split по переводу строки: раннер пишет без
экранирования разделителей U+2028/U+2029/U+0085, и splitlines рвёт целую строку
на куски.

Запуск:
    python3 check_cap.py <предел> <файл.jsonl> [ещё файлы...]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def read_rows(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").split("\n"):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            print(f"  битая строка в {path}")
    return rows


def main(argv: list[str]) -> int:
    if len(argv) < 3:
        raise SystemExit(__doc__)
    cap = int(argv[1])
    for raw in argv[2:]:
        path = Path(raw)
        lengths: list[int] = []
        reasons: dict[str, int] = {}
        traces = 0
        answered = 0
        for row in read_rows(path):
            work = row.get("raw_model_work") or {}
            traces += 1
            answered += bool(work.get("completed"))
            for response in work.get("responses") or []:
                usage = response.get("usage") or {}
                if usage.get("completion_tokens") is not None:
                    lengths.append(int(usage["completion_tokens"]))
                for choice in response.get("choices") or []:
                    reason = str(choice.get("finish_reason"))
                    reasons[reason] = reasons.get(reason, 0) + 1
        print(f"\n== {path}")
        print(f"  трасс {traces}, с ответом {answered}, ходов {len(lengths)}")
        if lengths:
            print(f"  длина хода: макс {max(lengths)}, медиана "
                  f"{sorted(lengths)[len(lengths) // 2]}, предел {cap}")
            over = [x for x in lengths if x > cap]
            print(f"  ходов длиннее предела: {len(over)}"
                  + (f", значения {sorted(over)[-5:]}" if over else ""))
            print(f"  ходов ровно на пределе: {sum(x == cap for x in lengths)}")
        print(f"  причины завершения хода: {dict(sorted(reasons.items(), key=lambda kv: -kv[1]))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
