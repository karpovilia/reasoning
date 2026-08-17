#!/usr/bin/env python3
"""Вливание добора окном 49152 в канон перепрогона 16 августа (задание 07).

Зачем отдельный скрипт, а не dedupe_traces.py. Тот схлопывает дубли по правилу
«первая строка побеждает» и запрещает отбор по успешности намеренно: иначе доля
дошедших до ответа завысилась бы, ведь у пересчитанных вопросов попыток было бы
больше. Здесь правило другое и оно принято отдельным решением:
новая строка побеждает ТОЛЬКО у тех вопросов, у которых прежняя строка не дошла
до ответа. Это не отбор лучшего из двух исходов одной и той же установки: прежние
строки — отказ движка на переполнении окна контекста, то есть сбой оснастки, а не
неудачная попытка модели. У вопросов, где прежняя строка с ответом, новая строка
игнорируется, даже если она есть.

Порядок строк канонического файла сохраняется: замена идёт на месте.

Читать строки только через split по переводу строки: раннер пишет без
экранирования разделителей U+2028/U+2029/U+0085, и splitlines рвёт целую строку
на куски.

Запуск:
    python3 merge_win49152.py <канонический.jsonl> <перезапуск.jsonl>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def read_rows(path: Path) -> list[tuple[str, dict]]:
    rows = []
    for line in path.read_text(encoding="utf-8").split("\n"):
        if not line.strip():
            continue
        rows.append((line, json.loads(line)))
    return rows


def answered(row: dict) -> bool:
    return bool((row.get("raw_model_work") or {}).get("completed"))


def merge(canon_path: Path, retry_path: Path) -> dict:
    canon = read_rows(canon_path)
    retry = {str(r["question_id"]): (line, r) for line, r in read_rows(retry_path)}

    replaced, still_failing, skipped_had_answer, unused = [], [], [], []
    out_lines = []
    seen = set()
    for line, row in canon:
        qid = str(row["question_id"])
        seen.add(qid)
        if qid in retry and not answered(row):
            new_line, new_row = retry[qid]
            out_lines.append(new_line)
            if answered(new_row):
                replaced.append(qid)
            else:
                still_failing.append(
                    {
                        "question_id": qid,
                        "failure_reason": (new_row.get("raw_model_work") or {}).get("failure_reason"),
                        "record_wall_s": (new_row.get("raw_model_work") or {}).get("record_wall_s"),
                    }
                )
        else:
            out_lines.append(line)
            if qid in retry:
                skipped_had_answer.append(qid)
    unused = sorted(set(retry) - seen)

    backup = canon_path.with_suffix(canon_path.suffix + ".before_win49152")
    if not backup.exists():
        backup.write_text(canon_path.read_text(encoding="utf-8"), encoding="utf-8")
    canon_path.write_text("\n".join(out_lines) + "\n", encoding="utf-8")

    return {
        "canon": str(canon_path),
        "retry": str(retry_path),
        "canon_rows": len(canon),
        "retry_rows": len(retry),
        "replaced_now_answered": sorted(replaced),
        "replaced_still_failing": still_failing,
        "skipped_because_already_answered": sorted(skipped_had_answer),
        "retry_ids_absent_from_canon": unused,
        "backup": str(backup),
    }


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        raise SystemExit(__doc__)
    print(json.dumps(merge(Path(argv[1]), Path(argv[2])), ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
