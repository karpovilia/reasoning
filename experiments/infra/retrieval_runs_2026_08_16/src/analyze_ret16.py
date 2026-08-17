#!/usr/bin/env python3
"""Разбор перепрогона 16 августа: исполнение предела хода и род новых отказов.

Что считает по каждому корпусу: длину ходов и долю ходов длиннее 7000, занятую
историю диалога, состав отказов. Отдельно сверяет, совпадают ли вопросы, которые
в перепрогоне отвергнуты движком по переполнению окна, с теми вопросами, у
которых в дефектном каноне 15 августа был разогнавшийся ход длиннее 7000 токенов.

Читать строки только через split по переводу строки: раннер пишет без
экранирования разделителей U+2028/U+2029/U+0085.

Запуск: python3 analyze_ret16.py <корень нового> <корень старого канона> <выход.json>
"""
from __future__ import annotations

import json
import statistics as st
import sys
from pathlib import Path

CAP = 7000


def read_rows(path: Path):
    for line in path.read_text(encoding="utf-8").split("\n"):
        if line.strip():
            try:
                yield json.loads(line)
            except ValueError:
                continue


def q(values, p):
    if not values:
        return 0
    values = sorted(values)
    return values[min(len(values) - 1, int(p * len(values)))]


def describe(path: Path) -> dict:
    turn_lengths = []
    reasons = {}
    failures = {}
    max_context = []
    failed_ids = []
    runaway_ids = []
    traces = 0
    answered = 0
    for row in read_rows(path):
        work = row.get("raw_model_work") or {}
        traces += 1
        qid = str(row.get("question_id"))
        ok = bool(work.get("completed"))
        answered += int(ok)
        if not ok:
            reason = str(work.get("failure_reason"))
            failures[reason] = failures.get(reason, 0) + 1
            failed_ids.append([qid, reason])
        usages = []
        for response in work.get("responses") or []:
            usage = response.get("usage") or {}
            if usage.get("completion_tokens") is not None:
                turn_lengths.append(int(usage["completion_tokens"]))
            if usage.get("prompt_tokens") is not None and usage.get("completion_tokens") is not None:
                usages.append((int(usage["prompt_tokens"]), int(usage["completion_tokens"])))
            for choice in response.get("choices") or []:
                key = str(choice.get("finish_reason"))
                reasons[key] = reasons.get(key, 0) + 1
        if usages:
            max_context.append(max(p + c for p, c in usages))
            if any(c > CAP for _, c in usages):
                runaway_ids.append(qid)

    return {
        "file": str(path),
        "traces": traces,
        "answered": answered,
        "answered_share": round(answered / traces, 4) if traces else 0.0,
        "turns": len(turn_lengths),
        "turn_max": max(turn_lengths) if turn_lengths else 0,
        "turn_median": st.median(turn_lengths) if turn_lengths else 0,
        "turn_q99": q(turn_lengths, 0.99),
        "turns_over_cap": sum(x > CAP for x in turn_lengths),
        "turns_over_cap_share": round(sum(x > CAP for x in turn_lengths) / len(turn_lengths), 6) if turn_lengths else 0.0,
        "turns_at_cap": sum(x == CAP for x in turn_lengths),
        "finish_reasons": dict(sorted(reasons.items(), key=lambda kv: -kv[1])),
        "failures": dict(sorted(failures.items(), key=lambda kv: -kv[1])),
        "context_max": max(max_context) if max_context else 0,
        "context_q99": q(max_context, 0.99),
        "context_median": st.median(max_context) if max_context else 0,
        "failed_ids": failed_ids,
        "runaway_ids": runaway_ids,
    }


def main(argv: list[str]) -> int:
    new_root, old_root, out_path = Path(argv[1]), Path(argv[2]), Path(argv[3])
    report = {"new": {}, "old": {}, "overlap": {}}
    for path in sorted(new_root.rglob("*.jsonl")):
        key = f"{path.parts[-4]}/{path.parts[-3]}"
        report["new"][key] = describe(path)
    for path in sorted(old_root.rglob("*.jsonl")):
        key = f"{path.parts[-4]}/{path.parts[-3]}"
        report["old"][key] = describe(path)

    for key, new in report["new"].items():
        old = report["old"].get(key)
        if not old:
            continue
        old_runaway = set(old["runaway_ids"])
        new_window_failed = {qid for qid, reason in new["failed_ids"] if reason == "endpoint_http_400"}
        old_failed = {qid for qid, _ in old["failed_ids"]}
        report["overlap"][key] = {
            "old_runaway_count": len(old_runaway),
            "new_window_failed_count": len(new_window_failed),
            "new_window_failed_were_runaway": len(new_window_failed & old_runaway),
            "new_window_failed_were_failed_before": len(new_window_failed & old_failed),
            "old_failed_count": len(old_failed),
            "answered_before_now_failed": len(new_window_failed - old_failed),
        }

    body = json.dumps(report, ensure_ascii=False, indent=1)
    out_path.write_text(body + "\n", encoding="utf-8")
    for key, new in report["new"].items():
        print(f"\n== {key}")
        print(f"  трасс {new['traces']}, с ответом {new['answered']} ({new['answered_share']})")
        print(f"  ходов {new['turns']}, наибольший {new['turn_max']}, медиана {new['turn_median']}, "
              f"q99 {new['turn_q99']}, длиннее 7000: {new['turns_over_cap']} "
              f"({new['turns_over_cap_share']}), ровно на пределе {new['turns_at_cap']}")
        print(f"  контекст: медиана {new['context_median']}, q99 {new['context_q99']}, макс {new['context_max']}")
        print(f"  причины завершения хода: {new['finish_reasons']}")
        print(f"  отказы: {new['failures']}")
        if key in report["overlap"]:
            print(f"  сверка со старым каноном: {report['overlap'][key]}")
    print(f"\n# разбор сохранён: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
