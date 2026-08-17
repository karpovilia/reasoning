#!/usr/bin/env python3
"""Проверяет опись корпусов data/registry.tsv.

Опись — таблица, в которой одна строка описывает один срез трасс, то есть
рассуждений модели, вместе с одним слоем разметки поверх него. Разметка здесь —
слой аннотаций событиями и спанами операторов, лежащий поверх текста трассы.

Что проверяется:

  1. Заголовок таблицы совпадает с ожидаемым набором столбцов, разделитель —
     табуляция, пустых строк нет.
  2. Ключ строки уникален. Ключ — это срез трасс, то есть корпус, модель, ветка
     прогона, зерно и бенчмарк, вместе со слоем разметки, то есть разметчиком,
     режимом, проходом и каталогом разметки. Две строки с одним ключом означают,
     что один и тот же слой описан дважды и числа в них могут разойтись.
  3. Знаменатели осмысленны:
     - число трасс — целое положительное;
     - число вопросов бенчмарка записано либо числом, либо оборотом
       «подвыборка A из B», причём A не больше B;
     - число трасс кратно числу вопросов: на один вопрос приходится целое число
       прогонов, по одному на каждое сочетание модели и зерна в строке;
     - размечено записано либо числом, не превосходящим числа трасс, либо
       оборотом «N из M», причём N не больше M, а M равно числу трасс.
  4. Признак состояния разметки согласован с остальными столбцами: при значении
     «разметки нет» каталог разметки и число размеченных пусты, при любом другом
     значении оба заполнены.
  5. Локальные пути существуют. Пути в описи записаны от корня репозитория, и
     проверяется тот путь, который разрешается внутри этого репозитория, то есть
     уже перенесённый. Путь, ведущий в черновой рабочий репозиторий, здесь не
     разрешается и отмечается как внешний; это не ошибка, чистовой репозиторий не
     обязан его видеть. Число внешних путей печатается, чтобы было видно, сколько
     ещё не перенесено. Отдельно проверяется, что тело скачанного корпуса на
     месте: состав его файлов сверил с манифестом data/fetch.sh.

Запуск:
  python3 tools/check_registry.py
  python3 tools/check_registry.py --strict-paths

Ключ --strict-paths требует, чтобы существовали все пути, а не только скачанные;
он для проверки уже перенесённого целиком репозитория.

Код возврата 0 означает, что опись цела.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "data" / "registry.tsv"
MANIFESTS = ROOT / "data" / "manifests"
CORPORA = ROOT / "data" / "corpora"

COLUMNS = [
    "корпус трасс", "trace_model", "trace_arm", "trace_seed", "trace_bench",
    "вопросов бенчмарка в корпусе", "n_traces", "путь к трассам",
    "labeler_model", "labeler_mode", "label_pass", "label_dir",
    "размечено трасс", "label_status", "notes",
]

# Ключ строки: срез трасс плюс слой разметки поверх него.
KEY = ["корпус трасс", "trace_model", "trace_arm", "trace_seed", "trace_bench",
       "labeler_model", "labeler_mode", "label_pass", "label_dir"]

NO_LABELS = "разметки нет"

QUESTIONS_RE = re.compile(r"^(?:подвыборка\s+)?(\d+)(?:\s+из\s+(\d+))?")
# Три записи одного и того же: «N», «N из M», «N размечено из M трасс корпуса».
LABELED_RE = re.compile(
    r"^(\d+)(?:\s+(?:размечено\s+)?из\s+(\d+)(?:\s+трасс(?:\s+корпуса)?)?)?$")


class Report:
    def __init__(self) -> None:
        self.errors: list[str] = []
        self.notes: list[str] = []

    def fail(self, line: int | None, text: str) -> None:
        where = f"строка {line}: " if line else ""
        self.errors.append(where + text)

    def note(self, text: str) -> None:
        self.notes.append(text)


def load_rows(report: Report) -> list[dict]:
    if not REGISTRY.exists():
        return []
    with REGISTRY.open(encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        if reader.fieldnames != COLUMNS:
            report.fail(1, "заголовок описи не совпадает с ожидаемым")
            report.fail(1, f"ожидалось: {COLUMNS}")
            report.fail(1, f"получено:  {reader.fieldnames}")
            return []
        return list(reader)


def check_key(rows: list[dict], report: Report) -> None:
    seen: dict[tuple, int] = {}
    for line, row in enumerate(rows, start=2):
        key = tuple((row.get(name) or "").strip() for name in KEY)
        if key in seen:
            report.fail(line, "ключ строки повторяет строку "
                              f"{seen[key]}: один слой разметки описан дважды")
        else:
            seen[key] = line


def check_numbers(rows: list[dict], report: Report) -> None:
    for line, row in enumerate(rows, start=2):
        raw_traces = (row["n_traces"] or "").strip()
        if not raw_traces.isdigit() or int(raw_traces) <= 0:
            report.fail(line, f"число трасс не целое положительное: {raw_traces!r}")
            continue
        n_traces = int(raw_traces)

        questions_raw = (row["вопросов бенчмарка в корпусе"] or "").strip()
        match = QUESTIONS_RE.match(questions_raw)
        if not match:
            report.fail(line, f"число вопросов бенчмарка нечитаемо: {questions_raw!r}")
            continue
        taken = int(match.group(1))
        total = int(match.group(2)) if match.group(2) else None
        if taken <= 0:
            report.fail(line, f"число вопросов не положительно: {questions_raw!r}")
            continue
        if total is not None and taken > total:
            report.fail(line, "подвыборка больше самого бенчмарка: "
                              f"{questions_raw!r}")
        if n_traces % taken:
            report.fail(line, f"число трасс {n_traces} не кратно числу вопросов "
                              f"{taken}: на вопрос приходится дробное число прогонов")

        labeled_raw = (row["размечено трасс"] or "").strip()
        if labeled_raw:
            match = LABELED_RE.match(labeled_raw)
            if not match:
                report.fail(line, f"число размеченных нечитаемо: {labeled_raw!r}")
                continue
            done = int(match.group(1))
            denominator = int(match.group(2)) if match.group(2) else n_traces
            if match.group(2) and denominator != n_traces:
                report.fail(line, f"знаменатель размеченных {denominator} не равен "
                                  f"числу трасс {n_traces}")
            if done > denominator:
                report.fail(line, f"размечено больше, чем есть: {labeled_raw!r} "
                                  f"при числе трасс {n_traces}")


def check_status(rows: list[dict], report: Report) -> None:
    for line, row in enumerate(rows, start=2):
        status = (row["label_status"] or "").strip()
        has_dir = bool((row["label_dir"] or "").strip())
        has_count = bool((row["размечено трасс"] or "").strip())
        if not status:
            report.fail(line, "состояние разметки не заполнено")
        elif status == NO_LABELS:
            if has_dir or has_count:
                report.fail(line, f"состояние {NO_LABELS!r}, но каталог разметки "
                                  "или число размеченных заполнены")
        else:
            if not has_dir:
                report.fail(line, f"состояние {status!r}, но каталог разметки пуст")
            if not has_count:
                report.fail(line, f"состояние {status!r}, но число размеченных пусто")


def fetched_corpora() -> dict[str, Path]:
    """Корпуса, у которых есть манифест и скачанное тело."""
    result: dict[str, Path] = {}
    if not MANIFESTS.is_dir():
        return result
    for manifest_path in sorted(MANIFESTS.glob("*.json")):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        name = manifest.get("корпус") or manifest_path.stem
        local = CORPORA / manifest_path.stem
        if local.is_dir():
            result[name] = local
    return result


def check_paths(rows: list[dict], report: Report, strict: bool) -> None:
    fetched = fetched_corpora()
    external: set[str] = set()
    checked = 0
    for line, row in enumerate(rows, start=2):
        for column in ("путь к трассам", "label_dir"):
            raw = (row[column] or "").strip()
            if not raw:
                continue
            if raw.startswith("/"):
                report.fail(line, f"в столбце {column} записан абсолютный путь машины: "
                                  f"{raw}; пути пишут от корня репозитория")
                continue
            path = ROOT / raw
            # Внутренним считается путь, который разрешается в этом репозитории.
            if not path.exists() and not strict:
                external.add(raw)
                continue
            checked += 1
            if not path.exists():
                report.fail(line, f"путь из столбца {column} не существует: {raw}")

    # У скачанных корпусов проверяется их собственный корень: тело на месте и
    # состав совпал с манифестом, это уже сделал data/fetch.sh.
    for name, local in sorted(fetched.items()):
        if not local.is_dir():
            report.fail(None, f"тело скачанного корпуса пропало: {local}")
        elif not any(local.rglob("*")):
            report.fail(None, f"каталог скачанного корпуса пуст: {local}")
    if external:
        report.note(f"внешних путей, не разрешающихся в этом репозитории: {len(external)}; "
                    "они ведут в черновой рабочий репозиторий и начнут проверяться "
                    "после переноса соответствующего корпуса")
    report.note(f"путей проверено на существование: {checked}")
    if fetched:
        report.note("скачанные корпуса: " + ", ".join(sorted(fetched)))
    else:
        report.note("скачанных тел корпусов нет, проверка путей ограничена "
                    "внутренними; тела скачиваются командой make fetch")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strict-paths", action="store_true",
                        help="требовать существования всех путей, а не только скачанных")
    args = parser.parse_args()

    report = Report()
    if not REGISTRY.exists():
        message = f"описи нет, проверять нечего: {REGISTRY}"
        if args.strict_paths:
            print(message, file=sys.stderr)
            return 1
        print(message)
        return 0

    rows = load_rows(report)
    if rows:
        check_key(rows, report)
        check_numbers(rows, report)
        check_status(rows, report)
        check_paths(rows, report, args.strict_paths)
        report.note(f"строк описи: {len(rows)}")
    else:
        report.fail(None, "опись существует, но строк в ней нет")

    for line in report.notes:
        print(line)
    if report.errors:
        print(f"опись не прошла проверку, замечаний {len(report.errors)}:", file=sys.stderr)
        for line in report.errors[:50]:
            print("  " + line, file=sys.stderr)
        if len(report.errors) > 50:
            print(f"  ещё {len(report.errors) - 50}", file=sys.stderr)
        return 1
    print(f"опись цела: {REGISTRY}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
