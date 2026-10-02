#!/usr/bin/env python3
"""Собирает оглавление документов docs/README.md.

Оглавление не пишут руками. Его собирает этот скрипт по заголовкам самих файлов:
берётся первый заголовок первого уровня, строка с датой, если она есть, и первый
содержательный абзац. Так оглавление не расходится с содержимым: расхождение
ловится проверкой и останавливает коммит.

Запуск:
  python3 tools/build_docs_index.py           записать оглавление
  python3 tools/build_docs_index.py --check   только сверить

Код возврата 0 означает, что записанное оглавление совпадает с собранным.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
INDEX = DOCS / "README.md"

# Раздел оглавления: каталог, заголовок раздела, пояснение назначения раздела.
SECTIONS = [
    ("release", "Код и модели выпуска",
     "Установка, обучение, инференс, устройство моделей и формат данных."),
    ("findings", "Выводы",
     "Короткие записи выводов, по одному выводу на файл. Отрицательный результат "
     "оформляется здесь так же полно, как положительный."),
    ("reports", "Отчёты",
     "Развёрнутые разборы линий работы: постановка, числа, оговорки."),
    ("papers", "Статьи",
     "Тексты статей и их сборки. Файлы pdf хранятся через git lfs."),
]

MARK_BEGIN = "<!-- собрано tools/build_docs_index.py, руками не править -->"
MARK_END = "<!-- конец собранной части -->"

DATE_RE = re.compile(r"^\s*Дата:\s*(?P<date>[0-9]{4}-[0-9]{2}-[0-9]{2})", re.M)
H1_RE = re.compile(r"^#\s+(?P<title>.+?)\s*$", re.M)


def first_paragraph(text: str) -> str:
    """Первый содержательный абзац после заголовка, сжатый в одну строку."""
    body = H1_RE.split(text, maxsplit=1)
    tail = body[-1] if len(body) > 1 else text
    chunk: list[str] = []
    for line in tail.splitlines():
        stripped = line.strip()
        if not stripped:
            if chunk:
                break
            continue
        if stripped.startswith(("#", "|", "-", "*", ">", "```")):
            if chunk:
                break
            continue
        chunk.append(stripped)
    return " ".join(chunk)


def shorten(text: str, limit: int = 200) -> str:
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    cut = text[:limit].rsplit(" ", 1)[0]
    return cut + "…"


def describe(path: Path) -> dict:
    """Заголовок, дата и краткое изложение одного документа."""
    if path.suffix.lower() != ".md":
        return {"title": path.name, "date": "", "summary": "", "path": path}
    text = path.read_text(encoding="utf-8", errors="replace")
    head = H1_RE.search(text)
    date = DATE_RE.search(text)
    return {
        "title": head.group("title") if head else path.stem,
        "date": date.group("date") if date else "",
        "summary": shorten(first_paragraph(text)),
        "path": path,
    }


def collect(folder: Path) -> list[dict]:
    if not folder.is_dir():
        return []
    items = [p for p in sorted(folder.iterdir())
             if p.is_file() and not p.name.startswith(".") and p.name != "README.md"]
    return [describe(p) for p in items]


def render() -> str:
    lines = [
        "# Оглавление документов",
        "",
        "Файл собирается скриптом "
        "tools/build_docs_index.py по заголовкам самих "
        "документов. Править его руками нельзя: проверка make check сверит записанное "
        "с собранным и остановит коммит при расхождении.",
        "",
        MARK_BEGIN,
        "",
    ]
    for folder_name, heading, purpose in SECTIONS:
        folder = DOCS / folder_name
        entries = collect(folder)
        lines.append(f"## {heading}")
        lines.append("")
        lines.append(purpose)
        lines.append("")
        lines.append(f"Каталог: {folder.relative_to(ROOT)}")
        lines.append("")
        if not entries:
            lines.append("Пока пусто.")
            lines.append("")
            continue
        for entry in entries:
            rel = entry["path"].relative_to(ROOT)
            date = entry["date"] or "дата не указана"
            lines.append(f"### {entry['title']}")
            lines.append("")
            lines.append(f"Дата: {date}. Файл: {rel}")
            if entry["summary"]:
                lines.append("")
                lines.append(entry["summary"])
            lines.append("")
    lines.append(MARK_END)
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                        help="ничего не писать, только сверить записанное с собранным")
    args = parser.parse_args()

    built = render()
    if args.check:
        if not INDEX.exists():
            print(f"оглавление отсутствует: {INDEX.relative_to(ROOT)}", file=sys.stderr)
            print("собрать: python3 tools/build_docs_index.py", file=sys.stderr)
            return 1
        current = INDEX.read_text(encoding="utf-8")
        if current != built:
            print(f"оглавление разошлось с содержимым каталога: {INDEX.relative_to(ROOT)}", file=sys.stderr)
            print("пересобрать: python3 tools/build_docs_index.py", file=sys.stderr)
            return 1
        print(f"оглавление совпадает с содержимым: {INDEX.relative_to(ROOT)}")
        return 0

    DOCS.mkdir(parents=True, exist_ok=True)
    INDEX.write_text(built, encoding="utf-8")
    print(f"оглавление записано: {INDEX.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
