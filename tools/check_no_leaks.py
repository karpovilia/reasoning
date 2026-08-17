#!/usr/bin/env python3
"""Ловит в репозитории то, чего в нём быть не должно: закрытые бенчмарки и ключи доступа.

Подход взят у проверки публикуемого снимка toloka/build/check_no_gpqa.py в
черновом рабочем репозитории и расширен на второй закрытый бенчмарк и на поиск
ключей.

Две группы проверок.

Первая — канарейки закрытых бенчмарков. Условия доступа к GPQA-diamond и к hle
закрытые, поэтому ни текст вопроса, ни его разбор, ни идентификатор записи, ни
текст рассуждения модели по такому вопросу в репозиторий не попадают. Проверка
сравнивает содержимое репозитория с самими бенчмарками, которые лежат вне
репозитория: это файлы gpqa/gpqa_diamond.csv и hle/test.parquet внутри корня
эталонов.

Корень эталонов задаётся переменной среды REASONING_BENCH_ROOT либо файлом
bench_root в каталоге настроек ~/.config/reasoning-data, который лежит вне git.
Из бенчмарков берутся идентификаторы записей, канареечные строки бенчмарка и
начальные фрагменты текстов вопросов и разборов длиной 120 символов; фрагмент
такой длины случайно совпасть не может. Совпадение хотя бы одного фрагмента —
утечка.

Имя бенчмарка само по себе уликой не считается: опись корпусов законно называет
gpqa_diamond и hle в столбце бенчмарка. Уликой считается текст и идентификатор
вопроса.

Вторая — ключи доступа. Ищутся ключи поставщиков моделей, токены Overleaf,
закрытые части ключей, ключи объектного хранилища и содержимое файлов среды.

Чтение hle требует библиотеки pyarrow. Если её нет в текущем интерпретаторе,
скрипт сам перезапускается интерпретатором из переменной среды REASONING_PY либо
из файла python в каталоге настроек. Если и там её нет, проверка hle объявляется
невыполненной: при ключе --strict это ошибка, без него — предупреждение, чтобы
проверка работала на машине без библиотек.

Запуск:
  python3 tools/check_no_leaks.py
  python3 tools/check_no_leaks.py --staged
  python3 tools/check_no_leaks.py --strict

Ключ --staged проверяет только то, что подготовлено к коммиту; его использует
крючок перед коммитом.

Код возврата 0 означает, что улик нет.
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Пути к эталонам и к интерпретатору в репозитории не хранятся: они разные на
# разных машинах, а абсолютных путей машин в файлах репозитория быть не должно.
# Значение берётся из переменной среды, а если её нет — из файла в каталоге
# настроек, лежащем вне git.
SETTINGS = Path(os.environ.get("REASONING_DATA_SETTINGS",
                               Path.home() / ".config" / "reasoning-data"))


def setting(variable: str, file_name: str) -> str:
    """Значение настройки: сначала переменная среды, потом файл вне git."""
    from_env = os.environ.get(variable, "")
    if from_env:
        return from_env
    path = SETTINGS / file_name
    if path.is_file():
        return path.read_text(encoding="utf-8").strip()
    return ""


BENCH_ROOT = Path(setting("REASONING_BENCH_ROOT", "bench_root") or "reasonops_benchmarks")
GPQA_CSV = BENCH_ROOT / "gpqa" / "gpqa_diamond.csv"
HLE_PARQUET = BENCH_ROOT / "hle" / "test.parquet"

FALLBACK_PYTHON = [setting("REASONING_PY", "python")]

FRAGMENT = 120                  # длина фрагмента текста, по которому ищется утечка
IGNORED_DIRS = {".git", ".venv", "__pycache__", "node_modules", "data/corpora"}
BINARY_SUFFIXES = {".pdf", ".pptx", ".png", ".jpg", ".jpeg", ".gif", ".zst",
                   ".gz", ".xz", ".npz", ".npy", ".parquet", ".so", ".bin",
                   ".safetensors", ".pt", ".pyc", ".woff", ".woff2", ".ico"}
MAX_SCAN_BYTES = 64 * 2 ** 20   # файл крупнее не сканируется, а объявляется ошибкой

GPQA_TEXT_COLUMNS = ("Question", "Explanation", "Pre-Revision Question",
                     "Pre-Revision Explanation", "Extra Revised Question",
                     "Extra Revised Explanation")

# Ключи доступа. Каждое правило — имя и выражение.
SECRET_RULES = [
    ("ключ Overleaf", re.compile(r"olp_[A-Za-z0-9]{16,}")),
    ("закрытая часть ключа", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
    ("ключ OpenAI или совместимого поставщика", re.compile(r"\bsk-[A-Za-z0-9_\-]{20,}")),
    ("ключ Anthropic", re.compile(r"\bsk-ant-[A-Za-z0-9_\-]{20,}")),
    ("ключ OpenRouter", re.compile(r"\bsk-or-v1-[A-Za-z0-9]{32,}")),
    ("ключ Hugging Face", re.compile(r"\bhf_[A-Za-z0-9]{30,}")),
    ("ключ доступа AWS или совместимого хранилища", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("токен GitHub", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}")),
    ("объявление секрета в тексте",
     re.compile(r"(?i)\b(secret[_-]?key|api[_-]?key|password|access[_-]?key)\b\s*[:=]\s*"
                r"['\"]?[A-Za-z0-9/+_\-]{16,}")),
]

# Имена файлов, которых в репозитории быть не должно вовсе.
FORBIDDEN_NAMES = re.compile(
    r"(?i)^(\.env|\.overleaf|.*\.key|.*\.pem|id_rsa|id_ed25519|credentials|"
    r"config\.json\.mcli)$")

# Файлы данных с закрытым бенчмарком в имени: трассы по закрытым вопросам.
DATA_SUFFIXES = {".jsonl", ".json", ".csv", ".tsv", ".txt", ".parquet"}
CLOSED_BENCH_IN_NAME = re.compile(r"(?i)(gpqa|hle)")

# Файлы, которым разрешено называть закрытый бенчмарк в имени: это описи, а не тела.
NAME_ALLOWLIST = {"data/registry.tsv"}


class Report:
    def __init__(self) -> None:
        self.errors: list[str] = []
        self.warnings: list[str] = []
        self.notes: list[str] = []


def norm(text: str) -> str:
    return " ".join((text or "").split())


def ensure_pyarrow() -> bool:
    """Есть ли pyarrow; при отсутствии — перезапуск подходящим интерпретатором."""
    try:
        import pyarrow  # noqa: F401
        return True
    except ImportError:
        pass
    if os.environ.get("REASONING_LEAKS_REEXEC"):
        return False
    for candidate in FALLBACK_PYTHON:
        if not candidate:
            continue
        exe = Path(candidate)
        if not exe.is_file() or not os.access(exe, os.X_OK):
            continue
        probe = subprocess.run([str(exe), "-c", "import pyarrow"],
                               capture_output=True)
        if probe.returncode == 0:
            env = dict(os.environ, REASONING_LEAKS_REEXEC="1")
            os.execve(str(exe), [str(exe), str(Path(__file__).resolve()), *sys.argv[1:]], env)
    return False


def gpqa_canaries(report: Report) -> tuple[set[str], set[str]]:
    """Идентификаторы и фрагменты текстов GPQA-diamond."""
    ids: set[str] = set()
    fragments: set[str] = set()
    if not GPQA_CSV.exists():
        report.warnings.append(f"эталон GPQA не найден, проверка не выполнена: {GPQA_CSV}")
        return ids, fragments
    csv.field_size_limit(10 ** 7)
    with GPQA_CSV.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            for key in ("Record ID", "Canary String"):
                value = norm(row.get(key, ""))
                if value:
                    ids.add(value)
            for key in GPQA_TEXT_COLUMNS:
                text = norm(row.get(key, ""))
                if len(text) >= FRAGMENT:
                    fragments.add(text[:FRAGMENT])
    report.notes.append(f"эталон GPQA: идентификаторов {len(ids)}, "
                        f"фрагментов текста {len(fragments)}")
    return ids, fragments


def hle_canaries(report: Report, strict: bool) -> tuple[set[str], set[str]]:
    """Идентификаторы и фрагменты текстов hle."""
    ids: set[str] = set()
    fragments: set[str] = set()
    if not HLE_PARQUET.exists():
        message = f"эталон hle не найден, проверка не выполнена: {HLE_PARQUET}"
        (report.errors if strict else report.warnings).append(message)
        return ids, fragments
    try:
        import pyarrow.parquet as pq
    except ImportError:
        message = ("для чтения hle нужна библиотека pyarrow; проверка hle не "
                   "выполнена. Интерпретатор с библиотекой задаётся переменной "
                   "среды REASONING_PY либо файлом python в каталоге настроек "
                   f"{SETTINGS}")
        (report.errors if strict else report.warnings).append(message)
        return ids, fragments

    table = pq.read_table(HLE_PARQUET)
    names = set(table.column_names)
    for column in ("id",):
        if column in names:
            for value in table.column(column).to_pylist():
                value = norm(str(value))
                if len(value) >= 8:
                    ids.add(value)
    for column in ("question", "answer", "rationale", "solution"):
        if column not in names:
            continue
        for value in table.column(column).to_pylist():
            text = norm(str(value or ""))
            if len(text) >= FRAGMENT:
                fragments.add(text[:FRAGMENT])
    report.notes.append(f"эталон hle: идентификаторов {len(ids)}, "
                        f"фрагментов текста {len(fragments)}")
    return ids, fragments


def staged_files() -> list[Path]:
    """Подготовленное к коммиту.

    До первого коммита ссылки HEAD ещё нет, и сравнение с ней даёт пустой
    список. Это опасная тишина: именно первый коммит и уносит в удалённый адрес
    всё содержимое сразу. Поэтому при отсутствии HEAD берётся весь индекс.
    """
    has_head = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--verify", "HEAD"],
                              capture_output=True).returncode == 0
    # Разделитель -z обязателен: без него git заменяет не-ASCII имена файлов
    # восьмеричными последовательностями в кавычках, такой путь не открывается,
    # файл молча выпадает из проверки и уезжает в коммит непроверенным.
    if has_head:
        command = ["git", "-C", str(ROOT), "diff", "--cached",
                   "--name-only", "-z", "--diff-filter=ACMR"]
    else:
        command = ["git", "-C", str(ROOT), "ls-files", "--cached", "-z"]
    out = subprocess.run(command, capture_output=True, check=True)
    names = [chunk for chunk in out.stdout.split(b"\0") if chunk]
    return [ROOT / os.fsdecode(name) for name in names]


def tracked_and_untracked_files() -> list[Path]:
    result: list[Path] = []
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(ROOT).as_posix()
        if any(rel == d or rel.startswith(d + "/") for d in IGNORED_DIRS):
            continue
        if any(part in IGNORED_DIRS for part in path.relative_to(ROOT).parts):
            continue
        result.append(path)
    return result


def check_names(paths: list[Path], report: Report) -> None:
    for path in paths:
        rel = path.relative_to(ROOT).as_posix()
        if FORBIDDEN_NAMES.match(path.name):
            report.errors.append(f"файл такого имени в репозитории не хранят: {path}")
        if rel in NAME_ALLOWLIST:
            continue
        if path.suffix.lower() in DATA_SUFFIXES and CLOSED_BENCH_IN_NAME.search(path.name):
            report.errors.append(
                f"файл данных назван по закрытому бенчмарку, это тело трасс: {path}")


def read_text(path: Path, report: Report) -> str | None:
    if path.suffix.lower() in BINARY_SUFFIXES:
        return None
    size = path.stat().st_size
    if size > MAX_SCAN_BYTES:
        report.errors.append(
            f"файл крупнее порога сканирования и не проверен: {path}, "
            f"{size} байт; такие файлы в репозитории не хранят, тело идёт в "
            "объектное хранилище")
        return None
    try:
        raw = path.read_bytes()
    except OSError as error:
        report.errors.append(f"файл не прочитан: {path}: {error}")
        return None
    if b"\x00" in raw[:8192]:
        return None
    return raw.decode("utf-8", errors="replace")


def check_contents(paths: list[Path], report: Report,
                   ids: set[str], fragments: set[str]) -> None:
    scanned = 0
    for path in paths:
        text = read_text(path, report)
        if text is None:
            continue
        scanned += 1
        for name, rule in SECRET_RULES:
            found = rule.search(text)
            if found:
                report.errors.append(
                    f"похоже на {name}: {path}, начиная с позиции {found.start()}")
        flat = norm(text)
        for value in ids:
            if value in flat:
                report.errors.append(
                    f"идентификатор закрытого бенчмарка в файле: {path}")
                break
        for fragment in fragments:
            if fragment in flat:
                report.errors.append(
                    f"текст закрытого бенчмарка в файле: {path}")
                break
    report.notes.append(f"просканировано текстовых файлов: {scanned}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--staged", action="store_true",
                        help="проверять только подготовленное к коммиту")
    parser.add_argument("--strict", action="store_true",
                        help="считать ошибкой невыполнимость любой из проверок")
    args = parser.parse_args()

    ensure_pyarrow()

    report = Report()
    paths = staged_files() if args.staged else tracked_and_untracked_files()
    # Путь, который назвал git, но которого нет на диске, — не повод молчать:
    # именно так файл и проскакивает мимо проверки.
    missing = [p for p in paths if not p.is_file()]
    paths = [p for p in paths if p.is_file()]
    for path in missing:
        report.errors.append(f"git называет файл, но открыть его не удалось: {path}")
    if not paths and not missing:
        print("проверять нечего: файлов нет")
        return 0

    gpqa_ids, gpqa_fragments = gpqa_canaries(report)
    hle_ids, hle_fragments = hle_canaries(report, args.strict)
    if args.strict and not gpqa_ids:
        report.errors.append("эталон GPQA недоступен, а запрошена строгая проверка")

    check_names(paths, report)
    check_contents(paths, report, gpqa_ids | hle_ids, gpqa_fragments | hle_fragments)

    for line in report.notes:
        print(line)
    for line in report.warnings:
        print("предупреждение: " + line, file=sys.stderr)
    if report.errors:
        print(f"найдено улик: {len(report.errors)}", file=sys.stderr)
        for line in report.errors[:50]:
            print("  " + line, file=sys.stderr)
        if len(report.errors) > 50:
            print(f"  ещё {len(report.errors) - 50}", file=sys.stderr)
        return 1
    print(f"улик нет, проверено файлов: {len(paths)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
