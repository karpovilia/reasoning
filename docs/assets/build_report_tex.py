#!/usr/bin/env python3
"""Сборка report_2026_08_19.tex из docs/reports/report_2026_08_18.md для Overleaf.

Запуск: python3 docs/assets/build_report_tex.py <путь к клону проекта Overleaf>

Что делает и почему именно так:
  1. подписи таблиц вида «Таблица N. Текст», стоящие в markdown сразу за таблицей,
     вынимаются из текста и превращаются в \\caption, иначе печатались бы дважды;
  2. pandoc переводит markdown в LaTeX;
  3. знак минус U+2212 заменяется на математический, шрифт T2A его не набирает;
  4. в длинных путях и составных именах расставляются точки разрыва строки,
     иначе они вылезают за поля;
  5. таблицы переводятся из longtable в tabular. Это не косметика: longtable
     пишет свои метки в файл .aux, и устаревший .aux в кэше Overleaf ломает
     сборку. Все таблицы отчёта короче страницы, перенос им не нужен;
  6. ширина колонки считается по длине её содержимого, равные доли одновременно
     рвут числовые колонки и ломают переносом длинные названия моделей;
  7. рисунки оформляются плавающими с подписями;
  8. счётчики разделов молчат, нумерация в отчёте своя, внутри текста заголовков.
"""
import re
import subprocess
import sys
import tempfile
from pathlib import Path

MD = Path("/home/ki/repos/reasoning_clean/docs/reports/report_2026_08_18.md")
ASSETS = Path("/home/ki/repos/reasoning_clean/docs/assets")
FIGS = {
    "fig_dispersion_by_effort.png": "Разброс длины рассуждения на одном и том же вопросе.",
    "fig_student_loss.png": "Кривые потери двух прогонов дообучения ученика-разметчика: "
                            "темп 1e-4 против 5e-5, логарифмическая шкала по потере.",
}

PREAMBLE = r"""% Отчёт о работе. Собран из docs/reports/report_2026_08_18.md чистового
% репозитория скриптом docs/assets/build_report_tex.py; правки вносить в markdown.
% В Overleaf: Menu -> Main document -> report_2026_08_19.tex, компилятор pdfLaTeX.
\documentclass[11pt]{article}
\usepackage[T2A]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage[english,russian]{babel}
\usepackage{amsmath,amssymb,amsfonts}
\usepackage{booktabs}
\usepackage{array}
\usepackage{graphicx}
\usepackage[margin=2.2cm]{geometry}
\usepackage{microtype}
\usepackage[labelformat=empty,skip=4pt]{caption}
% lineno нужен не для нумерации строк: он определяет \@LN@col и родню, которыми
% засорён кэшированный output.aux Overleaf после сборки budget_eacl.tex в режиме
% review. Без него первая же компиляция после смены главного файла падает с
% Undefined control sequence в output.aux. Команда \linenumbers не вызывается,
% поэтому на вид документа пакет не влияет. Не удалять.
\usepackage{lineno}
\usepackage[colorlinks=true,linkcolor=blue,urlcolor=blue]{hyperref}
\graphicspath{{}}
\setlength{\emergencystretch}{3em}
\setlength{\tabcolsep}{4pt}
\setcounter{secnumdepth}{0}
\setcounter{tocdepth}{2}
\providecommand{\tightlist}{%
  \setlength{\itemsep}{0pt}\setlength{\parskip}{0pt}}

\title{Отчёт о работе: разметка трасс рассуждения, бюджетирование и события}
\author{}
\date{19 августа 2026 года}

\begin{document}

\maketitle
\tableofcontents
\newpage

"""

CAPTION_RE = re.compile(r"^Таблица \d+\.\s.+$")
ROW_SEP = "\\" + "\\"          # разделитель строк таблицы в LaTeX
CELL_CLEAN = re.compile(r"\\[a-zA-Z]+\s*|[{}]")


def split_captions(md_text):
    """Вынимает подписи таблиц из markdown: текст без них и список подписей."""
    lines = md_text.split("\n")
    out, caps, i = [], [], 0
    while i < len(lines):
        ln = lines[i]
        if ln.startswith("|") and i + 1 < len(lines) and lines[i + 1].startswith("|---"):
            while i < len(lines) and lines[i].startswith("|"):
                out.append(lines[i])
                i += 1
            j = i
            while j < len(lines) and lines[j].strip() == "":
                j += 1
            if j < len(lines) and CAPTION_RE.match(lines[j]):
                caps.append(lines[j])
                i = j + 1
                while i < len(lines) and lines[i].strip() == "":
                    i += 1
                out.append("")
            continue
        out.append(ln)
        i += 1
    return "\n".join(out), caps


def col_spec(body, n):
    """Спецификация колонок: доля ширины пропорциональна длине содержимого."""
    widths = [1] * n
    for row in body.split(ROW_SEP):
        cells = row.split("&")
        if len(cells) != n:
            continue
        for k, cell in enumerate(cells):
            txt = CELL_CLEAN.sub("", cell).strip()
            widths[k] = max(widths[k], min(len(txt), 44))
    total = sum(widths)
    share = [w / total for w in widths]
    cap = 0.34
    for _ in range(4):
        over = [i for i, x in enumerate(share) if x > cap]
        if not over:
            break
        excess = sum(share[i] - cap for i in over)
        for i in over:
            share[i] = cap
        rest = [i for i in range(n) if i not in over]
        if not rest:
            break
        for i in rest:
            share[i] += excess / len(rest)
    cols = "".join(
        r">{\raggedright\arraybackslash}p{\dimexpr%.4f\linewidth-2\tabcolsep\relax}" % x
        for x in share)
    return "@{}" + cols + "@{}"


def mark_cols(m):
    """Помечает число колонок; ширины считаются позже, когда доступно тело."""
    return "\\begin{longtable}[]{@@COLS%d@@}" % len(m.group(1))


def to_tabular(blk, caps, counter):
    mark = re.match(r"\\begin\{longtable\}\[\]\{(.*?)\}\n", blk, re.S).group(1)
    ncols = int(re.match(r"@@COLS(\d+)@@", mark).group(1))
    body = blk.split("\n", 1)[1].rsplit("\\end{longtable}", 1)[0]
    for junk in ("\\endhead\n", "\\endlastfoot\n", "\\bottomrule\n"):
        body = body.replace(junk, "")
    body = body.replace("\\noalign{}\n", "\n").replace("\\noalign{}", "").strip("\n")
    cap = ""
    if counter[0] < len(caps):
        cap = "\\caption{%s}\n" % caps[counter[0]]
        counter[0] += 1
    return ("\\begin{table}[htbp]\n\\centering\n\\small\n%s\\begin{tabular}{%s}\n%s\n"
            "\\bottomrule\n\\end{tabular}\n\\end{table}" % (cap, col_spec(body, ncols), body))


def main(dest: Path):
    md_clean, caps = split_captions(MD.read_text())
    with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False) as fh:
        fh.write(md_clean)
        tmp = fh.name
    body = subprocess.run(["pandoc", "-f", "gfm", "-t", "latex", "--wrap=none", tmp],
                          capture_output=True, text=True, check=True).stdout
    print("подписей таблиц найдено:", len(caps))

    for name, cap in FIGS.items():
        old = "\\pandocbounded{\\includegraphics[keepaspectratio]{../assets/%s}}" % name
        if old not in body:
            raise SystemExit("не найден рисунок в конвертированном тексте: " + name)
        body = body.replace(old, "@@FIG:%s:%s@@" % (name, cap))

    body = body.replace("\u2212", "$-$")
    body = body.replace("/", "/\\allowbreak{}").replace("\\_", "\\_\\allowbreak{}")
    body = re.sub(r"\\begin\{longtable\}\[\]\{@\{\}([lcr]+)@\{\}\}", mark_cols, body)

    counter = [0]
    body = re.sub(r"\\begin\{longtable\}\[\]\{.*?\}\n.*?\\end\{longtable\}",
                  lambda m: to_tabular(m.group(0), caps, counter), body, flags=re.S)
    if counter[0] != len(caps):
        raise SystemExit("подписей %d, таблиц %d - расхождение" % (len(caps), counter[0]))
    if "@@COLS" in body:
        raise SystemExit("остались неразобранные метки колонок")

    body = re.sub(r"@@FIG:([^:]+):([^@]+)@@",
                  lambda m: ("\\begin{figure}[htbp]\n\\centering\n"
                             "\\includegraphics[width=\\linewidth]{%s}\n"
                             "\\caption{%s}\n\\end{figure}" % (m.group(1), m.group(2))),
                  body)

    (dest / "report_2026_08_19.tex").write_text(PREAMBLE + body + "\n\n\\end{document}\n")
    for name in FIGS:
        (dest / name).write_bytes((ASSETS / name).read_bytes())
    print("собрано в", dest / "report_2026_08_19.tex")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    main(Path(sys.argv[1]))
