#!/usr/bin/env python3
"""Сборка report_2026_08_19.tex из docs/reports/report_2026_08_18.md для Overleaf.

Запуск: python3 docs/assets/build_report_tex.py <путь к клону проекта Overleaf>

Что делает и почему именно так:
  1. pandoc переводит markdown в LaTeX;
  2. знак минус U+2212 заменяется на математический, шрифт T2A его не набирает;
  3. в длинных путях и составных именах расставляются точки разрыва строки,
     иначе они вылезают за поля;
  4. таблицы переводятся из longtable в tabular. Это не косметика: longtable
     пишет свои метки в файл .aux, и устаревший .aux в кэше Overleaf ломает
     сборку сообщением про неопределённую команду. Все таблицы отчёта короче
     страницы, перенос им не нужен;
  5. рисунки оформляются плавающими с подписями;
  6. счётчики разделов молчат, нумерация в отчёте своя, внутри текста заголовков.
"""
import re
import subprocess
import sys
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


def to_tabular(m):
    blk = m.group(0)
    spec = re.match(r"\\begin\{longtable\}\[\]\{(.*?)\}\n", blk, re.S).group(1)
    body = blk.split("\n", 1)[1].rsplit("\\end{longtable}", 1)[0]
    for junk in ("\\endhead\n", "\\endlastfoot\n", "\\bottomrule\n"):
        body = body.replace(junk, "")
    body = body.replace("\\noalign{}\n", "\n").replace("\\noalign{}", "").strip("\n")
    return ("\\begin{table}[htbp]\n\\centering\n\\small\n\\begin{tabular}{%s}\n%s\n"
            "\\bottomrule\n\\end{tabular}\n\\end{table}" % (spec, body))


def widen(m):
    n = len(m.group(1))
    col = (r">{\raggedright\arraybackslash}p{\dimexpr(\linewidth-%d\tabcolsep)/%d\relax}"
           % (2 * n, n))
    return "\\begin{longtable}[]{@{}" + col * n + "@{}}"


def main(dest: Path):
    body = subprocess.run(["pandoc", "-f", "gfm", "-t", "latex", "--wrap=none", str(MD)],
                          capture_output=True, text=True, check=True).stdout

    for name, cap in FIGS.items():
        old = "\\pandocbounded{\\includegraphics[keepaspectratio]{../assets/%s}}" % name
        if old not in body:
            raise SystemExit("не найден рисунок в конвертированном тексте: " + name)
        body = body.replace(old, "@@FIG:%s:%s@@" % (name, cap))

    body = body.replace("\u2212", "$-$")
    body = body.replace("/", "/\\allowbreak{}").replace("\\_", "\\_\\allowbreak{}")
    body = re.sub(r"\\begin\{longtable\}\[\]\{@\{\}([lcr]+)@\{\}\}", widen, body)
    body = re.sub(r"\\begin\{longtable\}\[\]\{.*?\}\n.*?\\end\{longtable\}",
                  to_tabular, body, flags=re.S)
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
