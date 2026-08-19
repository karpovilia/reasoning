#!/usr/bin/env python3
"""Два столбчатых рисунка про события: по наборам вопросов и по моделям.

Слева: сколько событий каждого типа приходится на трассу в каждом из трёх
наборов вопросов. Абсолютные счётчики сравнивать нельзя, наборы разного объёма,
поэтому всё нормировано на трассу.

Справа: сколько событий каждого типа приходится на тысячу токенов у двух
моделей на трёх уровнях усилия рассуждения. Нормировка на длину нужна потому,
что уровень усилия меняет длину в разы, и счётчик на трассу мерил бы длину.

Источники: числа наборов вопросов из
/home/ki/repos/reasoning/internal_signals_poc/lora_judge_2026_08_15 сводки,
перенесённые в таблицы отчёта; числа моделей из
/home/ki/repos/reasoning/internal_signals_poc/effort_repeats_2026_08_14/outputs/events.json
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path("/home/ki/repos/reasoning_clean/docs/assets/fig_events_profile.png")
EVENTS = Path("/home/ki/repos/reasoning/internal_signals_poc/effort_repeats_2026_08_14/outputs/events.json")

INK, MUTED, GRID = "#22252a", "#6b7280", "#dfe3e8"
TYPES = [("subgoal_done", "закрытие\nподцели"), ("verify", "проверка"),
         ("branch", "ветвление"), ("commit", "консолидация"),
         ("backtrack", "возврат\nназад"), ("failed_attempt", "неудачная\nпопытка")]

# события по наборам вопросов и число трасс в каждом
BENCH = {
    "bbh":          {"n": 2810, "subgoal_done": 13551, "verify": 11162, "branch": 7070,
                     "commit": 5629, "backtrack": 3989, "failed_attempt": 2609},
    "gpqa_diamond": {"n": 850,  "subgoal_done": 7024,  "verify": 5445,  "branch": 5983,
                     "commit": 1830, "backtrack": 3187, "failed_attempt": 2938},
    "math500":      {"n": 2707, "subgoal_done": 18469, "verify": 20158, "branch": 6763,
                     "commit": 5901, "backtrack": 3533, "failed_attempt": 2364},
}
C_BENCH = {"bbh": "#2f5fd0", "gpqa_diamond": "#c2570f", "math500": "#2a8a5e"}
C_MODEL = {"gpt-oss-20b": "#2f5fd0", "Qwen3.6-35B-A3B": "#c2570f"}
LEVEL_ALPHA = {"low": 0.42, "medium": 0.7, "high": 1.0}
LEVEL_RU = {"low": "низкий", "medium": "средний", "high": "высокий"}

by_arm = json.loads(EVENTS.read_text())["by_arm"]


def per_1k(model_key, level):
    """Среднее по трём зёрнам: событий каждого типа на тысячу токенов."""
    cells = [v for k, v in by_arm.items() if k.startswith(f"{model_key}/{level}/")]
    return {t: float(np.mean([c[f"per_1k_{t}"] for c in cells])) for t, _ in TYPES}


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.2, 4.8),
                               gridspec_kw={"width_ratios": [1.0, 1.25]})
for ax in (ax1, ax2):
    ax.grid(True, axis="y", color=GRID, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)

x = np.arange(len(TYPES))
w = 0.26
for i, (bench, d) in enumerate(BENCH.items()):
    vals = [d[t] / d["n"] for t, _ in TYPES]
    ax1.bar(x + (i - 1) * w, vals, w * 0.92, color=C_BENCH[bench], zorder=3, label=bench)
    for xi, v in zip(x + (i - 1) * w, vals):
        ax1.text(xi, v + 0.15, f"{v:.1f}", ha="center", fontsize=7.2, color=MUTED)
ax1.set_xticks(x)
ax1.set_xticklabels([ru for _, ru in TYPES], fontsize=8.5)
ax1.set_ylabel("событий на трассу", fontsize=9.5, color=MUTED)
ax1.set_title("Профиль событий по наборам вопросов", fontsize=11.5, color=INK, pad=10)
ax1.legend(frameon=False, fontsize=9, labelcolor=INK)

w2 = 0.14
pos = 0
handles = []
for mi, (model_ru, model_key) in enumerate([("gpt-oss-20b", "gptoss20"),
                                            ("Qwen3.6-35B-A3B", "qwen36_35b")]):
    for li, level in enumerate(["low", "medium", "high"]):
        vals = [per_1k(model_key, level)[t] for t, _ in TYPES]
        off = (mi * 3 + li - 2.5) * w2
        b = ax2.bar(x + off, vals, w2 * 0.9, color=C_MODEL[model_ru],
                    alpha=LEVEL_ALPHA[level], zorder=3,
                    label=f"{model_ru}, {LEVEL_RU[level]}")
        handles.append(b)
ax2.set_xticks(x)
ax2.set_xticklabels([ru for _, ru in TYPES], fontsize=8.5)
ax2.set_ylabel("событий на 1000 токенов", fontsize=9.5, color=MUTED)
ax2.set_title("Профиль событий по моделям и уровням усилия", fontsize=11.5, color=INK, pad=10)
ax2.legend(frameon=False, fontsize=8, ncol=2, labelcolor=INK)

fig.text(0.5, 0.015,
         "Слева нормировка на трассу, наборы вопросов разного объёма. Справа нормировка на длину: "
         "уровень усилия меняет длину рассуждения в разы.",
         ha="center", fontsize=8.5, color=MUTED)
fig.tight_layout(rect=[0.005, 0.055, 0.995, 1.0])
fig.savefig(OUT, dpi=170, facecolor="white")
print("сохранено:", OUT)
