#!/usr/bin/env python3
"""Кривые потери двух прогонов дообучения ученика-разметчика.

Слева потеря на обучающих окнах по шагам оптимизатора, справа потеря на
отложенном срезе. Ось потери логарифмическая в обеих панелях: значения лежат
в узкой полосе от 0.23 до 0.55, и на линейной оси поздняя часть кривой
сливается в горизонталь.

Источники чисел:
  1e-4: /home/ki/repos/reasoning/internal_signals_poc/lora_judge_2026_08_15/outputs/train_log.jsonl
  5e-5: /home/ki/repos/reasoning/internal_signals_poc/lora_judge_full_2026_08_18/outputs/train/train_log.jsonl
  срез: /home/ki/repos/reasoning/internal_signals_poc/lora_judge_full_2026_08_18/outputs/val_loss_curve.json
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FixedFormatter

ROOT = Path("/home/ki/repos/reasoning/internal_signals_poc")
OUT = Path("/home/ki/repos/reasoning_clean/docs/assets/fig_student_loss.png")

C_FAST = "#c2570f"   # темп 1e-4
C_SLOW = "#2f5fd0"   # темп 5e-5
INK = "#22252a"
MUTED = "#6b7280"
GRID = "#dfe3e8"


def read_log(path):
    steps, loss = [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            steps.append(r["step"])
            loss.append(r["loss"])
    return steps, loss


s_fast, l_fast = read_log(ROOT / "lora_judge_2026_08_15/outputs/train_log.jsonl")
s_slow, l_slow = read_log(ROOT / "lora_judge_full_2026_08_18/outputs/train/train_log.jsonl")

val = json.loads((ROOT / "lora_judge_full_2026_08_18/outputs/val_loss_curve.json").read_text())
v_fast = [(r["шаг"], r["потеря на срезе"]) for r in val["прежний прогон темп 1e-4"]]
v_slow = [(r["шаг"], r["потеря на срезе"]) for r in val["новый прогон темп 5e-5"]]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.6, 4.5))

for ax in (ax1, ax2):
    ax.set_yscale("log")
    ax.grid(True, which="both", color=GRID, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)

# --- панель 1: обучающая потеря -------------------------------------------
ax1.plot(s_fast, l_fast, color=C_FAST, linewidth=2.0, zorder=3,
         label="темп 1e-4, снят по времени")
ax1.plot(s_slow, l_slow, color=C_SLOW, linewidth=2.0, zorder=3,
         label="темп 5e-5, полный проход")
ax1.axvline(973, color=C_FAST, linewidth=1.0, linestyle=":", zorder=2)
ax1.text(968, 0.2335, "останов по времени, 973  ", color=C_FAST, fontsize=8.5,
         va="bottom", ha="right", rotation=90)
ax1.axvline(s_slow[-1], color=C_SLOW, linewidth=1.0, linestyle=":", zorder=2)
ax1.text(s_slow[-1] - 5, 0.2335, f"досчитано на 10:20 МСК, {s_slow[-1]}  ",
         color=C_SLOW, fontsize=8.5, va="bottom", ha="right", rotation=90)
ax1.set_title("Потеря на обучающих окнах", fontsize=11.5, color=INK, pad=10)
ax1.set_xlabel("шаг оптимизатора", fontsize=9.5, color=MUTED)
ax1.set_ylabel("потеря, логарифмическая шкала", fontsize=9.5, color=MUTED)
ax1.set_xlim(0, 2120)

# --- панель 2: потеря на срезе --------------------------------------------
ax2.plot([p[0] for p in v_fast], [p[1] for p in v_fast], color=C_FAST,
         linewidth=2.0, marker="o", markersize=6, zorder=3, label="темп 1e-4")
ax2.plot([p[0] for p in v_slow], [p[1] for p in v_slow], color=C_SLOW,
         linewidth=2.0, marker="o", markersize=8, zorder=3, label="темп 5e-5, полный проход")
for st, lo in v_fast:
    ax2.annotate(f"{lo:.4f}", (st, lo), textcoords="offset points",
                 xytext=(0, 9), ha="center", fontsize=8, color=MUTED)
for st, lo in v_slow:
    ax2.annotate(f"{lo:.4f}", (st, lo), textcoords="offset points",
                 xytext=(0, -17), ha="center", fontsize=8.5, color=C_SLOW)
ax2.set_title("Потеря на отложенном срезе", fontsize=11.5, color=INK, pad=10)
ax2.set_xlabel("шаг оптимизатора", fontsize=9.5, color=MUTED)
ax2.set_xlim(60, 2120)

ticks = [0.23, 0.25, 0.28, 0.32, 0.36, 0.42, 0.48, 0.55]
for ax in (ax1, ax2):
    ax.yaxis.set_major_locator(FixedLocator(ticks))
    ax.yaxis.set_minor_locator(FixedLocator([]))
    ax.yaxis.set_major_formatter(FixedFormatter([f"{t:g}" for t in ticks]))
ax1.set_ylim(0.225, 0.56)
ax2.set_ylim(0.2295, 0.261)
ax2.yaxis.set_major_locator(FixedLocator([0.2326, 0.2356, 0.238, 0.244, 0.249, 0.256]))
ax2.yaxis.set_major_formatter(FixedFormatter(["0.2326", "0.2356", "0.238", "0.244", "0.249", "0.256"]))

ax1.legend(frameon=False, fontsize=9.5, loc="upper center", labelcolor=INK)
ax2.legend(frameon=False, fontsize=9.5, loc="upper right", labelcolor=INK)

fig.suptitle("Дообучение ученика-разметчика: два темпа обучения при прочих равных",
             fontsize=13, color=INK, y=0.99)
fig.text(0.5, 0.005,
         "Ранг 32, множитель 64, отсев 0.05, зерно 20260815, две карты RTX A6000. "
         "Различаются только темп обучения и длина прогона.",
         ha="center", fontsize=8.5, color=MUTED)
fig.tight_layout(rect=[0, 0.03, 1, 0.955])
fig.savefig(OUT, dpi=170, facecolor="white")
print("сохранено:", OUT)
print("точек обучающей кривой:", len(s_fast), "и", len(s_slow))
print("последний шаг:", s_fast[-1], "и", s_slow[-1])
