# Схема четырёх единиц описания трассы для рисунка 1 отчёта.
# Запуск: python3 docs/assets/draw_fig_levels.py — перезаписывает fig_levels.png и .pdf рядом.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10.5,
                     "axes.linewidth": 0, "figure.dpi": 200})

fig = plt.figure(figsize=(13.0, 6.0))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")

X0, X1 = 15.0, 57.0           # поле схемы
XC = 62.0                      # колонка вклада
LVL = {4: 83.0, 3: 61.0, 2: 44.0, 1: 19.0}

GREY, INK = "#8a8a8a", "#1a1a1a"
C_VER, C_BACK, C_COM = "#1b9e91", "#e04b4b", "#111111"
C_INT = ["#f3c98b", "#9fc7e8", "#c9b3de", "#a8d5a2", "#f3c98b"]

def level_label(y, n, name):
    ax.text(X0 - 2.0, y, f"Уровень {n}", ha="right", va="center", fontsize=11.5, color=INK)
    ax.text(X0 - 2.0, y - 4.0, name, ha="right", va="center", fontsize=9.5, color=GREY)

def contrib(y, text):
    ax.text(XC, y, text, ha="left", va="center", fontsize=10.2, color=INK, linespacing=1.5)

# ---------- уровень 1: поток токенов ----------
level_label(LVL[1], 1, "поток токенов")
ax.add_patch(FancyBboxPatch((X0, LVL[1] - 2.8), X1 - X0, 5.6,
             boxstyle="round,pad=0.0,rounding_size=0.8",
             fc="#eef2fb", ec="#7f93c9", lw=1.0))
n_tok = 70
for i in range(n_tok):
    x = X0 + 1.1 + i * (X1 - X0 - 2.2) / (n_tok - 1)
    ax.plot([x, x], [LVL[1] - 2.0, LVL[1] + 2.0], color="#7f93c9", lw=0.7, alpha=0.85)
ax.annotate("", xy=(X1, LVL[1] - 7.6), xytext=(X0, LVL[1] - 7.6),
            arrowprops=dict(arrowstyle="->", color=GREY, lw=1.1))
ax.text((X0 + X1) / 2, LVL[1] - 10.9, "время генерации, токены",
        ha="center", va="center", fontsize=9.5, color=GREY)

# ---------- уровень 2: типизированные события, подписи в два ряда ----------
level_label(LVL[2], 2, "типизированные события")
ax.plot([X0, X1], [LVL[2], LVL[2]], color=GREY, lw=1.0)
EVENTS = [(20.0, C_VER, "проверка"), (27.0, C_BACK, "возврат"), (34.0, C_VER, "проверка"),
          (40.5, C_COM, "консолидация"), (45.5, C_VER, "проверка"), (51.0, C_BACK, "возврат"),
          (57.0, C_COM, "консолидация")]
for i, (x, c, lab) in enumerate(EVENTS):
    ax.plot([x], [LVL[2]], marker="o", ms=8.5, color=c, zorder=3)
    dy = -4.4 if i % 2 == 0 else -8.2
    ax.plot([x, x], [LVL[2] - 1.6, LVL[2] + dy + 1.6], color="#cccccc", lw=0.6, zorder=1)
    last = (i == len(EVENTS) - 1)
    ax.text(x if not last else X1, LVL[2] + dy, lab,
            ha="center" if not last else "right", va="center", fontsize=8.6, color=INK)

# ---------- уровень 3: интервалы ----------
level_label(LVL[3], 3, "интервалы")
INTERVALS = [(15.5, 23.5, "поиск"), (24.5, 31.5, "проверка"), (32.5, 40.5, "вывод"),
             (41.5, 48.5, "поиск"), (49.5, 57.0, "проверка")]
for i, (a, b, lab) in enumerate(INTERVALS):
    ax.add_patch(FancyBboxPatch((a, LVL[3] - 2.5), b - a, 5.0,
                 boxstyle="round,pad=0.0,rounding_size=0.7",
                 fc=C_INT[i], ec="none", alpha=0.92))
    ax.text((a + b) / 2, LVL[3], lab, ha="center", va="center", fontsize=9.2, color="#222222")

# ---------- уровень 4: циклы ----------
level_label(LVL[4], 4, "циклы рассуждений")
for a, b in [(15.5, 40.5), (41.5, 57.0)]:
    ax.plot([a, a], [LVL[4] - 3.6, LVL[4] - 1.0], color="#6a6a6a", lw=1.3)
    ax.plot([b, b], [LVL[4] - 3.6, LVL[4] - 1.0], color="#6a6a6a", lw=1.3)
    ax.plot([a, b], [LVL[4] - 1.0, LVL[4] - 1.0], color="#6a6a6a", lw=1.3)
    ax.plot([b], [LVL[4] - 3.6], marker="o", ms=8.0, color=C_COM, zorder=3)
    ax.text((a + b) / 2, LVL[4] + 2.4, "цикл", ha="center", va="center", fontsize=9.4, color=INK)
ax.text((X0 + X1) / 2, LVL[4] - 7.6, "цикл замыкает консолидация",
        ha="center", va="center", fontsize=8.8, color=GREY)

# ---------- колонка вклада ----------
ax.plot([XC - 2.6, XC - 2.6], [10, 90], color="#dddddd", lw=1.0)
ax.text(XC, 92.5, "вклад уровня", ha="left", va="center", fontsize=10.5, color=GREY)
contrib(LVL[4], "самые воспроизводимые границы,\nединица управления")
contrib(LVL[3], "режимы рассуждения,\nпрогноз числа оставшихся событий")
contrib(LVL[2], "дискретны и обнаружимы из текста,\nточки обновления прогноза")
contrib(LVL[1], "основной сигнал прогноза остатка")

# Пишем рядом с самим скриптом, чтобы путь не зависел от машины.
import os
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fig_levels")
fig.savefig(out + ".png", bbox_inches="tight", pad_inches=0.12, facecolor="white")
fig.savefig(out + ".pdf", bbox_inches="tight", pad_inches=0.12, facecolor="white")
print("записано:", out + ".png", "и", out + ".pdf")
