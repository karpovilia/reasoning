#!/usr/bin/env python3
"""
Общие утилиты арма GRPO-бюджет (сессия 09, AAAI-27 горка контроля).

Содержит: пути (локально/HPC), резолюцию harmony-токенов gpt-oss,
сборку промпта (реплика системного промпта gpt_oss_runner), парсер
completion'а по токен-ids (каналы analysis/final, длины, извлечение
concise_answer) и импорт канонического грейдера is_correct.

Бюджет = число контент-токенов reasoning-каналов (не final) — реальные
декод-токены, как требует docs/session_prompts/2026_07_06/15_shared_eval_protocol.md.
"""
from __future__ import annotations

import importlib.util
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

# Переключение модели семейства без правки кода: env GPTOSS_MODEL_ID
# (тот же harmony-формат, токенайзер и цели LoRA у 20b и 120b).
MODEL_ID = os.environ.get("GPTOSS_MODEL_ID", "openai/gpt-oss-20b")
SEED = 42

# ---------------------------------------------------------------------------






def grading_module():
    from . import grader
    return grader


# ---------------------------------------------------------------------------
# Промпт: дословная реплика системного промпта gpt_oss_runner (не multi-hop),
# reasoning_trajectories/gpt_oss_runner/src/gpt_oss_runner/runner.py:260-271.

FINAL_ANSWER_EXAMPLE = (
    '{"concise_answer": "B", "comment": "The answer to this question is B) '
    'First mass is bigger than second mass"}'
)

SYSTEM_PROMPT = (
    "You are solving a benchmark question. "
    "Answer the question directly and keep the final answer concise. "
    "When you are ready to answer, respond with EXACTLY one JSON object in this format: "
    '{"concise_answer": "<short final answer (str)>", "comment": "<optional brief explanation (str) or null>"}'
    f" Example: {FINAL_ANSWER_EXAMPLE}"
)


def budget_instruction(target_budget: int) -> str:
    """Строка целевого бюджета для L2 (length-conditioned). Модель через RL
    учится связывать это число с реальной длиной reasoning."""
    return (f" Aim to keep your reasoning to about {target_budget} tokens "
            f"before you give the final answer.")


def build_prompt_text(tok, question: str, system: str = SYSTEM_PROMPT,
                      target_budget: int | None = None,
                      reasoning_effort: str | None = None) -> str:
    """Harmony-префикс как текст. target_budget!=None -> L2: в системный
    промпт добавляется целевой бюджет reasoning (одно число двигает точку
    на фронте на инференсе). reasoning_effort управляет строкой
    "Reasoning: <effort>" шаблона gpt-oss; None сохраняет прежнее поведение
    (шаблон подставляет medium), поэтому все старые вызовы побайтово
    воспроизводятся."""
    sys = system if target_budget is None else system + budget_instruction(target_budget)
    msgs = [{"role": "system", "content": sys},
            {"role": "user", "content": question}]
    kw = {} if reasoning_effort is None else {"reasoning_effort": reasoning_effort}
    return tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, **kw)


# ---------------------------------------------------------------------------
# Harmony-токены (резолвим из токенайзера, никогда не хардкодим —
# как в reasoning_budget/iv_wait_2026_07_02/generate_iv.py::Special).


class Special:
    def __init__(self, tok):
        def tid(s):
            i = tok.convert_tokens_to_ids(s)
            if i is None or i < 0:
                raise RuntimeError(f"special token {s!r} not in vocab")
            return i
        self.start = tid("<|start|>")
        self.end = tid("<|end|>")
        self.message = tid("<|message|>")
        self.channel = tid("<|channel|>")
        self.ret = tid("<|return|>")


# Ловит "concise_answer": "..." даже когда LaTeX-бэкслеши ломают строгий JSON
# (та же регулярка, что в cleaning_scripts/clean_gpt_oss_20b.py).
_CONCISE_ANSWER_RE = re.compile(r'"concise_answer"\s*:\s*"((?:[^"\\]|\\.)*)"')


def extract_concise_answer(final_text: str) -> tuple[str | None, str]:
    """(answer, source): строгий JSON -> regex-fallback -> None."""
    text = final_text.strip()
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict) and "concise_answer" in parsed:
            return str(parsed["concise_answer"]), "json_parsed"
    except (json.JSONDecodeError, ValueError):
        pass
    m = _CONCISE_ANSWER_RE.search(text)
    if m:
        raw = m.group(1).replace('\\"', '"').replace("\\\\", "\\")
        return raw, "json_parsed_regex_fallback"
    return None, "missing"


@dataclass
class ParsedCompletion:
    reasoning_tokens: int = 0          # контент-токены всех не-final сообщений (бюджет)
    final_tokens: int = 0              # контент-токены final-сообщения
    total_tokens: int = 0              # длина completion в токенах
    n_messages: int = 0
    has_final: bool = False
    hit_length: bool = False           # оборван по max_completion_length (нет <|return|>)
    final_text: str = ""
    answer: str | None = None
    answer_source: str = "missing"
    channels: list = field(default_factory=list)
    reasoning_text: str = ""           # декодированный текст reasoning (только при decode_reasoning=True)


def parse_completion(ids: list[int], tok, sp: Special,
                     decode_reasoning: bool = False) -> ParsedCompletion:
    """Разбор harmony-продолжения после '<|start|>assistant'.

    Ожидаемая структура: <|channel|>analysis<|message|>...<|end|>
    [<|start|>assistant<|channel|>...<|end|>]*
    <|start|>assistant<|channel|>final<|message|>...<|return|>
    Генерация могла быть оборвана по длине в любом месте.
    """
    out = ParsedCompletion(total_tokens=len(ids))
    n = len(ids)
    i = 0
    saw_return = False
    while i < n:
        # заголовок сообщения: всё до <|message|>
        chan_name = ""
        j = i
        header_end = None
        while j < n:
            if ids[j] == sp.message:
                header_end = j
                break
            j += 1
        if header_end is None:
            # незавершённый заголовок (оборвано по длине) — токены не считаем контентом
            break
        # имя канала: токены между <|channel|> и <|message|>
        if sp.channel in ids[i:header_end]:
            k = ids.index(sp.channel, i, header_end)
            chan_name = tok.decode(ids[k + 1:header_end]).strip()
        # контент: от <|message|>+1 до <|end|>/<|return|>/следующего <|start|>
        c0 = header_end + 1
        c1 = c0
        terminated = False
        while c1 < n:
            t = ids[c1]
            if t in (sp.end, sp.ret, sp.start):
                terminated = True
                if t == sp.ret:
                    saw_return = True
                break
            c1 += 1
        content_len = c1 - c0
        out.n_messages += 1
        out.channels.append(chan_name or "?")
        if chan_name == "final":
            out.has_final = True
            out.final_tokens += content_len
            out.final_text = tok.decode(ids[c0:c1])
        else:
            out.reasoning_tokens += content_len
            if decode_reasoning and content_len > 0:
                # склеиваем reasoning-сегменты через \n — событийный скорер
                # сам сегментирует текст, разделитель лишь отбивает сообщения
                seg = tok.decode(ids[c0:c1])
                out.reasoning_text = seg if not out.reasoning_text else out.reasoning_text + "\n" + seg
        if not terminated:            # контент оборван по длине
            out.hit_length = True
            break
        # следующий сегмент: пропускаем терминатор; если это <|start|> — он начало заголовка
        i = c1 if ids[c1] == sp.start else c1 + 1
        if saw_return:
            break
    if not out.has_final:
        out.hit_length = out.hit_length or not saw_return
    if out.has_final:
        out.answer, out.answer_source = extract_concise_answer(out.final_text)
    return out
