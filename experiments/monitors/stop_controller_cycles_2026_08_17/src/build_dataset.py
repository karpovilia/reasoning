"""Сборка датасета трасс с разметкой ученика для эксперимента stop_controller_cycles_2026_08_17.

Читает описи корпусов, восстанавливает сегментацию трасс той же канонической функцией,
что и нарезка окон разметки, приклеивает разметку ученика по опознавателю окна, строит
токенную ось и цели: остаток до конца текущего цикла и остаток до конца следующего.

Выход: по строке на трассу, поля перечислены в конце файла. Тексты сегментов в выход не
попадают, из них заранее считаются только длины и отметка кандидата ответа.

Запуск на боксе tesla:
  /home/user/ki_sft_judge/.venv/bin/python src/build_dataset.py \
      --root /home/user/ki_cycles --out /home/user/ki_cycles/data/ds
"""

import argparse
import hashlib
import json
import os
import re
import sys
import time
from collections import Counter

M_TYPES = ["backtrack", "verify", "branch", "subgoal_done", "failed_attempt", "commit"]
TIDX = {t: i for i, t in enumerate(M_TYPES)}
COMMIT = TIDX["commit"]
OPERATORS = ["SETUP", "DERIVING", "EXPLORING", "VERIFYING", "REVISING", "CONCLUDING"]
OPIDX = {o: i for i, o in enumerate(OPERATORS)}

MIN_CHARS = 200
MIN_SEGMENTS = 4
CHUNK_WINDOW = 30
CHUNK_CONTEXT = 15
H_HORIZON = 256.0


def cluster_root():
    """Приставка пути, с которой записаны файлы трасс в описи корпусов.

    Опись собиралась на счётной машине, и путь в ней абсолютный. В файлах
    репозитория абсолютных путей машин не держат, поэтому приставка берётся из
    переменной среды CLUSTER_ROOT либо из файла cluster_root в каталоге
    настроек, который лежит вне git. Если настройки нет, приставка пустая и путь
    из описи используется как есть.
    """
    from_env = os.environ.get("CLUSTER_ROOT", "")
    if from_env:
        return from_env
    settings = os.environ.get(
        "REASONING_DATA_SETTINGS",
        os.path.join(os.path.expanduser("~"), ".config", "reasoning-data"))
    path = os.path.join(settings, "cluster_root")
    if os.path.isfile(path):
        return open(path, encoding="utf-8").read().strip()
    return ""


CLUSTER_ROOT = cluster_root()


def local_path(recorded, tr_root):
    """Путь к файлу трасс на этой машине по пути из описи корпусов."""
    rel = recorded
    if CLUSTER_ROOT and rel.startswith(CLUSTER_ROOT):
        rel = rel[len(CLUSTER_ROOT):]
    return os.path.join(tr_root, rel.lstrip("/"))

CAND_PATTERNS = [
    re.compile(r"\\boxed\{"),
    re.compile(r"(?<!to )(?<!To )\b(?:the\s+)?(?:final\s+)?(?:answer|result)\s*(?:is\b|:|=)",
               re.IGNORECASE),
    re.compile(r"\bответ\b", re.IGNORECASE),
    re.compile(r"=\s*(?:-?\d[\d,./]*|\(?[A-E]\)?)\s*(?:[.!?]|\n|$)"),
    re.compile(r"\banswer\s+\(?([A-E])\)?(?:[.!?\s]|$)"),
]


# ---------------------------------------------------------------- сегментация
def segment(text):
    """Каноническая сегментация проекта, дословная копия из
    internal_signals_poc/detector_decentering_2026_08_12/scripts/judge_label_nativepo.py,
    строки 87-98."""
    segs = []
    for ch in re.split(r"\n+", text):
        ch = ch.strip()
        if not ch:
            continue
        for p in re.split(r"(?<=[.!?])\s+(?=[A-Z(])", ch):
            p = p.strip()
            if p:
                segs.append(p)
    return segs


def chunk_ranges(n_seg, chunk, ctx):
    """Дословная копия из
    internal_signals_poc/qwen_judge.py, строки 108-118."""
    if n_seg <= chunk:
        return [(0, n_seg, 0)]
    out = []
    start = 0
    while start < n_seg:
        end = min(start + chunk, n_seg)
        out.append((start, end, max(0, start - ctx)))
        start = end
    return out


# ------------------------------------------------- склейка текста размышления
def stream_text(rec):
    return rec.get("reasoning_text") or rec.get("reasoning") or ""


def one_segment(text, limit=180):
    t = " ".join(str(text).split())
    t = t.replace(".", " ").replace("!", " ").replace("?", " ")
    t = " ".join(t.split())
    return t[:limit]


def tool_result_note(content):
    raw = (content or "").strip()
    try:
        obj = json.loads(raw)
    except Exception:
        obj = None
    if isinstance(obj, dict):
        if "results" in obj:
            titles = [str((r or {}).get("title", "")) for r in (obj.get("results") or [])][:4]
            note = ("ответ поиска на запрос %s, статей %d, среди них %s"
                    % (obj.get("query", ""), len(obj.get("results") or []),
                       "; ".join(t for t in titles if t)))
        elif "extract" in obj or "title" in obj:
            note = ("ответ поиска, страница %s, текста %d знаков"
                    % (obj.get("title", ""), len(str(obj.get("extract") or ""))))
        elif "stdout" in obj or "exit_code" in obj:
            note = ("ответ вычислителя, код возврата %s, вывод %s, ошибки %s"
                    % (obj.get("exit_code"), str(obj.get("stdout") or "")[:80],
                       str(obj.get("stderr") or "")[:40]))
        else:
            note = "ответ инструмента, поля %s" % ", ".join(sorted(obj))
    else:
        note = "ответ инструмента, %s" % raw[:100]
    return "[" + one_segment(note) + "]"


def tool_call_note(calls):
    names = []
    for c in calls or []:
        fn = (c or {}).get("function") or {}
        nm = fn.get("name") or (c or {}).get("name") or ""
        arg = str(fn.get("arguments") or "")[:80]
        names.append("%s %s" % (nm, arg) if arg else str(nm))
    body = "; ".join(n for n in names if n) or "без имени"
    return "[" + one_segment("обращение к инструменту: %s" % body) + "]"


def toolloop_text(rec):
    work = rec.get("raw_model_work") or {}
    msgs = work.get("messages") or []
    resps = work.get("responses") or []
    reasonings = []
    for resp in resps:
        ch = (resp.get("choices") or [{}])[0]
        m = ch.get("message") or {}
        reasonings.append(m.get("reasoning") or "")
    parts = []
    i_assistant = 0
    for m in msgs:
        role = m.get("role")
        content = m.get("content") or ""
        if role == "assistant":
            think = ""
            if i_assistant < len(reasonings):
                think = reasonings[i_assistant] or ""
            if not think.strip():
                think = content
            i_assistant += 1
            if think.strip():
                parts.append(think.strip())
            calls = m.get("tool_calls") or []
            if calls:
                parts.append(tool_call_note(calls))
        elif role == "tool":
            parts.append(tool_result_note(content))
    return "\n\n".join(p for p in parts if p.strip())


def token_total_of(rec, kind, n_chars):
    """Полное число токенов размышления.

    Поток: поле reasoning_tokens записи трассы. Цикл с инструментом: сумма поля
    completion_tokens по ответам движка; там это число включает и текст ответа, а не
    только размышление, поэтому ось вторичного корпуса чуть растянута, и это оговорено в
    предрегистрации. Запасной путь для обоих устройств — число символов делить на четыре.
    """
    if kind == "stream":
        for k in ("reasoning_tokens", "natural_reasoning_tokens"):
            v = rec.get(k)
            if isinstance(v, (int, float)) and v > 0:
                return float(v), k
    else:
        s = 0.0
        for resp in ((rec.get("raw_model_work") or {}).get("responses") or []):
            u = resp.get("usage") or {}
            v = u.get("completion_tokens")
            if isinstance(v, (int, float)):
                s += float(v)
        if s > 0:
            return s, "completion_tokens"
    return max(1.0, n_chars / 4.0), "chars_div4"


def token_axis(seg_texts, token_total):
    lens = [max(len(s), 1) for s in seg_texts]
    total_chars = float(sum(lens))
    tt = float(token_total or 0)
    if tt <= 0:
        tt = total_chars / 4.0
    ends, acc = [], 0.0
    for L in lens:
        acc += L
        ends.append(acc / total_chars * tt)
    return ends, tt


def candidate_seg(seg_texts):
    """Номер сегмента, в котором впервые появляется кандидат ответа."""
    text = "\n".join(seg_texts)
    best = None
    for rx in CAND_PATTERNS:
        m = rx.search(text)
        if m and (best is None or m.start() < best):
            best = m.start()
    if best is None:
        return -1
    pos, cp = 0, int(best)
    for i, s in enumerate(seg_texts):
        pos += len(s) + 1
        if pos > cp:
            return i
    return len(seg_texts) - 1


def split_of_question(qid, salt="sft_judge_distill", frac_test=0.30, frac_val=0.15):
    """Каноническое разбиение по вопросу, дословно из
    internal_signals_poc/cycle_end_lora_2026_08_16/src/build_dataset.py,
    строки 58-65."""
    h = hashlib.sha1((salt + "::" + str(qid)).encode("utf-8")).hexdigest()
    u = int(h[:12], 16) / float(1 << 48)
    if u < frac_test:
        return "test"
    if u < frac_test + frac_val:
        return "val"
    return "train"


FAMILY = [
    ("openai/gpt-oss-20b", "gpt_oss_20b"),
    ("Qwen/Qwen3.6-35B-A3B-FP8", "qwen36_35b"),
    ("Qwen/Qwen3-32B", "qwen3_32b"),
    ("Qwen/Qwen3-30B-A3B", "qwen3_30b"),
]


def family_of(model):
    m = str(model or "")
    for pref, name in FAMILY:
        if pref.lower() in m.lower():
            return name
    ml = m.lower()
    if "gpt-oss" in ml:
        return "gpt_oss_20b"
    if "qwen3.6" in ml:
        return "qwen36_35b"
    if "qwen3-32b" in ml:
        return "qwen3_32b"
    if "qwen3-30b" in ml:
        return "qwen3_30b"
    return "other"


def load_student(paths):
    """Опознаватель окна -> запись разметки."""
    idx = {}
    bad = 0
    for p in paths:
        with open(p, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                idx[d["uid"]] = d
        print("прочитана разметка %s, всего окон %d" % (p, len(idx)), flush=True)
    return idx, bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="корень рабочего каталога на боксе")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    lab_dir = os.path.join(a.root, "data", "labels")
    tr_root = os.path.join(a.root, "data", "traces")
    idx, _ = load_student([os.path.join(lab_dir, "pred_student_wave1.jsonl"),
                           os.path.join(lab_dir, "pred_student_wave2.jsonl")])

    manifests = [("wave1", os.path.join(lab_dir, "corpora_wave1.json")),
                 ("wave2", os.path.join(lab_dir, "corpora_wave2.json"))]
    fh = {s: open(os.path.join(a.out, "traces_%s.jsonl" % s), "w", encoding="utf-8")
          for s in ("train", "val", "test")}
    stats = {"по волнам": {}, "по семействам": Counter(), "по частям": Counter(),
             "по корпусам": Counter(), "источник токенов": Counter(),
             "отсев": Counter(), "трасс": 0, "сегментов": 0, "событий": Counter(),
             "точек": 0, "uid не найден": 0, "uid найден": 0}
    t0 = time.time()

    for wave, mpath in manifests:
        entries = json.load(open(mpath, encoding="utf-8"))
        w_tr = 0
        for e in entries:
            path = local_path(e["path"], tr_root)
            tag, kind = e["tag"], e.get("kind", "stream")
            if not os.path.exists(path):
                stats["отсев"]["нет файла %s" % tag] += 1
                continue
            n_tr = 0
            for line in open(path, encoding="utf-8"):
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                txt = stream_text(r) if kind == "stream" else toolloop_text(r)
                if len(txt.strip()) < MIN_CHARS:
                    stats["отсев"]["короткая"] += 1
                    continue
                segs = segment(txt)
                n_seg = len(segs)
                if n_seg < MIN_SEGMENTS:
                    stats["отсев"]["мало сегментов"] += 1
                    continue
                rule = e.get("arm_rule")
                if rule == "level_ersatz":
                    arm_name = "level_%s_ersatz%s" % (r.get("effort_level"), r.get("cap"))
                else:
                    arm_name = e.get("arm") or r.get("arm") or tag
                bench = r.get("benchmark") or e.get("benchmark") or "math500"
                qid = r["question_id"]
                thash = hashlib.sha1(("\n".join(segs)).encode("utf-8")).hexdigest()[:8]
                uid_arm = "%s|" % arm_name if rule else ""
                chunks = chunk_ranges(n_seg, CHUNK_WINDOW, CHUNK_CONTEXT)

                events, spans, missing, broken = [], [], 0, 0
                for ci, (lo, hi, _clo) in enumerate(chunks):
                    uid = "%s|%s%s|%s|%d" % (tag, uid_arm, qid, thash, ci)
                    d = idx.get(uid)
                    if d is None:
                        missing += 1
                        continue
                    try:
                        obj = json.loads(d["raw"])
                    except Exception:
                        broken += 1
                        continue
                    for ev in (obj.get("events") or []):
                        ty = ev.get("type")
                        try:
                            sid = int(ev.get("seg_id"))
                        except Exception:
                            continue
                        if ty in TIDX and lo <= sid < hi:
                            events.append((sid, TIDX[ty]))
                    for sp in (obj.get("spans") or []):
                        try:
                            s0, s1 = int(sp["seg_start"]), int(sp["seg_end"])
                        except Exception:
                            continue
                        op = OPIDX.get(sp.get("operator"), -1)
                        spans.append((max(s0, lo), min(s1, hi), op))
                stats["uid найден"] += len(chunks) - missing
                stats["uid не найден"] += missing
                if missing or broken:
                    stats["отсев"]["неполная разметка"] += 1
                    continue

                n_chars = sum(len(s) for s in segs)
                tt, tt_src = token_total_of(r, kind, n_chars)
                stats["источник токенов"][tt_src] += 1
                ends, T = token_axis(segs, tt)
                seg_tok = [ends[0]] + [ends[i] - ends[i - 1] for i in range(1, n_seg)]
                ops = [-1] * n_seg
                for (s0, s1, op) in spans:
                    for i in range(max(0, s0), min(n_seg, s1)):
                        ops[i] = op
                events.sort()
                sp = split_of_question(qid)
                fam = family_of(e.get("trace_model") or r.get("model"))
                rec = {
                    "trace_uid": "%s|%s%s|%s" % (tag, uid_arm, qid, thash),
                    "qid": qid, "split": sp, "wave": wave, "kind": kind,
                    "corpus": e["corpus"], "tag": tag, "arm": arm_name,
                    "benchmark": bench, "family": fam,
                    "model": e.get("trace_model") or r.get("model"),
                    "effort": r.get("effort_level") or r.get("effort") or "",
                    "cap": r.get("cap"), "n_seg": n_seg, "T": T,
                    "token_total_src": tt_src,
                    "seg_end": [round(x, 3) for x in ends],
                    "seg_tok": [round(x, 3) for x in seg_tok],
                    "seg_op": ops,
                    "cand_seg": candidate_seg(segs),
                    "events": [[int(s), int(k)] for (s, k) in events],
                }
                fh[sp].write(json.dumps(rec, ensure_ascii=False) + "\n")
                stats["трасс"] += 1
                stats["сегментов"] += n_seg
                stats["точек"] += max(0, n_seg - 2)
                stats["по семействам"][fam] += 1
                stats["по частям"][sp] += 1
                stats["по корпусам"][e["corpus"]] += 1
                for (_s, k) in events:
                    stats["событий"][M_TYPES[k]] += 1
                n_tr += 1
                w_tr += 1
            print("плечо %-44s трасс %5d" % (tag, n_tr), flush=True)
        stats["по волнам"][wave] = w_tr

    for f in fh.values():
        f.close()
    stats["секунд"] = round(time.time() - t0, 1)
    for k in ("по семействам", "по частям", "по корпусам", "событий", "отсев",
              "источник токенов"):
        stats[k] = dict(stats[k])
    with open(os.path.join(a.out, "ds_stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=1)
    print(json.dumps(stats, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
