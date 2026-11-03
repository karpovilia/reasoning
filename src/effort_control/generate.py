#!/usr/bin/env python3
"""Generate matched gpt-oss branches with the frozen effort policy."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import signal
import sys
import time
from pathlib import Path

import numpy as np

ARM = Path("outputs/effort")


from .harmony import (MODEL_ID, SEED, Special, build_prompt_text,  # noqa: E402
                    extract_concise_answer, grading_module, parse_completion)
from . import gate_common_v3 as GC  # noqa: E402

REVISION = "6cee5e81ee83917806bbde320786a8fb61efebee"
COMMIT = GC.T_COMMIT


class _GradeTimeout(Exception):
    pass


def _alarm(signum, frame):
    raise _GradeTimeout()


def safe_is_correct(is_correct, gold, answer, bench, timeout_s=5.0):
    old = signal.signal(signal.SIGALRM, _alarm)
    signal.setitimer(signal.ITIMER_REAL, timeout_s)
    try:
        return bool(is_correct(gold, answer, bench)), False
    except _GradeTimeout:
        return False, True
    except Exception:
        return False, False
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)


def force_finalize_ids(tok, sp: Special) -> list[int]:
    """<|end|><|start|>assistant<|channel|>final<|message|> — актуатор довершения."""
    return ([sp.end, sp.start] + tok.encode("assistant", add_special_tokens=False)
            + [sp.channel] + tok.encode("final", add_special_tokens=False)
            + [sp.message])


def seg_end_tokens(segs, tok_now: float) -> list[float]:
    lens = [max(len(s), 1) for s in segs]
    tot = float(sum(lens)) or 1.0
    out, acc = [], 0.0
    for L in lens:
        acc += L
        out.append(float(tok_now) * acc / tot)
    return out


class Req:
    """Один вопрос под одним условием."""

    __slots__ = ("qid", "row", "cond", "budget", "policy", "q", "version",
                 "prompts", "effort", "start_effort", "start_info",
                 "start_effort_no_diff", "effort_switch_tok", "n_switch",
                 "gen", "content_start", "spend", "panel",
                 "deadline", "mode", "terminal", "closed", "close_rule",
                 "decisions", "last_dec_spend", "last_dec_commits",
                 "n_rounds", "finish_reason", "forced", "text",
                 "stop_state", "mult")

    def __init__(self, qid, row, cond, budget, policy, q, version, prompts,
                 panel, W, effort, start_info, start_effort_no_diff, mult):
        self.qid, self.row, self.cond = qid, row, cond
        self.budget, self.policy, self.q = budget, policy, q
        self.version = version
        self.prompts = prompts            # {"normal": ids, "low": ids}
        self.effort = effort
        self.start_effort = effort
        self.start_info = start_info
        self.start_effort_no_diff = start_effort_no_diff
        self.mult = mult
        self.effort_switch_tok = None
        self.n_switch = 0
        self.gen: list[int] = []
        self.content_start = None
        self.spend = 0
        self.panel = panel
        self.deadline = float(budget) - W
        self.mode = "normal"
        self.terminal = False
        self.closed = None
        self.close_rule = ""
        self.decisions: list[dict] = []
        self.last_dec_spend = 0.0
        self.last_dec_commits = 0
        self.n_rounds = 0
        self.finish_reason = ""
        self.forced = False
        self.text = ""
        self.stop_state = {}

    def full(self) -> list[int]:
        return list(self.prompts[self.effort]) + list(self.gen)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", required=True)
    ap.add_argument("--out-dir", default=str(ARM / "outputs"))
    ap.add_argument("--pack-dir", default=str(ARM / "packs"))
    ap.add_argument("--fc-dir", default=str(ARM / "data"))
    ap.add_argument("--student-ckpt", default=str(ARM / "packs" / "light_pooled.pt"))
    ap.add_argument("--prereg", default=str(ARM / "PREREGISTRATION.json"))
    ap.add_argument("--tag", default="run")
    ap.add_argument("--budgets", default="200,287,411,589,845,1211,1736,2489")
    ap.add_argument("--cap-extra", default="140")
    ap.add_argument("--qs", default="0.3,0.5")
    ap.add_argument("--policy-versions", default="v2.1,v3")
    ap.add_argument("--low-cap-arm", type=int, default=1,
                    help="считать ли плечо фиксированного предела при низком "
                         "уровне усилия: без него фронт конфигураций не "
                         "содержит ручки усилия и сравнение нечестно")
    ap.add_argument("--chunk", type=int, default=128)
    ap.add_argument("--tick", type=int, default=256)
    ap.add_argument("--W", type=float, default=55.0)
    ap.add_argument("--answer-allowance", type=int, default=384)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--gpu-mem-util", type=float, default=0.85)
    ap.add_argument("--prefix-caching", type=int, default=1)
    ap.add_argument("--qshard", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--conditions", default="")
    ap.add_argument("--device-heads", default="cuda")
    ap.add_argument("--model-tag", default="gptoss20")
    ap.add_argument("--grade-timeout", type=float, default=5.0)
    ap.add_argument("--effort-check", type=int, default=0,
                    help="сколько вопросов пустить на проверку смены заголовка "
                         "усилия перед основным циклом; 0 — не проверять")
    ap.add_argument("--effort-check-only", type=int, default=0)
    args = ap.parse_args()

    t_start = time.time()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    prereg = json.load(open(args.prereg, encoding="utf-8"))
    P3 = dict(GC.V3_DEFAULTS)
    P3.update(prereg["policy_v3"]["params"])
    print("[политика v3] параметры из предрегистрации: %s"
          % json.dumps(P3, ensure_ascii=False), flush=True)

    from . import heads as HEADS  # noqa: E402

    from transformers import AutoTokenizer
    import torch

    print("[головы] загрузка ученика судьи %s на %s"
          % (args.student_ckpt, args.device_heads), flush=True)
    student = HEADS.StudentHead(args.student_ckpt, device=args.device_heads,
                                model_tag=args.model_tag)
    print("[головы] условие ученика: %s" % student.tag_used, flush=True)

    forecaster = GC.ForestForecaster.load(args.fc_dir)
    vtab = GC.VerifyTable(os.path.join(args.fc_dir, "cverify_table.json"))
    diff_head = GC.DifficultyHead(os.path.join(args.fc_dir, "difficulty_head.json"))
    assert list(forecaster.features) == list(GC.FEATURE_ORDER), \
        "порядок признаков прогноза разошёлся с живым сборщиком"
    model_key = "openai/gpt-oss-20b"
    model_idx = forecaster.models.index(model_key)
    arm_idx = {e: forecaster.arms.index(GC.ARM_OF_EFFORT[e]) for e in GC.EFFORTS}
    print("[прогноз] признаков %d, квантили %s, код модели %d, коды плеч %s, "
          "ключ усилия починен: %s"
          % (len(forecaster.features), forecaster.qs, model_idx, arm_idx,
             forecaster.meta.get("effort_key_fixed")), flush=True)
    print("[трудность] Спирмен на отложенных вопросах обучающего корпуса %s"
          % diff_head.meta.get("spearman_holdout"), flush=True)

    proto = HEADS.Panel(pack_dir=args.pack_dir, events_backend="regex",
                        model=model_key, model_tag=args.model_tag)
    print("[панель] модель пакета интенсивностей %s, модель таблицы цикла %s, "
          "пороги ученика %s" % (proto.model, proto.table_model,
                                 [round(float(t), 4) for t in proto.stream.thr]),
          flush=True)

    def make_panel():
        p = object.__new__(HEADS.Panel)
        p.hawkes_pack = proto.hawkes_pack
        p.cycle = proto.cycle
        p.model = proto.model
        p.table_model = proto.table_model
        p.hawkes = HEADS.HawkesHead.from_pack(proto.hawkes_pack, proto.model)
        p.stream = HEADS.EventStream(backend="regex",
                                     thresholds=proto.stream.thr,
                                     model_tag=args.model_tag, device="cpu")
        p.stream.backend = "student"
        p.stream.student = student
        p.reset()
        return p

    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    import vllm as _vllm

    tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION,
                                        trust_remote_code=True)
    sp = Special(tok)
    is_correct = grading_module().is_correct

    rows = [json.loads(l) for l in open(args.questions, encoding="utf-8")
            if l.strip()]
    if args.limit:
        rows = rows[:args.limit]
    budgets = [int(b) for b in args.budgets.split(",") if b]
    qlist = [float(x) for x in args.qs.split(",") if x]
    cap_extra = [int(b) for b in args.cap_extra.split(",") if b.strip()]
    versions = [v.strip() for v in args.policy_versions.split(",") if v.strip()]
    vtag = {v: v.replace(".", "") for v in versions}

    conds = []
    for B in sorted(set(budgets + cap_extra)):
        conds.append(("cap_%d" % B, B, "cap", None, None, "normal"))
        if B not in budgets:
            continue
        if args.low_cap_arm:
            conds.append(("capl_%d" % B, B, "cap", None, None, "low"))
        for v in versions:
            for q in qlist:
                conds.append(("%s_q%02d_%d" % (vtag[v], round(q * 100), B),
                              B, "gate", q, v, None))
    if args.conditions:
        want = set(c for c in args.conditions.split(",") if c)
        conds = [c for c in conds if c[0] in want]
    print("[план] вопросов %d, условий %d: %s"
          % (len(rows), len(conds), [c[0] for c in conds]), flush=True)

    # два заголовка запроса: обычный уровень усилия (тот же байт в байт, что у
    # прогонов v2, потому что reasoning_effort=None) и низкий
    txt_default = build_prompt_text(tok, rows[0]["question"])
    txt_medium = build_prompt_text(tok, rows[0]["question"], reasoning_effort="medium")
    print("[заголовок] умолчание совпадает с явным medium: %s"
          % (txt_default == txt_medium), flush=True)
    prompts = {}
    for r in rows:
        pn = tok(build_prompt_text(tok, r["question"]),
                 add_special_tokens=False)["input_ids"]
        pl = tok(build_prompt_text(tok, r["question"], reasoning_effort="low"),
                 add_special_tokens=False)["input_ids"]
        prompts[r["question_id"]] = {"normal": pn, "low": pl}
    longest = max(max(len(v["normal"]), len(v["low"]))
                  for v in prompts.values())
    need = longest + max(budgets + cap_extra) + args.answer_allowance + 64
    assert need <= args.max_model_len, "нужен контекст %d" % need

    llm = LLM(model=MODEL_ID, revision=REVISION, seed=SEED,
              max_model_len=args.max_model_len,
              gpu_memory_utilization=args.gpu_mem_util,
              enable_prefix_caching=bool(args.prefix_caching))

    def gen_batch(reqs, max_tokens_fn, stop_ids):
        params = [SamplingParams(temperature=0.0, seed=SEED,
                                 max_tokens=max(1, int(max_tokens_fn(r))),
                                 stop_token_ids=list(stop_ids)) for r in reqs]
        inputs = [TokensPrompt(prompt_token_ids=r.full()) for r in reqs]
        outs = llm.generate(inputs, params)
        res = []
        for o in outs:
            c = o.outputs[0]
            ids = list(c.token_ids)
            while ids and ids[-1] in stop_ids:
                ids.pop()
            res.append((ids, c.finish_reason))
        return res

    def content_start_of(gen):
        try:
            return gen.index(sp.message) + 1
        except ValueError:
            return None

    def first_channel(gen):
        try:
            j = gen.index(sp.message)
            k = gen.index(sp.channel)
        except ValueError:
            return ""
        return tok.decode(gen[k + 1:j]).strip() if k < j else ""

    # ── проверка корректности смены заголовка усилия ──────────────────────
    if args.effort_check:
        check_rows = rows[:args.effort_check]
        HALF = 128
        seqs = []
        params = [SamplingParams(temperature=0.0, seed=SEED, max_tokens=2 * HALF,
                                 stop_token_ids=[sp.end, sp.ret])
                  for _ in check_rows]
        outs = llm.generate([TokensPrompt(prompt_token_ids=prompts[r["question_id"]]["normal"])
                             for r in check_rows], params)
        for r, o in zip(check_rows, outs):
            ids = list(o.outputs[0].token_ids)
            while ids and ids[-1] in (sp.end, sp.ret):
                ids.pop()
            seqs.append((r, ids))
        # логарифмы вероятностей ОДНОГО И ТОГО ЖЕ продолжения под двумя
        # заголовками. Переиспользование префикса и запрос вероятностей на
        # префикс в одном движке уживаются не всегда, поэтому блок обёрнут:
        # если движок откажется, проверка обойдётся продолжениями
        lp = {"normal": [float("nan")] * len(seqs),
              "low": [float("nan")] * len(seqs)}
        lp_error = ""
        try:
            for eff in ("normal", "low"):
                pl = [SamplingParams(temperature=0.0, seed=SEED, max_tokens=1,
                                     prompt_logprobs=0) for _ in seqs]
                ins = [TokensPrompt(prompt_token_ids=list(prompts[r["question_id"]][eff]) + ids)
                       for (r, ids) in seqs]
                outs = llm.generate(ins, pl)
                vals = []
                for (r, ids), o in zip(seqs, outs):
                    plp = o.prompt_logprobs or []
                    n_tail = max(len(ids) - HALF, 0)
                    tail = plp[len(plp) - n_tail:] if n_tail else []
                    v = []
                    for d in tail:
                        if not d:
                            continue
                        v.append(float(list(d.values())[0].logprob))
                    vals.append(float(np.mean(v)) if v else float("nan"))
                lp[eff] = vals
        except Exception as exc:                      # noqa: BLE001
            lp_error = "%s: %s" % (type(exc).__name__, str(exc)[:300])
            print("[смена усилия] вероятности на префиксе недоступны: %s"
                  % lp_error, flush=True)
        # продолжение из середины под двумя заголовками
        cont = {}
        for eff in ("normal", "low"):
            ins, prm = [], []
            for (r, ids) in seqs:
                cs = content_start_of(ids)
                cut = (cs or 0) + HALF
                ins.append(TokensPrompt(prompt_token_ids=list(prompts[r["question_id"]][eff])
                                        + ids[:cut]))
                prm.append(SamplingParams(temperature=0.0, seed=SEED,
                                          max_tokens=HALF,
                                          stop_token_ids=[sp.end, sp.ret]))
            outs = llm.generate(ins, prm)
            cont[eff] = [(tok.decode(list(o.outputs[0].token_ids)),
                          o.outputs[0].finish_reason) for o in outs]
        # довершение ответа после продолжения со сменой заголовка
        fin = {}
        for eff in ("normal", "low"):
            ins, prm = [], []
            for (r, ids), (txt, _fr) in zip(seqs, cont[eff]):
                cs = content_start_of(ids)
                cut = (cs or 0) + HALF
                body = list(prompts[r["question_id"]][eff]) + ids[:cut] + \
                    tok(txt, add_special_tokens=False)["input_ids"] + \
                    force_finalize_ids(tok, sp)
                ins.append(TokensPrompt(prompt_token_ids=body))
                prm.append(SamplingParams(temperature=0.0, seed=SEED,
                                          max_tokens=args.answer_allowance,
                                          stop_token_ids=[sp.ret]))
            outs = llm.generate(ins, prm)
            fin[eff] = [tok.decode(list(o.outputs[0].token_ids)) for o in outs]
        chk = {
            "n": len(seqs),
            "half_tokens": HALF,
            "meaning": ("логарифм вероятности ОДНОГО И ТОГО ЖЕ продолжения, "
                        "написанного при обычном заголовке, под обычным и под "
                        "низким заголовком; плюс продолжения из середины под "
                        "обоими заголовками и их довершение"),
            "mean_logprob_normal": round(float(np.nanmean(lp["normal"])), 4),
            "mean_logprob_low": round(float(np.nanmean(lp["low"])), 4),
            "delta_mean_logprob": round(float(np.nanmean(np.asarray(lp["low"])
                                                         - np.asarray(lp["normal"]))), 4),
            "per_question_logprob": [
                {"qid": r["question_id"], "normal": round(a, 4), "low": round(b, 4)}
                for (r, _i), a, b in zip(seqs, lp["normal"], lp["low"])],
            "continuation_finish": {e: [f for (_t, f) in cont[e]] for e in cont},
            "continuation_chars": {e: [len(t) for (t, _f) in cont[e]] for e in cont},
            "answer_valid": {
                e: round(float(np.mean([1.0 if extract_concise_answer(t)[0]
                                        else 0.0 for t in fin[e]])), 3)
                for e in fin},
            "samples": [{"qid": seqs[i][0]["question_id"],
                         "cont_normal": cont["normal"][i][0][:400],
                         "cont_low": cont["low"][i][0][:400],
                         "final_low": fin["low"][i][:200]}
                        for i in range(min(3, len(seqs)))],
        }
        crit = prereg["effort_switch_check"]
        chk["criterion"] = crit
        chk["logprob_error"] = lp_error
        chk["logprob_available"] = bool(not lp_error
                                        and np.isfinite(chk["delta_mean_logprob"]))
        ok_lp = (chk["delta_mean_logprob"] >= -float(crit["max_logprob_drop"])
                 if chk["logprob_available"] else True)
        chk["passed"] = bool(ok_lp
                             and chk["answer_valid"]["low"] >= float(crit["min_answer_valid"])
                             and min(chk["continuation_chars"]["low"]) > 0)
        json.dump(chk, open(out_dir / ("effort_check_%s.json" % args.tag), "w"),
                  ensure_ascii=False, indent=1)
        print("[смена усилия] средний логарифм вероятности обычный %.4f, "
              "низкий %.4f, разность %.4f; ответ извлекается: %s; проверка: %s"
              % (chk["mean_logprob_normal"], chk["mean_logprob_low"],
                 chk["delta_mean_logprob"], chk["answer_valid"],
                 "пройдена" if chk["passed"] else "ПРОВАЛЕНА"), flush=True)
        if args.effort_check_only:
            return 0

    dump_path = out_dir / ("gate_%s.jsonl" % args.tag)
    dec_path = out_dir / ("gate_%s_decisions.jsonl" % args.tag)
    fh_dump = open(dump_path, "w", encoding="utf-8")
    fh_dec = open(dec_path, "w", encoding="utf-8")
    n_rounds_total = 0
    n_dec_total = 0

    # опорная стоимость первого цикла по бюджету и уровню усилия: один раз
    c0_of, c0l_of = {}, {}
    for B in sorted(set(budgets + cap_extra)):
        x0 = GC.build_features([], [], None, B, [0.0], model_idx, arm_idx["normal"])
        lad0 = forecaster.ladder(x0)
        c0_of[B] = {q: forecaster.at(lad0, q) for q in qlist}
        x0l = GC.build_features([], [], None, B, [0.0], model_idx, arm_idx["low"])
        lad0l = forecaster.ladder(x0l)
        c0l_of[B] = {q: forecaster.at(lad0l, q) for q in qlist}
    mult_of = {r["question_id"]: diff_head.mult(r["question"]) for r in rows}
    print("[старт] стоимость первого цикла по бюджету: %s"
          % {b: {q: round(v, 1) for q, v in d.items()} for b, d in c0_of.items()},
          flush=True)

    shards = [rows[i:i + args.qshard] for i in range(0, len(rows), args.qshard)]
    for si, shard in enumerate(shards):
        ts = time.time()
        reqs = []
        for r in shard:
            qid = r["question_id"]
            for (name, B, pol, q, ver, fixed_eff) in conds:
                if pol == "cap":
                    eff, info, eff_nd = fixed_eff, {}, fixed_eff
                elif ver == "v3":
                    eff, info = GC.start_effort(B, args.W, c0_of[B][q],
                                                mult_of[qid], P3["kappa_start"])
                    eff_nd = GC.start_effort(B, args.W, c0_of[B][q], 1.0,
                                             P3["kappa_start"])[0]
                else:
                    eff, info, eff_nd = "normal", {}, "normal"
                reqs.append(Req(qid, r, name, B, pol, q, ver,
                                prompts[qid], make_panel(), args.W,
                                eff, info, eff_nd, mult_of[qid]))
        print("[набор %d/%d] вопросов %d, запросов %d, низким усилием стартуют %d"
              % (si + 1, len(shards), len(shard), len(reqs),
                 sum(1 for r in reqs if r.effort == "low")), flush=True)

        rnd = 0
        while True:
            live = [r for r in reqs if r.closed is None]
            if not live:
                break
            rnd += 1
            n_rounds_total += 1

            def need_tokens(r):
                return max(1, min(args.chunk, int(round(r.deadline - r.spend))))

            res = gen_batch(live, need_tokens, [sp.end, sp.ret])
            for r, (ids, fr) in zip(live, res):
                r.gen = r.gen + ids
                r.n_rounds += 1
                r.finish_reason = fr
                if r.content_start is None:
                    cs = content_start_of(r.gen)
                    if cs is not None:
                        ch = first_channel(r.gen)
                        r.content_start = cs
                        if ch == "final":
                            r.closed = "direct_final"
                            r.close_rule = "no_analysis"
                if r.content_start is not None:
                    r.spend = len(r.gen) - r.content_start
                if fr != "length" and r.closed is None:
                    r.closed = "natural"
                    r.close_rule = "model_closed"

            for r in live:
                if r.content_start is None:
                    continue
                r.text = tok.decode(r.gen[r.content_start:])
                r.panel.feed(r.text, float(r.spend),
                             final=(r.closed is not None))

            for r in live:
                if r.closed is not None or r.content_start is None:
                    continue
                st = r.panel
                n_commit = int(st.stream.counts[COMMIT])
                A = st.cand_tok is not None
                new_commit = n_commit > r.last_dec_commits
                tick = A and (r.spend - r.last_dec_spend >= args.tick)
                at_deadline = r.spend >= r.deadline - 1e-9
                if r.policy == "gate" and (new_commit or tick or at_deadline):
                    events = [(t, k) for (t, k, _s) in st.stream.events]
                    cand = st.cand_tok
                    V = GC.verified_at(events, cand, float(r.spend))
                    bounds = [0.0] + sorted(t for (t, k, _s) in st.stream.events
                                            if k == COMMIT and t <= r.spend)
                    if float(r.spend) - bounds[-1] > args.chunk:
                        bounds.append(float(r.spend))
                    segs = st.stream._segment(r.text)
                    seg_tok = seg_end_tokens(segs[:-1] if len(segs) > 1 else segs,
                                             float(r.spend))
                    brem = float(r.budget) - float(r.spend)
                    cver = vtab.verify(model_key, r.q)
                    cunc = vtab.uncond(model_key, r.q)
                    x_cur = GC.build_features(events, seg_tok, cand, r.budget,
                                              bounds, model_idx,
                                              arm_idx[r.effort])
                    lad_cur = forecaster.ladder(x_cur)
                    cnext = forecaster.at(lad_cur, r.q)
                    x_low = GC.build_features(events, seg_tok, cand, r.budget,
                                              bounds, model_idx, arm_idx["low"])
                    cnext_low = forecaster.at(forecaster.ladder(x_low), r.q)
                    rec = {"question_id": r.qid, "condition": r.cond,
                           "policy_version": r.version, "budget": r.budget,
                           "q": r.q, "round": rnd, "spend": int(r.spend),
                           "brem": round(brem, 1), "W": args.W,
                           "A": bool(A), "V": bool(V), "effort": r.effort,
                           "n_commit": n_commit,
                           "n_events": int(st.stream.counts.sum()),
                           "cand_tok": (None if cand is None
                                        else round(float(cand), 1)),
                           "cnext": round(cnext, 1),
                           "cnext_low": round(cnext_low, 1),
                           "cverify": round(cver, 1), "cuncond": round(cunc, 1),
                           "trigger": ("commit" if new_commit else
                                       ("tick" if tick else "deadline"))}
                    if r.version == "v3":
                        probs = st.hawkes.next_probs()
                        wq = st.hawkes.wait_quantiles((r.q,))[0]
                        p_br = float(probs[GC.T_BR])
                        p_cm = float(probs[COMMIT])
                        tau_n = GC.tau_of(c0_of[r.budget][r.q], r.mult,
                                          P3["kappa_start"])
                        tau_l = GC.tau_of(c0l_of[r.budget][r.q], r.mult,
                                          P3["kappa_start"])
                        rec.update({"p_branch": round(p_br, 4),
                                    "p_commit": round(p_cm, 4),
                                    "next_argmax": GC.TYPES[int(np.argmax(probs))],
                                    "wait_q": round(float(wq), 1),
                                    "tau_normal": round(tau_n, 1),
                                    "tau_low": round(tau_l, 1)})

                        def run(cn, pb, pc, tn, tl, pp):
                            return GC.policy_v3(brem, args.W, A, V, r.effort,
                                                cn, cver, pb, pc, wq, tn, tl, pp)

                        d = run(cnext, p_br, p_cm, tau_n, tau_l, P3)
                        alt = {
                            "cnext_zero": run(0.0, p_br, p_cm, tau_n, tau_l, P3),
                            "cnext_inf": run(1e9, p_br, p_cm, tau_n, tau_l, P3),
                            "cnext_uncond": run(cunc, p_br, p_cm, tau_n, tau_l,
                                                P3),
                            "no_switch": run(cnext, p_br, p_cm, tau_n, tau_l,
                                             dict(P3, allow_switch=False)),
                            "no_next_type": run(cnext, p_br, p_cm, tau_n, tau_l,
                                                dict(P3, use_next_type=False)),
                            # без ключа усилия у прогнозной головы оба порога
                            # совпадают: полоса смены усилия пуста
                            "no_effort_key": run(cnext, p_br, p_cm, tau_n,
                                                 tau_n, P3),
                        }
                        rec.update({"rule_if_%s" % k: v["rule"]
                                    for k, v in alt.items()})
                        rec["rule_v21_same_state"] = GC.policy_v2(
                            brem, args.W, A, V, cnext, cver, "v2.1")["rule"]
                        rec["switched_here"] = bool(d["switched"])
                        rec["tight"] = bool(d["tight"])
                        rec["cnext_evaluated"] = d["rule"] in (
                            "1a", "1b", "2a", "2b", "3a")
                        rec["cnext_pivotal"] = bool(
                            d["rule"] != alt["cnext_zero"]["rule"]
                            or d["rule"] != alt["cnext_inf"]["rule"])
                        rec["cnext_uncond_flip"] = bool(
                            d["rule"] != alt["cnext_uncond"]["rule"])
                        # смена усилия действия не меняет, она меняет уровень
                        # усилия, поэтому связывание меряется по уровню
                        rec["switch_binding"] = bool(
                            d["rule"] != alt["no_switch"]["rule"]
                            or d["effort"] != alt["no_switch"]["effort"])
                        rec["next_type_binding"] = bool(
                            d["rule"] != alt["no_next_type"]["rule"])
                        rec["effort_key_binding"] = bool(
                            d["rule"] != alt["no_effort_key"]["rule"]
                            or d["effort"] != alt["no_effort_key"]["effort"])
                        rec["differs_from_v21"] = bool(
                            d["rule"] != rec["rule_v21_same_state"])
                    else:
                        d = GC.policy_v2(brem, args.W, A, V, cnext, cver,
                                         r.version)
                        d["effort"] = r.effort
                        d0 = GC.policy_v2(brem, args.W, A, V, 0.0, cver, r.version)
                        di = GC.policy_v2(brem, args.W, A, V, 1e9, cver, r.version)
                        du = GC.policy_v2(brem, args.W, A, V, cunc, cver, r.version)
                        rec.update({
                            "rule_if_cnext_zero": d0["rule"],
                            "rule_if_cnext_inf": di["rule"],
                            "rule_if_cnext_uncond": du["rule"],
                            "cnext_evaluated": d["rule"] in ("1a", "1b", "2a", "2b"),
                            "cnext_pivotal": bool(d["rule"] != d0["rule"]
                                                  or d["rule"] != di["rule"]),
                            "cnext_uncond_flip": bool(d["rule"] != du["rule"])})
                    rec.update({"rule": d["rule"], "act": d["act"],
                                "mode": d["mode"],
                                "allow": round(float(d["allow"]), 1),
                                "effort_after": d.get("effort", r.effort)})
                    r.decisions.append(rec)
                    fh_dec.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    n_dec_total += 1
                    r.last_dec_spend = float(r.spend)
                    r.last_dec_commits = n_commit
                    r.mode = d["mode"]
                    r.terminal = bool(d["terminal"])
                    new_eff = d.get("effort", r.effort)
                    if new_eff != r.effort:
                        r.effort = new_eff
                        r.n_switch += 1
                        if r.effort_switch_tok is None:
                            r.effort_switch_tok = float(r.spend)
                    if d["act"] == "finalize":
                        r.closed = "policy"
                        r.close_rule = d["rule"]
                    else:
                        r.deadline = min(float(r.budget) - args.W,
                                         float(r.spend) + float(d["allow"]))
                if r.closed is None and r.spend >= r.deadline - 1e-9:
                    r.closed = "terminal" if r.terminal else "budget"
                    r.close_rule = r.mode

            done = sum(1 for r in reqs if r.closed is not None)
            if rnd % 4 == 0 or done == len(reqs):
                print("   круг %3d: закрыто %4d из %4d, решений %d, %.0f с"
                      % (rnd, done, len(reqs), n_dec_total, time.time() - ts),
                      flush=True)

        for r in reqs:
            st = r.panel
            ev = [(t, k) for (t, k, _s) in st.stream.events]
            r.stop_state = {
                "A_online": st.cand_tok is not None,
                "V_online": GC.verified_at(ev, st.cand_tok, float(r.spend)),
                "n_commit_online": int(st.stream.counts[COMMIT]),
                "n_events_online": int(st.stream.counts.sum()),
            }
            if r.content_start is not None and r.text:
                st.feed(r.text, float(r.spend), final=True)

        for r in reqs:
            p = parse_completion(list(r.gen), tok, sp)
            if not p.has_final:
                r.gen = r.gen + force_finalize_ids(tok, sp)
                r.forced = True
        t0 = time.time()
        res = gen_batch(reqs, lambda r: args.answer_allowance, [sp.ret])
        for r, (ids, fr) in zip(reqs, res):
            r.gen = r.gen + ids
        print("   довершение: %d запросов за %.0f с"
              % (len(reqs), time.time() - t0), flush=True)

        for r in reqs:
            p = parse_completion(list(r.gen), tok, sp, decode_reasoning=True)
            st = r.panel
            events = [(t, k) for (t, k, _s) in st.stream.events]
            cand = st.cand_tok
            A = cand is not None
            V = GC.verified_at(events, cand, float(r.spend))
            commits = sorted(t for (t, k, _s) in st.stream.events
                             if k == COMMIT and t <= r.spend)
            last_commit = commits[-1] if commits else 0.0
            hanging = bool(r.closed not in ("natural", "direct_final")
                           and (r.spend - last_commit) > args.chunk)
            ans, src = (p.answer, p.answer_source)
            if ans is None:
                ans, src = extract_concise_answer(p.final_text)
            ok, timed_out = (False, False)
            if ans:
                ok, timed_out = safe_is_correct(is_correct,
                                                r.row["correct_answer"], ans,
                                                r.row["benchmark"],
                                                args.grade_timeout)
            rec = {
                "question_id": r.qid, "benchmark": r.row["benchmark"],
                "condition": r.cond, "budget": r.budget,
                "policy": r.policy, "q": r.q,
                "policy_version": r.version,
                "effort_start": r.start_effort, "effort_end": r.effort,
                "effort_start_if_no_difficulty": r.start_effort_no_diff,
                "difficulty_mult": round(float(r.mult), 3),
                "start_info": r.start_info,
                "n_effort_switch": r.n_switch,
                "effort_switch_tok": r.effort_switch_tok,
                "reasoning_tokens": p.reasoning_tokens,
                "spend_tracked": int(r.spend),
                "final_tokens": p.final_tokens,
                "total_tokens": p.reasoning_tokens + p.final_tokens,
                "leftover": round(float(r.budget) - p.reasoning_tokens
                                  - p.final_tokens, 1),
                "closed": r.closed, "close_rule": r.close_rule,
                "mode_at_stop": r.mode, "n_rounds": r.n_rounds,
                "forced": r.forced, "has_final": p.has_final,
                "n_decisions": len(r.decisions),
                "A_at_stop": bool(A), "V_at_stop": bool(V),
                "A_online": bool(r.stop_state.get("A_online")),
                "V_online": bool(r.stop_state.get("V_online")),
                "n_commit_online": int(r.stop_state.get("n_commit_online", 0)),
                "cand_tok": (None if cand is None else round(float(cand), 1)),
                "n_commit": int(st.stream.counts[COMMIT]),
                "n_events": int(st.stream.counts.sum()),
                "last_commit_tok": round(float(last_commit), 1),
                "wasted_in_cycle": round(float(r.spend) - float(last_commit), 1),
                "hanging": hanging,
                "answer": ans, "answer_source": src,
                "valid": bool(ans), "correct": bool(ok),
                "grade_timeout": bool(timed_out),
                "correct_answer": r.row["correct_answer"],
                "final_text": p.final_text,
                "reasoning": p.reasoning_text,
                "run_tag": args.tag,
            }
            fh_dump.write(json.dumps(rec, ensure_ascii=False) + "\n")
        fh_dump.flush()
        fh_dec.flush()
        student._cache.clear()
        print("[набор %d/%d] готов за %.0f с, кругов %d"
              % (si + 1, len(shards), time.time() - ts, rnd), flush=True)

    fh_dump.close()
    fh_dec.close()

    run = {
        "arm": "gate_v3_2026_08_14", "tag": args.tag,
        "model": MODEL_ID, "revision": REVISION, "vllm": _vllm.__version__,
        "seed": SEED, "sampling": "greedy",
        "prefix_caching": bool(args.prefix_caching),
        "policy_versions": versions, "policy_v3_params": P3,
        "chunk": args.chunk, "tick": args.tick, "W": args.W,
        "answer_allowance": args.answer_allowance,
        "budgets": budgets, "cap_extra": cap_extra, "qs": qlist,
        "low_cap_arm": bool(args.low_cap_arm),
        "conditions": [c[0] for c in conds],
        "questions": str(Path(args.questions).resolve()),
        "questions_sha256": hashlib.sha256(
            Path(args.questions).read_bytes()).hexdigest()[:16],
        "n_questions": len(rows),
        "student_ckpt": args.student_ckpt, "student_tag": student.tag_used,
        "pack_dir": args.pack_dir, "fc_dir": args.fc_dir,
        "forecaster_hp": forecaster.meta.get("hp"),
        "forecaster_selfcheck": forecaster.meta.get("selfcheck"),
        "forecaster_effort_key_fixed": forecaster.meta.get("effort_key_fixed"),
        "difficulty_head": diff_head.meta,
        "n_rounds_total": n_rounds_total, "n_decisions": n_dec_total,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
        "host": os.uname().nodename,
        "started": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(t_start)),
        "wall_s": round(time.time() - t_start, 1),
    }
    try:
        run["device"] = torch.cuda.get_device_name(0)
    except Exception:
        run["device"] = "?"
    meta_path = out_dir / ("gate_%s.meta.json" % args.tag)
    meta_path.write_text(json.dumps(run, ensure_ascii=False, indent=1),
                         encoding="utf-8")

    recs = [json.loads(l) for l in open(dump_path, encoding="utf-8")]
    print("[готово] %s\n[решения] %s\n[описание] %s"
          % (dump_path, dec_path, meta_path), flush=True)
    by = {}
    for r in recs:
        b = by.setdefault(r["condition"], {"n": 0, "ok": 0, "sp": 0.0, "ver": 0,
                                           "hang": 0, "left": 0.0, "low": 0,
                                           "sw": 0})
        b["n"] += 1
        b["ok"] += int(r["correct"])
        b["sp"] += r["total_tokens"]
        b["ver"] += int(r["V_at_stop"])
        b["hang"] += int(r["hanging"])
        b["left"] += max(r["leftover"], 0.0)
        b["low"] += int(r["effort_start"] == "low")
        b["sw"] += int(r["n_effort_switch"] > 0)
    for c in sorted(by, key=lambda k: (int(k.split("_")[-1]), k)):
        b = by[c]
        print("   %-16s n=%4d acc=%.3f расход=%6.0f проверено=%.3f "
              "повисших=%.3f обмылок=%6.0f низкий_старт=%.2f смен=%.2f"
              % (c, b["n"], b["ok"] / b["n"], b["sp"] / b["n"],
                 b["ver"] / b["n"], b["hang"] / b["n"], b["left"] / b["n"],
                 b["low"] / b["n"], b["sw"] / b["n"]), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
