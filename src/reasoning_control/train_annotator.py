"""
Обучение адаптера LoRA, превращающего локальную модель в судью-разметчика
(задание 09, 2026-08-15).

Потеря считается ТОЛЬКО на ответе — на токенах финального объекта JSON; промпт судьи и
текст окна замаскированы. Размышление судьи не воспроизводится: в цели его нет.

Адаптируются проекции внимания и общие слои, экспертные веса заморожены. У базовой
модели Qwen3.6-35B-A3B это значит: обычное внимание (q_proj, k_proj, v_proj, o_proj),
линейное внимание (in_proj_qkv, in_proj_a, in_proj_b, in_proj_z, out_proj) и общий
эксперт (shared_expert.gate_proj, up_proj, down_proj). Маршрутизатор mlp.gate и
экспертные веса mlp.experts не трогаются: экспертные веса хранятся слитыми трёхмерными
тензорами, а не слоями nn.Linear, и адаптеру недоступны. Зрительная башня и голова
многотокенного прогноза заморожены целиком.

Расход времени ограничен ключом --max-hours: по его достижении обучение штатно
останавливается и адаптер сохраняется. Это и есть соблюдение потолка партии в
GPU-часах при неизвестной заранее скорости.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time

import torch


def target_modules_for(model_name, model_type=""):
    """Имена подмодулей nn.Linear для адаптера. Совпадения по суффиксу имени."""
    if ("Qwen3.6" in model_name or "Qwen3.5" in model_name
            or "qwen3_5" in model_name.lower() or "qwen3_5" in model_type.lower()):
        return ["q_proj", "k_proj", "v_proj", "o_proj",
                "in_proj_qkv", "in_proj_a", "in_proj_b", "in_proj_z", "out_proj",
                "shared_expert.gate_proj", "shared_expert.up_proj",
                "shared_expert.down_proj"]
    # плотные модели семейства Llama и Qwen: внимание плюс перцептрон
    return ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--train", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--alpha", type=int, default=64)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--warmup-frac", type=float, default=0.03)
    ap.add_argument("--max-examples", type=int, default=0)
    ap.add_argument("--max-hours", type=float, default=3.0)
    ap.add_argument("--seed", type=int, default=20260815)
    ap.add_argument("--save-every", type=int, default=100, help="шагов оптимизатора")
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--device-map", default="balanced")
    ap.add_argument("--max-memory", default=None,
                    help="json вида {\"0\": \"30GiB\", \"1\": \"38GiB\"}")
    ap.add_argument("--attn", default="sdpa")
    ap.add_argument("--ssm-dtype", default=None,
                    help="переопределить mamba_ssm_dtype, например bfloat16")
    ap.add_argument("--smoke", type=int, default=0, help="сделать N шагов и выйти")
    ap.add_argument("--max-seq", type=int, default=0, help="выбросить окна длиннее")
    ap.add_argument("--ckpt-every", type=int, default=0,
                    help="сохранять ОТДЕЛЬНЫЙ чекпоинт каждые N примеров")
    ap.add_argument("--val", default=None,
                    help="токенизированный валидационный срез: на нём считается "
                         "потеря при каждом чекпоинте")
    ap.add_argument("--val-max", type=int, default=60)
    ap.add_argument("--init-adapter", default=None,
                    help="каталог обученного адаптера: продолжить его, а не начинать "
                         "с нуля")
    ap.add_argument("--probe-lengths", default=None,
                    help="через запятую: померить пик памяти на этих длинах и выйти")
    args = ap.parse_args()

    if os.path.exists(args.out):
        raise FileExistsError("Use a new output directory: " + args.out)

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    from peft import LoraConfig, PeftModel, get_peft_model          # noqa: E402
    from transformers import AutoConfig, AutoModelForCausalLM       # noqa: E402

    rows = [json.loads(l) for l in open(args.train)]
    if args.max_seq:
        before = len(rows)
        rows = [r for r in rows
                if len(r["prompt_ids"]) + len(r["target_ids"]) <= args.max_seq]
        print(f"по длине оставлено {len(rows)} из {before}", flush=True)
    if args.max_examples:
        rows = rows[:args.max_examples]
    print(f"примеров: {len(rows)}", flush=True)

    cfg = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    if args.ssm_dtype:
        # Состояние линейного внимания у Qwen3.6 по умолчанию держится в float32 и
        # стоит около десяти мегабайт на токен; половина этого — цена самой точности.
        # Ключ переводит его в bfloat16 и вдвое снижает пик.
        tc = getattr(cfg, "text_config", cfg)
        setattr(tc, "mamba_ssm_dtype", args.ssm_dtype)
        print("состояние линейного внимания:", getattr(tc, "mamba_ssm_dtype", None),
              flush=True)
    t0 = time.time()
    load_kwargs = dict(dtype=torch.bfloat16, device_map=args.device_map,
                       trust_remote_code=True, attn_implementation=args.attn,
                       config=cfg)
    if args.max_memory:
        load_kwargs["max_memory"] = {int(k): v for k, v in
                                     json.loads(args.max_memory).items()}
    model = AutoModelForCausalLM.from_pretrained(args.model, **load_kwargs)
    print(f"веса загружены за {time.time() - t0:.0f} c", flush=True)
    model.config.use_cache = False
    dmap = getattr(model, "hf_device_map", None) or {}
    print("карты весов:", dmap, flush=True)
    off = sorted(k for k, v in dmap.items() if v in ("disk", "cpu"))
    if off:
        # Вытеснение хотя бы одного слоя на диск замедляет шаг в разы: голова словаря
        # на 248320 строк читается с диска на каждом проходе. Лучше упасть сразу.
        raise SystemExit("на диск или процессор вытеснено %d частей: %s. "
                         "Поднимите --max-memory" % (len(off), off[:6]))
    for d in range(torch.cuda.device_count()):
        print(f"после загрузки карта {d}: "
              f"{torch.cuda.memory_allocated(d) / 2**30:.2f} ГиБ", flush=True)

    # Пересчёт активаций включается ДО обёртки адаптером: если звать его после,
    # обёртка peft уже спрятала базовые слои и включение молча не доходит до них —
    # именно так две карты A6000 переполнялись на первом же окне.
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False})

    # Имена подмодулей перечисляются полностью, а не суффиксами. Причина: суффиксы
    # q_proj и o_proj встречаются ещё и в голове многотокенного прогноза mtp, чьи веса
    # при разбиении по картам остаются на устройстве meta — адаптер на них падает в
    # обратном проходе с сообщением о несовпадении устройств. Заодно отсекается
    # зрительная башня visual: к разметке текста она отношения не имеет.
    suffixes = target_modules_for(args.model, getattr(getattr(cfg, "text_config", cfg), "model_type", ""))
    tm = []
    for name, mod in model.named_modules():
        if not isinstance(mod, torch.nn.Linear):
            continue
        if name.startswith("mtp") or ".mtp." in name or "visual" in name:
            continue
        w = getattr(mod, "weight", None)
        if w is None or w.device.type == "meta":
            continue
        if any(name.endswith(s) for s in suffixes):
            tm.append(name)
    print(f"адаптируемых подмодулей: {len(tm)}", flush=True)
    if args.init_adapter:
        # Дообучение продолжает уже обученный адаптер, а не начинает с нуля: на этапе
        # активного отбора сравниваются два ПРОДОЛЖЕНИЯ одного и того же адаптера, и
        # общая точка старта обязательна, иначе сравнение измеряло бы случайность
        # инициализации.
        model = PeftModel.from_pretrained(model, args.init_adapter, is_trainable=True)
        print(f"адаптер продолжен из {args.init_adapter}", flush=True)
    else:
        lconf = LoraConfig(r=args.rank, lora_alpha=args.alpha,
                           lora_dropout=args.dropout, bias="none",
                           task_type="CAUSAL_LM", target_modules=tm)
        model = get_peft_model(model, lconf)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_all = sum(p.numel() for p in model.parameters())
    print(f"обучаемых параметров {n_train} из {n_all} ({100 * n_train / n_all:.4f} %)",
          flush=True)
    named = [n for n, p in model.named_parameters() if p.requires_grad]
    print("пример адаптированных имён:", named[:4], "...всего", len(named), flush=True)
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False})
    if hasattr(model, "base_model"):
        try:
            model.base_model.model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False})
        except Exception:                                           # noqa: BLE001
            pass
    # Режим обучения обязателен: пересчёт активаций внутри слоя срабатывает только при
    # self.training, а from_pretrained возвращает модель в режиме вывода. Без этой
    # строки пересчёт числится включённым, но не работает, и активации всех сорока
    # слоёв остаются в памяти — так и переполнялись обе карты.
    model.train()
    print("пересчёт активаций включён:",
          getattr(getattr(model, "base_model", model), "is_gradient_checkpointing", None),
          "режим обучения:", model.training, flush=True)

    dev_in = next(model.parameters()).device
    params = [p for p in model.parameters() if p.requires_grad]

    if args.probe_lengths:
        ce_p = torch.nn.CrossEntropyLoss()
        base = [torch.cuda.memory_allocated(d) / 2**30
                for d in range(torch.cuda.device_count())]
        print("веса на картах, ГиБ:", [round(b, 2) for b in base], flush=True)
        for L in [int(v) for v in args.probe_lengths.split(",")]:
            for d in range(torch.cuda.device_count()):
                torch.cuda.reset_peak_memory_stats(d)
            n_t = 256
            ids = (rows[0]["prompt_ids"] + rows[0]["target_ids"])
            ids = (ids * (L // len(ids) + 1))[:L]
            try:
                x = torch.tensor([ids], device=dev_in)
                out = model(input_ids=x, logits_to_keep=n_t + 1)
                lg = out.logits[:, :-1, :]
                tg = torch.tensor([ids[-n_t:]], device=lg.device)
                loss = ce_p(lg.reshape(-1, lg.size(-1)).float(), tg.reshape(-1))
                loss.backward()
                peak = [round(torch.cuda.max_memory_allocated(d) / 2**30, 2)
                        for d in range(torch.cuda.device_count())]
                extra = [round(peak[d] - base[d], 2) for d in range(len(peak))]
                print(f"длина {L}: пик {peak} ГиБ, сверх весов {extra} ГиБ", flush=True)
            except torch.OutOfMemoryError as ex:                    # noqa: F821
                print(f"длина {L}: переполнение памяти — {str(ex)[:120]}", flush=True)
            for p in params:
                p.grad = None
            torch.cuda.empty_cache()
        return
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0, betas=(0.9, 0.95))

    total_steps = max(1, len(rows) // args.accum)
    warmup = max(1, int(args.warmup_frac * total_steps))

    def lr_at(step):
        if step < warmup:
            return args.lr * (step + 1) / warmup
        prog = (step - warmup) / max(1, total_steps - warmup)
        return args.lr * 0.5 * (1 + math.cos(math.pi * min(1.0, prog)))

    os.makedirs(args.out, exist_ok=True)
    log_path = os.path.join(args.out, "train_log.jsonl")
    logf = open(log_path, "a")
    deadline = time.time() + args.max_hours * 3600
    step, micro, run_loss, run_tok = 0, 0, 0.0, 0
    t_start = time.time()
    stopped = "completed"

    # Потеря считается только на ответе, поэтому логиты нужны лишь на его позициях.
    # Полная матрица логитов на окно в три тысячи токенов при словаре 248320 стоит
    # полтора гигабайта в bf16 и ещё три при повышении точности в перекрёстной
    # энтропии — с ней две карты A6000 переполняются. Ключ logits_to_keep обрезает
    # её до длины ответа, это примерно в четыре раза меньше.
    ce = torch.nn.CrossEntropyLoss()
    val_rows = []
    if args.val and os.path.exists(args.val):
        val_rows = [json.loads(l) for l in open(args.val)][:args.val_max]
        print(f"валидационных окон: {len(val_rows)}", flush=True)
    for i, r in enumerate(rows):
        ids = r["prompt_ids"] + r["target_ids"]
        n_t = len(r["target_ids"])
        x = torch.tensor([ids], device=dev_in)
        out = model(input_ids=x, logits_to_keep=n_t + 1)
        logits = out.logits[:, :-1, :]
        tgt = torch.tensor([r["target_ids"]], device=logits.device)
        loss_full = ce(logits.reshape(-1, logits.size(-1)).float(), tgt.reshape(-1))
        loss = loss_full / args.accum
        loss.backward()
        run_loss += loss_full.item()
        run_tok += n_t
        micro += 1
        if micro % args.accum == 0:
            for g in opt.param_groups:
                g["lr"] = lr_at(step)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if step % args.log_every == 0:
                el = time.time() - t_start
                rec = {"step": step, "example": i + 1,
                       "loss": round(run_loss / (args.log_every * args.accum), 4),
                       "lr": lr_at(step), "elapsed_s": round(el, 1),
                       "target_tokens_per_s": round(run_tok / el, 1)}
                print(json.dumps(rec), flush=True)
                logf.write(json.dumps(rec) + "\n")
                logf.flush()
                run_loss = 0.0
            if args.save_every and step % args.save_every == 0:
                model.save_pretrained(args.out)
        if args.ckpt_every and (i + 1) % args.ckpt_every == 0:
            # Отдельный нумерованный чекпоинт: обрыв обучения после любого из них
            # безопасен по построению, владелец решает по кривой, когда остановиться.
            cdir = os.path.join(args.out, "ckpt_%06d" % (i + 1))
            model.save_pretrained(cdir)
            vloss = None
            if val_rows:
                model.eval()
                tot, n_t_tok = 0.0, 0
                with torch.no_grad():
                    for vr in val_rows:
                        vin = vr["prompt_ids"] + vr["target_ids"]
                        nt = len(vr["target_ids"])
                        vx = torch.tensor([vin], device=dev_in)
                        vo = model(input_ids=vx, logits_to_keep=nt + 1)
                        vl = vo.logits[:, :-1, :]
                        vt = torch.tensor([vr["target_ids"]], device=vl.device)
                        tot += ce(vl.reshape(-1, vl.size(-1)).float(),
                                  vt.reshape(-1)).item() * nt
                        n_t_tok += nt
                vloss = tot / max(1, n_t_tok)
                model.train()
            row = {"checkpoint": os.path.basename(cdir), "examples": i + 1,
                   "step": step, "val_loss": (None if vloss is None
                                              else round(vloss, 4)),
                   "train_loss_recent": round(run_loss / max(1, args.log_every *
                                                             args.accum), 4),
                   "elapsed_hours": round((time.time() - t_start) / 3600, 3),
                   "gpu_hours": round((time.time() - t_start) / 3600 *
                                      torch.cuda.device_count(), 3),
                   "at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
            with open(os.path.join(args.out, "curve.jsonl"), "a") as cf:
                cf.write(json.dumps(row, ensure_ascii=False) + "\n")
            print("CKPT " + json.dumps(row, ensure_ascii=False), flush=True)
            if args.smoke and step >= args.smoke:
                stopped = "smoke"
                break
        if time.time() > deadline:
            stopped = "time_limit"
            break

    model.save_pretrained(args.out)
    elapsed = time.time() - t_start
    meta = {"model": args.model, "rank": args.rank, "alpha": args.alpha,
            "dropout": args.dropout, "lr": args.lr, "accum": args.accum,
            "seed": args.seed, "target_modules": tm,
            "init_adapter": args.init_adapter,
            "trainable_params": n_train, "all_params": n_all,
            "examples_total": len(rows), "examples_seen": min(i + 1, len(rows)),
            "optimizer_steps": step, "planned_steps": total_steps,
            "stopped": stopped, "elapsed_hours": round(elapsed / 3600, 3),
            "gpus": torch.cuda.device_count(),
            "gpu_hours": round(elapsed / 3600 * torch.cuda.device_count(), 3),
            "target_tokens_per_s": round(run_tok / elapsed, 1) if elapsed else None,
            "peak_mem_gb": [round(torch.cuda.max_memory_allocated(d) / 2**30, 2)
                            for d in range(torch.cuda.device_count())],
            "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    json.dump(meta, open(os.path.join(args.out, "train_meta.json"), "w"),
              ensure_ascii=False, indent=2)
    print(json.dumps(meta, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
