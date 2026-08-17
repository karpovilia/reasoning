#!/usr/bin/env bash
# Водитель обучения и оценки двух плеч прогнозиста конца цикла на боксе tesla.
#
# Ждёт, пока обе карты освободятся, выдерживает льготное окно на случай, если
# карты забирает задание 01 структурного терминатора, у которого приоритет,
# и только потом начинает свою работу. Каждая фаза записывает отметку времени,
# так что обрыв виден по журналу.
#
# Запуск: nohup bash box_run.sh > logs/driver.log 2>&1 &
set -u

ROOT=/home/user/ki_cycle_end
BASE=/home/user/hf_cache_reasoning/models--Qwen--Qwen3.6-35B-A3B/snapshots/995ad96eacd98c81ed38be0c5b274b04031597b0
PY=/home/user/ki_sft_judge/.venv/bin/python
TRAIN=/home/user/ki_lora_judge/src/train_lora.py
MM='{"0": "36GiB", "1": "40GiB"}'
FREE_MIB=${FREE_MIB:-3000}
GRACE_MIN=${GRACE_MIN:-15}
N_TRAIN=${N_TRAIN:-2800}
N_EVAL=${N_EVAL:-1500}

cd "$ROOT" || exit 1
mkdir -p logs outputs

stamp() { date -u '+%Y-%m-%d %H:%M:%S UTC'; }
say() { echo "[$(stamp)] $*"; }

cards_free() {
    local u
    u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
    for v in $u; do
        if [ "$v" -gt "$FREE_MIB" ]; then return 1; fi
    done
    return 0
}

wait_free() {
    local need=$1 ok=0
    say "жду освобождения карт, порог $FREE_MIB МиБ, льготное окно $need минут"
    while true; do
        if [ -f "$ROOT/HOLD" ]; then
            if [ "$ok" -ne -1 ]; then say "найден файл HOLD, уступаю карты и жду его удаления"; fi
            ok=-1
            sleep 60
            continue
        fi
        if [ "$ok" -eq -1 ]; then say "файл HOLD снят, продолжаю ожидание карт"; ok=0; fi
        if cards_free; then
            ok=$((ok + 1))
        else
            if [ "$ok" -gt 0 ]; then say "карты снова заняты, счётчик льготного окна сброшен"; fi
            ok=0
        fi
        if [ "$ok" -ge "$need" ]; then
            say "карты свободны $need минут подряд, начинаю"
            return 0
        fi
        sleep 60
    done
}

train_arm() {
    local arm=$1 hours=$2
    if [ -f "outputs/lora_${arm}/adapter_model.safetensors" ]; then
        say "плечо $arm уже обучено, пропускаю"
        return 0
    fi
    say "обучение плеча $arm начато, потолок $hours ч, окон $N_TRAIN"
    CUDA_VISIBLE_DEVICES=0,1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "$PY" -u "$TRAIN" \
        --model "$BASE" \
        --train "data/shuf_${arm}_train.jsonl" \
        --val "data/shuf_${arm}_val.jsonl" --val-max 60 \
        --out "outputs/lora_${arm}" \
        --rank 32 --alpha 64 --dropout 0.05 --lr 1e-4 --accum 4 \
        --warmup-frac 0.005 --seed 20260816 \
        --max-examples "$N_TRAIN" --max-hours "$hours" \
        --ckpt-every 350 --save-every 100 --log-every 20 \
        --max-memory "$MM" > "logs/train_${arm}.log" 2>&1
    local rc=$?
    say "обучение плеча $arm закончено, код $rc"
    return $rc
}

eval_arm() {
    local arm=$1
    if [ -f "outputs/pred_${arm}.jsonl" ]; then
        say "оценка плеча $arm уже есть, пропускаю"
        return 0
    fi
    say "оценка плеча $arm начата, окон $N_EVAL"
    CUDA_VISIBLE_DEVICES=0,1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "$PY" -u src/eval_head.py \
        --model "$BASE" --adapter "outputs/lora_${arm}" \
        --data "data/shuf_${arm}_test.jsonl" --limit "$N_EVAL" \
        --out "outputs/pred_${arm}.jsonl" \
        --label-ids outputs/label_tokens.json \
        --max-memory "$MM" > "logs/eval_${arm}.log" 2>&1
    local rc=$?
    say "оценка плеча $arm закончена, код $rc"
    return $rc
}

say "водитель ki_cycle_end поднят"
wait_free "$GRACE_MIN"
train_arm noinj 2.2 || say "плечо noinj завершилось с ошибкой, смотри logs/train_noinj.log"
wait_free 1
eval_arm noinj || say "оценка noinj завершилась с ошибкой"
wait_free 1
train_arm inj 3.2 || say "плечо inj завершилось с ошибкой, смотри logs/train_inj.log"
wait_free 1
eval_arm inj || say "оценка inj завершилась с ошибкой"
say "водитель закончил"
