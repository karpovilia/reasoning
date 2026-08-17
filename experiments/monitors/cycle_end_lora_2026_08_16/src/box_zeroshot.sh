#!/usr/bin/env bash
# Необязательный нулевой замер: та же голова, но без адаптера. Показывает, что именно
# дало обучение, а что модель умела и так. Запускается вручную ПОСЛЕ того, как водитель
# box_run.sh закончил обе ветви, и только если карты свободны и задание 01 на них не
# претендует. Срез укорочен до 600 окон, чтобы не съедать потолок партии.
set -u
ROOT=/home/user/ki_cycle_end
BASE=/home/user/hf_cache_reasoning/models--Qwen--Qwen3.6-35B-A3B/snapshots/995ad96eacd98c81ed38be0c5b274b04031597b0
PY=/home/user/ki_sft_judge/.venv/bin/python
MM='{"0": "36GiB", "1": "40GiB"}'
cd "$ROOT" || exit 1
echo "[$(date -u '+%F %T UTC')] нулевой замер начат"
CUDA_VISIBLE_DEVICES=0,1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
"$PY" -u src/eval_head.py --model "$BASE" \
    --data data/shuf_noinj_test.jsonl --limit 600 \
    --out outputs/pred_zeroshot_noinj.jsonl \
    --label-ids outputs/label_tokens.json \
    --max-memory "$MM" > logs/eval_zeroshot.log 2>&1
echo "[$(date -u '+%F %T UTC')] нулевой замер закончен, код $?"
