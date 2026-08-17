#!/usr/bin/env bash
# Справочный бар Б3: принятый прогнозист остатка текущего цикла на наших окнах.
#
# Три шага: запрос тем же кодом, что у принятого эксперимента; токенизация; один прямой проход
# на окно с наложенным адаптером. Веса не сливаются, движок вывода не поднимается.
set -u
ROOT=/home/user/ki_cycles
BASE=/home/user/hf_cache_reasoning/models--Qwen--Qwen3.6-35B-A3B/snapshots/995ad96eacd98c81ed38be0c5b274b04031597b0
ADAPTER=/home/user/ki_cycle_end/outputs/lora_noinj
PY=/home/user/ki_sft_judge/.venv/bin/python
cd "$ROOT" || exit 1
mkdir -p logs outputs/main data/lora_windows

echo "[$(date -u '+%F %T UTC')] запросы"
"$PY" -u src/build_lora_prompts.py --src /home/user/ki_cycle_end/src \
    --inp data/lora_windows/windows_test.jsonl \
    --out data/lora_windows/prompts_noinj_test.jsonl || exit 1

echo "[$(date -u '+%F %T UTC')] токенизация"
"$PY" -u /home/user/ki_cycle_end/src/tokenize_pairs_cycle.py --model "$BASE" \
    --inp data/lora_windows/prompts_noinj_test.jsonl \
    --out data/lora_windows/tok_noinj_test.jsonl \
    --label-check outputs/main/label_tokens.json || exit 1

echo "[$(date -u '+%F %T UTC')] прямой проход с адаптером"
CUDA_VISIBLE_DEVICES=0,1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
"$PY" -u /home/user/ki_cycle_end/src/eval_head.py \
    --model "$BASE" --adapter "$ADAPTER" \
    --data data/lora_windows/tok_noinj_test.jsonl \
    --out outputs/main/pred_lora_ref.jsonl \
    --label-ids outputs/main/label_tokens.json \
    --max-memory '{"0": "36GiB", "1": "40GiB"}' || exit 1
echo "[$(date -u '+%F %T UTC')] готово"
