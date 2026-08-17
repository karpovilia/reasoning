#!/bin/bash
#SBATCH --job-name=ret16_oss_cap
#SBATCH --account=proj_1782
#SBATCH --partition=test
#SBATCH --constraint=type_e
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:25:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#
# Смоук добавленного предела хода, модель gpt-oss-20b. Логика та же, что у
# смоука Qwen: два прохода по одним и тем же трём вопросам, сначала с пределом
# 512 (доказательство, что поле дошло до движка — ходы обрываются причиной
# length), затем с рабочим пределом 7000.
#
# Партиция test: не больше 10 ядер на узел и не больше 30 минут.
set -eo pipefail
set -x

module purge
module load Python/Anaconda_v02.2024
eval "$(conda shell.bash hook)"
conda activate vllm_qwen
export PATH="${CONDA_PREFIX}/bin:${PATH}"
export PYTHONNOUSERSITE=1

REPO="${REPO:-$HOME/reasoning}"
ARM=${REPO}/internal_signals_poc/retrieval_runs_2026_08_16_fixed
OLD_ARM=${REPO}/internal_signals_poc/retrieval_runs_2026_08_15
RUNNER_SRC=${REPO}/reasoning_trajectories/gpt_oss_runner/src

export HF_HOME=${REPO}/.hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCHINDUCTOR_CACHE_DIR=${REPO}/.cache/inductor
export VLLM_CACHE_ROOT=${REPO}/.cache/vllm
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export TIKTOKEN_RS_CACHE_DIR=${REPO}/.harmony_cache

SERVED_NAME=openai/gpt-oss-20b
SEED=101
TEMPERATURE=1.0
DATASET=hotpotqa
SMOKE_LIMIT=${SMOKE_LIMIT:-3}
WORKERS=${WORKERS:-3}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-24576}
KIWIX_BIN=${REPO}/tools/kiwix-tools_linux-x86_64-3.8.2/kiwix-serve
ZIM_PATH="${ZIM_PATH:?путь к архиву энциклопедии задаётся переменной среды ZIM_PATH}"

mkdir -p "${ARM}/logs" "${TORCHINDUCTOR_CACHE_DIR}" "${VLLM_CACHE_ROOT}"

PORT=$(( 8000 + (SLURM_JOB_ID % 20000) ))
KPORT=$(( PORT + 1 ))

"${KIWIX_BIN}" --port "${KPORT}" --threads 8 --address 127.0.0.1 "${ZIM_PATH}" \
    > "${ARM}/logs/kiwix_${SLURM_JOB_ID}.log" 2>&1 &
KPID=$!
trap 'kill ${KPID} ${VLLM_PID} 2>/dev/null || true' EXIT
for i in $(seq 1 120); do
    if curl -sf "http://127.0.0.1:${KPORT}/catalog/v2/entries?count=1" >/dev/null 2>&1; then
        echo "[kiwix] поднялся за ${i} с"; break; fi
    if ! kill -0 ${KPID} 2>/dev/null; then echo "[kiwix] умер"; exit 1; fi
    sleep 1
done

vllm serve "${SERVED_NAME}" \
    --served-model-name "${SERVED_NAME}" \
    --port "${PORT}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --enable-auto-tool-choice \
    --tool-call-parser openai \
    --enforce-eager \
    --no-enable-chunked-prefill \
    --gpu-memory-utilization 0.90 \
    > "${ARM}/logs/vllm_${SLURM_JOB_ID}.log" 2>&1 &
VLLM_PID=$!
for i in $(seq 1 180); do
    if curl -sf "http://localhost:${PORT}/health" >/dev/null 2>&1; then echo "[vllm] поднялся за ${i}0 с"; break; fi
    if ! kill -0 ${VLLM_PID} 2>/dev/null; then echo "[vllm] движок умер, смотри лог vllm"; exit 1; fi
    sleep 10
done

for CAP in 512 7000; do
    OUTDIR=${ARM}/smoke_cap/gpt-oss-20b/cap${CAP}
    mkdir -p "${OUTDIR}"
    echo "[смоук] предел хода ${CAP}, вопросов ${SMOKE_LIMIT}, окно ${MAX_MODEL_LEN}"
    WIKI_BASE_URL="http://127.0.0.1:${KPORT}" \
    WIKI_BACKEND=kiwix \
    GPT_OSS_ENDPOINT_URL="http://localhost:${PORT}/v1" \
    GPT_OSS_MULTI_HOP_TOOL_MODE=structured \
    PYTHONPATH="${RUNNER_SRC}" \
    python -u -m gpt_oss_runner.cli \
        --dataset "${DATASET}" \
        --endpoint-url "http://localhost:${PORT}/v1" \
        --model "${SERVED_NAME}" \
        --multi-hop-tool-mode structured \
        --max-tool-rounds 12 \
        --max-tokens "${CAP}" \
        --send-max-tokens \
        --question-ids-file "${OLD_ARM}/questions/qids_${DATASET}_full.txt" \
        --temperature "${TEMPERATURE}" \
        --seed "${SEED}" \
        --workers "${WORKERS}" \
        --output-dir "${OUTDIR}" \
        --limit "${SMOKE_LIMIT}"
done

echo "### ГОТОВО ret16_oss_cap ###"
