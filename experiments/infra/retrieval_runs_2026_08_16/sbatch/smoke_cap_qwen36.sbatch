#!/bin/bash
#SBATCH --job-name=ret16_q36_cap
#SBATCH --account=proj_1782
#SBATCH --partition=test
#SBATCH --constraint=type_e
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:28:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#
# Смоук починки предела хода, модель Qwen3.6-35B-A3B-FP8.
#
# Что проверяем. Раньше предел хода оседал в поле объекта клиента и до движка не
# доходил. Прямая проверка «ни один ход не длиннее 7000» на трёх вопросах ничего
# не доказывает: по корпусу 15 августа длиннее 7000 оказались лишь 0.35 процента
# ходов, на трёх вопросах их скорее всего не будет и без починки. Поэтому смоук
# идёт двумя проходами по одним и тем же трём вопросам.
#
# Проход 1, предел хода 512. Если поле дошло до движка, ни один ход не длиннее
# 512 токенов и ходы завершаются причиной length. Это и есть доказательство, что
# параметр работает, а не молча теряется.
# Проход 2, предел хода 7000 — рабочее значение перепрогона: трассы должны идти
# обычным порядком и укладываться в предел.
#
# Партиция test: не больше 10 ядер на узел и не больше 30 минут, отсюда 8 ядер и
# 28 минут. Подъём движка при окне 32768 занимал 130-140 секунд.
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
RUNNER_SRC=${REPO}/reasoning_trajectories/qwen_runner/src

export HF_HOME=${REPO}/.hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCHINDUCTOR_CACHE_DIR=${REPO}/.cache/inductor
export VLLM_CACHE_ROOT=${REPO}/.cache/vllm
export VLLM_USE_DEEP_GEMM=0
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

SERVED_NAME=Qwen/Qwen3.6-35B-A3B-FP8
SEED=101
TEMPERATURE=1.0
DATASET=hotpotqa
SMOKE_LIMIT=${SMOKE_LIMIT:-3}
WORKERS=${WORKERS:-3}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-32768}
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
    --tool-call-parser qwen3_xml \
    --gpu-memory-utilization 0.92 \
    > "${ARM}/logs/vllm_${SLURM_JOB_ID}.log" 2>&1 &
VLLM_PID=$!
for i in $(seq 1 180); do
    if curl -sf "http://localhost:${PORT}/health" >/dev/null 2>&1; then echo "[vllm] поднялся за ${i}0 с"; break; fi
    if ! kill -0 ${VLLM_PID} 2>/dev/null; then echo "[vllm] движок умер, смотри лог vllm"; exit 1; fi
    sleep 10
done

for CAP in 512 7000; do
    OUTDIR=${ARM}/smoke_cap/qwen3.6-35b-a3b-fp8/cap${CAP}
    mkdir -p "${OUTDIR}"
    echo "[смоук] предел хода ${CAP}, вопросов ${SMOKE_LIMIT}, окно ${MAX_MODEL_LEN}"
    WIKI_BASE_URL="http://127.0.0.1:${KPORT}" \
    WIKI_BACKEND=kiwix \
    PYTHONPATH="${RUNNER_SRC}" \
    python -u -m qwen_runner.cli \
        --dataset "${DATASET}" \
        --endpoint-url "http://localhost:${PORT}/v1" \
        --model "${SERVED_NAME}" \
        --no-reasoning-field \
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

echo "### ГОТОВО ret16_q36_cap ###"
