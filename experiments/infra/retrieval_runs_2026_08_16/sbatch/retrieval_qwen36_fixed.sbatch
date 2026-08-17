#!/bin/bash
#SBATCH --job-name=ret16_q36
#SBATCH --account=proj_1782
#SBATCH --partition=rocky
#SBATCH --constraint=type_e
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=03:30:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#
# Перепрогон поисковых трасс Qwen3.6-35B-A3B-FP8 с починенным пределом хода.
#
# Отличие от прогона 15 августа ровно одно содержательное: раннер теперь кладёт
# предел длины хода в тело запроса. Раньше поле max_tokens оседало в объекте
# клиента и до движка не доходило, поэтому свой сервер отдавал под один ход весь
# свободный остаток окна: самый длинный ход занял 28265 токенов на hotpotqa и
# 43073 на musique, такие трассы умирали по пределу ожидания в 1200 секунд.
#
# Окно диалога выбрано по фактам корпуса 15 августа, а не по запасу наугад.
# Движок отвергает запрос ответом 400, когда история плюс запрошенный предел
# ответа перерастают окно, значит нужно окно не меньше, чем наибольшая история
# плюс 7000. По трассам, где ни один ход не превысил 7000 токенов, наибольшая
# история составила 19980 токенов на hotpotqa и 16235 на musique, то есть
# требуется 26980. Окно 32768 покрывает все такие трассы, окно 24576 покрывает
# 99.79 процента hotpotqa. Взято 32768.
#
# Число одновременных запросов поднято с 16 до 32. Пул кэша ключей и значений на
# A100 движок измерил как 1631846 токенов, при окне 32768 это 49.8 запроса, так
# что 32 запроса занимают 64 процента пула. Прежние 16 запросов давали 10.94
# секунды на трассу, чего на 1000 трасс не хватало в три часа.
#
# Веса квантованы блочным fp8. На счётных узлах нет компилятора nvcc, а
# библиотека DeepGEMM собирает ядро умножения в рантайме именно им и падает на
# пустом бинарнике — поэтому умножение уводится на чистый Triton переменной
# VLLM_USE_DEEP_GEMM=0. Режим eager при этом НЕ нужен: он только замедляет.
#
# Не ставить --mem на rocky: тихий отказ за пять секунд.
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
TRAJ=${REPO}/reasoning_trajectories
RUNNER_SRC=${TRAJ}/qwen_runner/src

export HF_HOME=${REPO}/.hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCHINDUCTOR_CACHE_DIR=${REPO}/.cache/inductor
export VLLM_CACHE_ROOT=${REPO}/.cache/vllm
export VLLM_USE_DEEP_GEMM=0
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

MODEL_HF=${MODEL_HF:-Qwen/Qwen3.6-35B-A3B-FP8}
SERVED_NAME=${SERVED_NAME:-Qwen/Qwen3.6-35B-A3B-FP8}
MODEL_LABEL=${MODEL_LABEL:-qwen3.6-35b-a3b-fp8}
SEED=${SEED:-101}
TEMPERATURE=${TEMPERATURE:-1.0}
DATASETS=${DATASETS:-"hotpotqa musique"}
OUT_ROOT=${OUT_ROOT:-${ARM}/${MODEL_LABEL}}
KIWIX_BIN=${REPO}/tools/kiwix-tools_linux-x86_64-3.8.2/kiwix-serve
ZIM_PATH="${ZIM_PATH:?путь к архиву энциклопедии задаётся переменной среды ZIM_PATH}"
WORKERS=${WORKERS:-32}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-32768}
# Предел длины одного хода модели. Уходит в тело запроса только вместе с флагом
# --send-max-tokens, иначе поведение остаётся прежним.
MAX_TOKENS=${MAX_TOKENS:-7000}
MAX_TOOL_ROUNDS=${MAX_TOOL_ROUNDS:-12}
TOOL_MODE=${TOOL_MODE:-structured}
TP=${TP:-1}
# Списки вопросов те же, что у прогона 15 августа: перепрогон обязан идти по тем
# же 500 плюс 500 вопросам, иначе корпуса несравнимы.
QIDS_TMPL=${QIDS_TMPL:-${OLD_ARM}/questions/qids_DATASET_full.txt}

mkdir -p "${ARM}/logs" "${TORCHINDUCTOR_CACHE_DIR}" "${VLLM_CACHE_ROOT}"

PORT=$(( 8000 + (SLURM_JOB_ID % 20000) ))
KPORT=$(( PORT + 1 ))

# ---- офлайн-Википедия ----
"${KIWIX_BIN}" --port "${KPORT}" --threads 16 --address 127.0.0.1 "${ZIM_PATH}" \
    > "${ARM}/logs/kiwix_${SLURM_JOB_ID}.log" 2>&1 &
KPID=$!
trap 'kill ${KPID} ${VLLM_PID} 2>/dev/null || true' EXIT
for i in $(seq 1 120); do
    if curl -sf "http://127.0.0.1:${KPORT}/catalog/v2/entries?count=1" >/dev/null 2>&1; then
        echo "[kiwix] поднялся за ${i} с"; break; fi
    if ! kill -0 ${KPID} 2>/dev/null; then echo "[kiwix] умер"; exit 1; fi
    sleep 1
done

# ---- модель ----
vllm serve "${MODEL_HF}" \
    --served-model-name "${SERVED_NAME}" \
    --port "${PORT}" \
    --tensor-parallel-size "${TP}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_xml \
    --gpu-memory-utilization 0.92 \
    ${VLLM_EXTRA:-} \
    > "${ARM}/logs/vllm_${SLURM_JOB_ID}.log" 2>&1 &
VLLM_PID=$!
for i in $(seq 1 180); do
    if curl -sf "http://localhost:${PORT}/health" >/dev/null 2>&1; then echo "[vllm] поднялся за ${i}0 с"; break; fi
    if ! kill -0 ${VLLM_PID} 2>/dev/null; then echo "[vllm] движок умер, смотри лог vllm"; exit 1; fi
    sleep 10
done

LIMIT_ARG=""
if [ -n "${SMOKE_LIMIT:-}" ]; then LIMIT_ARG="--limit ${SMOKE_LIMIT}"; fi

for DATASET in ${DATASETS}; do
    OUTDIR=${OUT_ROOT}/${DATASET}/seed${SEED}
    QIDS_FILE=${QIDS_TMPL/DATASET/${DATASET}}
    mkdir -p "${OUTDIR}"
    echo "[прогон] ${MODEL_LABEL} ${DATASET} seed=${SEED} temp=${TEMPERATURE} окно=${MAX_MODEL_LEN} предел_хода=${MAX_TOKENS} раундов=${MAX_TOOL_ROUNDS} параллельно=${WORKERS} список=${QIDS_FILE} ${LIMIT_ARG}"
    WIKI_BASE_URL="http://127.0.0.1:${KPORT}" \
    WIKI_BACKEND=kiwix \
    PYTHONPATH="${RUNNER_SRC}" \
    python -u -m qwen_runner.cli \
        --dataset "${DATASET}" \
        --endpoint-url "http://localhost:${PORT}/v1" \
        --model "${SERVED_NAME}" \
        --no-reasoning-field \
        --multi-hop-tool-mode "${TOOL_MODE}" \
        --max-tool-rounds "${MAX_TOOL_ROUNDS}" \
        --max-tokens "${MAX_TOKENS}" \
        --send-max-tokens \
        --question-ids-file "${QIDS_FILE}" \
        --temperature "${TEMPERATURE}" \
        --seed "${SEED}" \
        --workers "${WORKERS}" \
        --output-dir "${OUTDIR}" \
        ${LIMIT_ARG}
done

echo "### ГОТОВО ret16_q36 ###"
