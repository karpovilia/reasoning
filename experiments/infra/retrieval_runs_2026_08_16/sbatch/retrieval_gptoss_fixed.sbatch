#!/bin/bash
#SBATCH --job-name=ret16_oss
#SBATCH --account=proj_1782
#SBATCH --partition=rocky
#SBATCH --constraint=type_e
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:30:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#
# Перепрогон поисковых трасс gpt-oss-20b с добавленным пределом хода.
#
# У этого раннера предела длины хода не было вовсе: в тело запроса уходило
# только поле reasoning_effort, которое задаёт манеру рассуждения, но не ставит
# жёсткой границы. По корпусу 15 августа дефект не бил — самый длинный ход занял
# 2152 токена на hotpotqa и 1870 на musique, то есть до 7000 не дотянул ни один
# ход. Предел ставится как страховка и ради одинакового протокола с Qwen.
#
# Окно диалога поднято с 16384 до 24576. Причина в самом пределе: движок
# отвергает запрос ответом 400, когда история плюс запрошенные 7000 токенов
# перерастают окно, а при окне 16384 под историю остаётся 9384 токена. По
# корпусу 15 августа наибольшая история достигала 9864 токенов на hotpotqa и
# 10660 на musique, то есть 1.1 процента трасс hotpotqa и 1.5 процента musique
# упёрлись бы в отказ. Окно 24576 оставляет под историю 17576 токенов и
# покрывает все снятые трассы.
#
# Число одновременных запросов поднято с 12 до 24. Пул кэша ключей и значений
# движок измерил как 1213503 токена, при окне 24576 это 49.4 запроса, значит 24
# запроса занимают половину пула.
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
RUNNER_SRC=${TRAJ}/gpt_oss_runner/src

export HF_HOME=${REPO}/.hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCHINDUCTOR_CACHE_DIR=${REPO}/.cache/inductor
export VLLM_CACHE_ROOT=${REPO}/.cache/vllm
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
# Словарь harmony у gpt-oss тянется из интернета, а счётные узлы офлайн:
# берём заранее прогретый кэш на Lustre.
export TIKTOKEN_RS_CACHE_DIR=${REPO}/.harmony_cache

MODEL_LABEL=gpt-oss-20b
SERVED_NAME=openai/gpt-oss-20b
SEED=${SEED:-101}
TEMPERATURE=${TEMPERATURE:-1.0}
DATASETS=${DATASETS:-"hotpotqa musique"}
OUT_ROOT=${OUT_ROOT:-${ARM}/${MODEL_LABEL}}
KIWIX_BIN=${REPO}/tools/kiwix-tools_linux-x86_64-3.8.2/kiwix-serve
ZIM_PATH="${ZIM_PATH:?путь к архиву энциклопедии задаётся переменной среды ZIM_PATH}"
WORKERS=${WORKERS:-24}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-24576}
MAX_TOKENS=${MAX_TOKENS:-7000}
MAX_TOOL_ROUNDS=${MAX_TOOL_ROUNDS:-12}
TOOL_MODE=${TOOL_MODE:-structured}
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
# Вызовы инструментов идут штатным интерфейсом OpenAI, поэтому движку нужен
# разбор вызовов: у gpt-oss он называется openai и опирается на разбор формата
# harmony. Текстовый протокол на своём сервере не работает — модель размышляет,
# но вызова не пишет.
vllm serve "${SERVED_NAME}" \
    --served-model-name "${SERVED_NAME}" \
    --port "${PORT}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --enable-auto-tool-choice \
    --tool-call-parser openai \
    --enforce-eager \
    --no-enable-chunked-prefill \
    --gpu-memory-utilization 0.90 \
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
    GPT_OSS_ENDPOINT_URL="http://localhost:${PORT}/v1" \
    GPT_OSS_MULTI_HOP_TOOL_MODE="${TOOL_MODE}" \
    PYTHONPATH="${RUNNER_SRC}" \
    python -u -m gpt_oss_runner.cli \
        --dataset "${DATASET}" \
        --endpoint-url "http://localhost:${PORT}/v1" \
        --model "${SERVED_NAME}" \
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

echo "### ГОТОВО ret16_oss ###"
