#!/bin/bash
set -euo pipefail

if [ "${TRACE:-0}" = "1" ]; then
    set -x
fi

PROJECT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"

resolve_profile_path() {
    local candidate="$1"
    if [ -z "$candidate" ]; then
        return 1
    fi
    if [ -f "$candidate" ]; then
        printf '%s\n' "$candidate"
        return 0
    fi
    if [ -f "$PROJECT_DIR/$candidate" ]; then
        printf '%s\n' "$PROJECT_DIR/$candidate"
        return 0
    fi
    return 1
}

PROFILE_PATH="${PROFILE_PATH:-}"
if [ -n "$PROFILE_PATH" ]; then
    RESOLVED_PROFILE_PATH="$(resolve_profile_path "$PROFILE_PATH")" || {
        echo "PROFILE_PATH not found: $PROFILE_PATH"
        exit 1
    }
    # shellcheck disable=SC1090
    source "$RESOLVED_PROFILE_PATH"
fi

MODEL_PATH="${MODEL_PATH:-}"
if [ -z "${MODEL_PATH}" ]; then
    echo "MODEL_PATH must point to the base chat model or tokenizer-compatible checkpoint"
    exit 1
fi

TRAIN_FILES="${TRAIN_FILES:-${SFT_TRAIN_FILES:-$PROJECT_DIR/dataset/ett_sft_etth1_runtime_ot/train.parquet}}"
VAL_FILES="${VAL_FILES:-${SFT_VAL_FILES:-$PROJECT_DIR/dataset/ett_sft_etth1_runtime_ot/val.parquet}}"
if [ ! -f "${TRAIN_FILES}" ]; then
    echo "SFT train parquet not found: ${TRAIN_FILES}"
    echo "Generate or restore the SFT dataset first."
    exit 1
fi
if [ ! -f "${VAL_FILES}" ]; then
    echo "SFT val parquet not found: ${VAL_FILES}"
    echo "Generate or restore the SFT dataset first."
    exit 1
fi
SAVE_DIR="${SAVE_DIR:-${SFT_SAVE_DIR:-$PROJECT_DIR/checkpoints/time_series_forecast_sft}}"
PROJECT_NAME="${PROJECT_NAME:-${SFT_PROJECT_NAME:-TimeSeriesForecast-SFT}}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-${SFT_EXPERIMENT_NAME:-qwen3-1.7b-etth1-ot-sft}}"
NUM_GPUS="${NUM_GPUS:-${SFT_NUM_GPUS:-1}}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-${SFT_TRAIN_BATCH_SIZE:-32}}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-${SFT_MICRO_BATCH_SIZE:-2}}"
MAX_TOKEN_LEN_PER_GPU="${MAX_TOKEN_LEN_PER_GPU:-${SFT_MAX_TOKEN_LEN_PER_GPU:-8192}}"
MAX_LENGTH="${MAX_LENGTH:-${SFT_MAX_LENGTH:-8192}}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-${SFT_TOTAL_EPOCHS:-1}}"
SAVE_FREQ="${SAVE_FREQ:-${SFT_SAVE_FREQ:-after_each_epoch}}"
TEST_FREQ="${TEST_FREQ:-${SFT_TEST_FREQ:-after_each_epoch}}"
LR="${LR:-${SFT_LR:-1e-5}}"
LR_SCHEDULER_TYPE="${LR_SCHEDULER_TYPE:-${SFT_LR_SCHEDULER_TYPE:-cosine}}"
LOGGER="${LOGGER:-${SFT_LOGGER:-console}}"

if command -v torchrun >/dev/null 2>&1; then
    LAUNCHER=(torchrun --standalone --nnodes=1 --nproc_per_node="${NUM_GPUS}")
else
    LAUNCHER=(python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node="${NUM_GPUS}")
fi

CMD=(
    "${LAUNCHER[@]}"
    -m verl.trainer.sft_trainer
    "data.train_files=${TRAIN_FILES}"
    "data.val_files=${VAL_FILES}"
    "data.messages_key=messages"
    "data.tools_key=tools"
    "data.enable_thinking_key=enable_thinking"
    "data.ignore_input_ids_mismatch=true"
    "data.pad_mode=no_padding"
    "data.max_length=${MAX_LENGTH}"
    "data.max_token_len_per_gpu=${MAX_TOKEN_LEN_PER_GPU}"
    "data.train_batch_size=${TRAIN_BATCH_SIZE}"
    "data.micro_batch_size_per_gpu=${MICRO_BATCH_SIZE}"
    "data.truncation=error"
    "model.path=${MODEL_PATH}"
    "optim.lr=${LR}"
    "optim.lr_scheduler_type=${LR_SCHEDULER_TYPE}"
    "checkpoint.save_contents=['model','optimizer','extra','hf_model']"
    "trainer.default_local_dir=${SAVE_DIR}"
    "trainer.project_name=${PROJECT_NAME}"
    "trainer.experiment_name=${EXPERIMENT_NAME}"
    "trainer.logger=${LOGGER}"
    "trainer.total_epochs=${TOTAL_EPOCHS}"
    "trainer.save_freq=${SAVE_FREQ}"
    "trainer.test_freq=${TEST_FREQ}"
    "trainer.n_gpus_per_node=${NUM_GPUS}"
)

if declare -p SFT_EXTRA_ARGS >/dev/null 2>&1; then
    CMD+=("${SFT_EXTRA_ARGS[@]}")
fi

CMD+=("$@")

if [ "${PRINT_CMD_ONLY:-0}" = "1" ]; then
    printf '%q ' "${CMD[@]}"
    printf '\n'
    exit 0
fi

"${CMD[@]}"
