#!/bin/bash
set -euo pipefail

if [ "${TRACE:-0}" = "1" ]; then
    set -x
fi

PROJECT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"
DEFAULT_PROFILE_PATH="$PROJECT_DIR/examples/time_series_forecast/configs/etth1_ot_qwen3_gpu012.sh"

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

PROFILE_PATH="${PROFILE_PATH:-$DEFAULT_PROFILE_PATH}"
RESOLVED_PROFILE_PATH="$(resolve_profile_path "$PROFILE_PATH")" || {
    echo "PROFILE_PATH not found: $PROFILE_PATH"
    exit 1
}
# shellcheck disable=SC1090
source "$RESOLVED_PROFILE_PATH"

if [ -n "${SFT_CUDA_VISIBLE_DEVICES:-}" ]; then
    export CUDA_VISIBLE_DEVICES="$SFT_CUDA_VISIBLE_DEVICES"
elif [ -n "${SFT_GPU_IDS:-}" ]; then
    export CUDA_VISIBLE_DEVICES="$SFT_GPU_IDS"
fi

MODEL_PATH="${MODEL_PATH:-}"
if [ -z "${MODEL_PATH}" ]; then
    echo "MODEL_PATH must point to the base chat model or tokenizer-compatible checkpoint"
    exit 1
fi

resolve_consistent_env_value() {
    local label="$1"
    shift
    local resolved=""
    local env_name
    local candidate
    for env_name in "$@"; do
        candidate="${!env_name:-}"
        if [ -z "$candidate" ]; then
            continue
        fi
        if [ -n "$resolved" ] && [ "$candidate" != "$resolved" ]; then
            echo "Conflicting ${label} values:" >&2
            printf '  %s=%s\n' "$env_name" "$candidate" >&2
            printf '  resolved=%s\n' "$resolved" >&2
            exit 1
        fi
        resolved="$candidate"
    done
    printf '%s\n' "$resolved"
}

SFT_DATASET_DIR="${SFT_DATASET_DIR:-}"
TRAIN_FILES_EXPLICIT="$(resolve_consistent_env_value 'SFT train parquet' TRAIN_FILES SFT_TRAIN_FILES)"
VAL_FILES_EXPLICIT="$(resolve_consistent_env_value 'SFT val parquet' VAL_FILES SFT_VAL_FILES)"

if [ -n "$SFT_DATASET_DIR" ]; then
    DERIVED_TRAIN_FILES="$SFT_DATASET_DIR/train.parquet"
    DERIVED_VAL_FILES="$SFT_DATASET_DIR/val.parquet"
    if [ -n "$TRAIN_FILES_EXPLICIT" ] && [ "$TRAIN_FILES_EXPLICIT" != "$DERIVED_TRAIN_FILES" ]; then
        echo "Conflicting SFT dataset specification for train split." >&2
        echo "SFT_DATASET_DIR implies: $DERIVED_TRAIN_FILES" >&2
        echo "Explicit train file: $TRAIN_FILES_EXPLICIT" >&2
        exit 1
    fi
    if [ -n "$VAL_FILES_EXPLICIT" ] && [ "$VAL_FILES_EXPLICIT" != "$DERIVED_VAL_FILES" ]; then
        echo "Conflicting SFT dataset specification for val split." >&2
        echo "SFT_DATASET_DIR implies: $DERIVED_VAL_FILES" >&2
        echo "Explicit val file: $VAL_FILES_EXPLICIT" >&2
        exit 1
    fi
    TRAIN_FILES="${TRAIN_FILES_EXPLICIT:-$DERIVED_TRAIN_FILES}"
    VAL_FILES="${VAL_FILES_EXPLICIT:-$DERIVED_VAL_FILES}"
else
    TRAIN_FILES="$TRAIN_FILES_EXPLICIT"
    VAL_FILES="$VAL_FILES_EXPLICIT"
fi

if [ -z "${TRAIN_FILES}" ] || [ -z "${VAL_FILES}" ]; then
    echo "SFT dataset path is required." >&2
    echo "Set SFT_DATASET_DIR to a single dataset directory, or set both TRAIN_FILES and VAL_FILES explicitly." >&2
    exit 1
fi

TRAIN_DIR="$(cd "$(dirname "$TRAIN_FILES")" && pwd)"
VAL_DIR="$(cd "$(dirname "$VAL_FILES")" && pwd)"
if [ "$TRAIN_DIR" != "$VAL_DIR" ]; then
    echo "SFT train/val parquet must come from the same dataset directory." >&2
    echo "train dir: $TRAIN_DIR" >&2
    echo "val dir:   $VAL_DIR" >&2
    exit 1
fi

SFT_METADATA_PATH="$TRAIN_DIR/metadata.json"
if [ ! -f "$SFT_METADATA_PATH" ]; then
    echo "SFT dataset metadata not found: $SFT_METADATA_PATH" >&2
    echo "Use a dataset directory produced by the ETTh1 SFT builders, or pass the correct parquet paths." >&2
    exit 1
fi
python3 - "$SFT_METADATA_PATH" <<'PY'
import sys
from recipe.time_series_forecast.dataset_identity import (
    DATASET_KIND_RUNTIME_SFT_PARQUET,
    DATASET_KIND_TEACHER_CURATED_SFT,
    validate_metadata_file,
)

metadata_path = sys.argv[1]
payload, _ = validate_metadata_file(
    metadata_path,
    expected_kind=(DATASET_KIND_RUNTIME_SFT_PARQUET, DATASET_KIND_TEACHER_CURATED_SFT),
)
print(
    f"[SFT DATASET] kind={payload.get('dataset_kind')} stage={payload.get('pipeline_stage', '')} "
    f"metadata={metadata_path}"
)
PY

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
python3 - "$TRAIN_FILES" "$VAL_FILES" <<'PY'
import sys
import pandas as pd

from recipe.time_series_forecast.validate_turn3_format import (
    check_paper_turn3_protocol,
    get_last_assistant_content,
    record_requires_paper_turn3_protocol,
)


def inspect(path: str) -> None:
    frame = pd.read_parquet(path)
    failures = []
    checked = 0
    for row_idx, row in frame.iterrows():
        record = {
            "messages": row["messages"],
            "turn_stage": row.get("turn_stage", ""),
            "paper_turn3_required": row.get("paper_turn3_required", None),
        }
        if not record_requires_paper_turn3_protocol(record):
            continue
        checked += 1
        content = get_last_assistant_content(record)
        expected_len = int(row.get("forecast_horizon", 96) or 96)
        ok, reason, pred_len = check_paper_turn3_protocol(content, expected_len=expected_len)
        if not ok:
            failures.append((int(row_idx), reason, int(pred_len)))
            if len(failures) >= 5:
                break
    if checked <= 0:
        raise SystemExit(f"SFT parquet {path} does not contain any refinement rows to validate.")
    if failures:
        raise SystemExit(
            f"SFT parquet {path} is not paper-aligned turn-3 protocol. "
            f"First failures: {failures}"
        )
    print(f"[SFT TURN3 PROTOCOL] path={path} checked={checked} total={len(frame)} status=ok")


inspect(sys.argv[1])
inspect(sys.argv[2])
PY
SAVE_DIR="${SAVE_DIR:-${SFT_SAVE_DIR:-$PROJECT_DIR/artifacts/checkpoints/sft/time_series_forecast_sft}}"
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
