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

require_env() {
    local name="$1"
    local hint="$2"
    if [ -z "${!name:-}" ]; then
        echo "Missing required config: $name" >&2
        echo "$hint" >&2
        exit 1
    fi
}

if [ -n "${SFT_CUDA_VISIBLE_DEVICES:-}" ]; then
    export CUDA_VISIBLE_DEVICES="$SFT_CUDA_VISIBLE_DEVICES"
elif [ -n "${SFT_GPU_IDS:-}" ]; then
    export CUDA_VISIBLE_DEVICES="$SFT_GPU_IDS"
fi

require_env SFT_MODEL_PATH "Set SFT_MODEL_PATH in PROFILE_PATH or override it in the launch command."
require_env SFT_DATASET_DIR "Set SFT_DATASET_DIR in PROFILE_PATH or override it in the launch command."
require_env SFT_SAVE_DIR "Set SFT_SAVE_DIR in PROFILE_PATH or override it in the launch command."
require_env SFT_PROJECT_NAME "Set SFT_PROJECT_NAME in PROFILE_PATH or override it in the launch command."
require_env SFT_EXPERIMENT_NAME "Set SFT_EXPERIMENT_NAME in PROFILE_PATH or override it in the launch command."
require_env SFT_NUM_GPUS "Set SFT_NUM_GPUS in PROFILE_PATH or override it in the launch command."
require_env SFT_TRAIN_BATCH_SIZE "Set SFT_TRAIN_BATCH_SIZE in PROFILE_PATH or override it in the launch command."
require_env SFT_MICRO_BATCH_SIZE "Set SFT_MICRO_BATCH_SIZE in PROFILE_PATH or override it in the launch command."
require_env SFT_MAX_TOKEN_LEN_PER_GPU "Set SFT_MAX_TOKEN_LEN_PER_GPU in PROFILE_PATH or override it in the launch command."
require_env SFT_MAX_LENGTH "Set SFT_MAX_LENGTH in PROFILE_PATH or override it in the launch command."
require_env SFT_TOTAL_EPOCHS "Set SFT_TOTAL_EPOCHS in PROFILE_PATH or override it in the launch command."
require_env SFT_SAVE_FREQ "Set SFT_SAVE_FREQ in PROFILE_PATH or override it in the launch command."
require_env SFT_TEST_FREQ "Set SFT_TEST_FREQ in PROFILE_PATH or override it in the launch command."
require_env SFT_LR "Set SFT_LR in PROFILE_PATH or override it in the launch command."
require_env SFT_LR_SCHEDULER_TYPE "Set SFT_LR_SCHEDULER_TYPE in PROFILE_PATH or override it in the launch command."
require_env SFT_LOGGER "Set SFT_LOGGER in PROFILE_PATH or override it in the launch command."

SFT_MODEL_PATH="$SFT_MODEL_PATH"
SFT_DATASET_DIR="$SFT_DATASET_DIR"

TRAIN_FILES="$SFT_DATASET_DIR/train.parquet"
VAL_FILES="$SFT_DATASET_DIR/val.parquet"

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
    require_multivariate_etth1_metadata,
    validate_metadata_file,
)

metadata_path = sys.argv[1]
payload, _ = validate_metadata_file(
    metadata_path,
    expected_kind=(DATASET_KIND_RUNTIME_SFT_PARQUET, DATASET_KIND_TEACHER_CURATED_SFT),
)
require_multivariate_etth1_metadata(payload, metadata_path=metadata_path)
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
if [ "${PRINT_CMD_ONLY:-0}" != "1" ]; then
python3 - "$TRAIN_FILES" "$VAL_FILES" "$SFT_METADATA_PATH" <<'PY'
import sys
import json
import pandas as pd

from recipe.time_series_forecast.build_etth1_sft_dataset import _summarize_paper_turn3_protocol

metadata_path = sys.argv[3]
with open(metadata_path, "r", encoding="utf-8") as handle:
    metadata = json.load(handle)
allow_no_refinement = str(metadata.get("sft_stage_mode", "") or "").strip().lower() == "routing_only"


def inspect(path: str) -> None:
    frame = pd.read_parquet(path)
    summary = _summarize_paper_turn3_protocol(frame)
    checked = int(summary.get("turn3_protocol_checked_count", 0))
    failures = summary.get("turn3_protocol_invalid_examples", [])[:5]
    if checked <= 0 and allow_no_refinement:
        print(f"[SFT TURN3 PROTOCOL] path={path} checked=0 total={len(frame)} status=skipped(routing_only)")
        return
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
fi
SAVE_DIR="$SFT_SAVE_DIR"
PROJECT_NAME="$SFT_PROJECT_NAME"
EXPERIMENT_NAME="$SFT_EXPERIMENT_NAME"
NUM_GPUS="$SFT_NUM_GPUS"
TRAIN_BATCH_SIZE="$SFT_TRAIN_BATCH_SIZE"
MICRO_BATCH_SIZE="$SFT_MICRO_BATCH_SIZE"
MAX_TOKEN_LEN_PER_GPU="$SFT_MAX_TOKEN_LEN_PER_GPU"
MAX_LENGTH="$SFT_MAX_LENGTH"
TOTAL_EPOCHS="$SFT_TOTAL_EPOCHS"
SAVE_FREQ="$SFT_SAVE_FREQ"
TEST_FREQ="$SFT_TEST_FREQ"
LR="$SFT_LR"
LR_SCHEDULER_TYPE="$SFT_LR_SCHEDULER_TYPE"
LOGGER="$SFT_LOGGER"

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
    "model.path=${SFT_MODEL_PATH}"
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
