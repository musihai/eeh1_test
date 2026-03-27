#!/bin/bash
set -euo pipefail

if [ "${TRACE:-0}" = "1" ]; then
    set -x
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
DEFAULT_PROFILE_PATH="$SCRIPT_DIR/configs/etth1_ot_qwen3_gpu012.sh"
ORIGINAL_RL_ROLLOUT_GPU_MEMORY_UTILIZATION="${RL_ROLLOUT_GPU_MEMORY_UTILIZATION-__unset__}"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/rl_launch_utils.sh"

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

if [ "${PRINT_CMD_ONLY:-0}" != "1" ]; then
    trap cleanup_ray_and_stale_vllm EXIT
fi

resolve_transformers_model_dir() {
    python3 - "$1" <<'PY'
import sys

from recipe.time_series_forecast.model_path_utils import resolve_transformers_model_dir

print(resolve_transformers_model_dir(sys.argv[1]))
PY
}

INITIAL_MODEL_PATH="${RL_MODEL_PATH:-}"
if [ -z "${INITIAL_MODEL_PATH}" ]; then
    echo "RL_MODEL_PATH is required for curriculum RL." >&2
    exit 1
fi
INITIAL_MODEL_PATH="$(resolve_transformers_model_dir "$INITIAL_MODEL_PATH")" || {
    echo "Initial RL model path is not loadable: $INITIAL_MODEL_PATH" >&2
    exit 1
}

CURRICULUM_DATASET_DIR="${RL_CURRICULUM_DATASET_DIR:-}"
VAL_FILES="$CURRICULUM_DATASET_DIR/val.jsonl"
PHASES="${RL_CURRICULUM_PHASES:-stage1,stage12,stage123}"
BASE_EXP_NAME="${RL_EXP_NAME:-etth1_ot_qwen3_1_7b_rl_paper_20260326}"
BASE_LOCAL_DIR="${RL_TRAINER_LOCAL_DIR:-$PROJECT_DIR/artifacts/checkpoints/rl/$BASE_EXP_NAME}"
RUN_MODE="${RUN_MODE:-train}"

resolve_phase_rollout_gpu_memory_utilization() {
    local phase="$1"

    if [ "$ORIGINAL_RL_ROLLOUT_GPU_MEMORY_UTILIZATION" != "__unset__" ]; then
        printf '%s\n' "$ORIGINAL_RL_ROLLOUT_GPU_MEMORY_UTILIZATION"
        return 0
    fi

    case "$phase" in
        stage1)
            printf '%s\n' "${RL_CURRICULUM_STAGE1_ROLLOUT_GPU_MEMORY_UTILIZATION:-0.25}"
            ;;
        stage12|stage123)
            printf '%s\n' "${RL_CURRICULUM_LATER_ROLLOUT_GPU_MEMORY_UTILIZATION:-0.20}"
            ;;
        *)
            printf '%s\n' "${RL_CURRICULUM_LATER_ROLLOUT_GPU_MEMORY_UTILIZATION:-0.20}"
            ;;
    esac
}

if [ -z "$CURRICULUM_DATASET_DIR" ]; then
    echo "RL_CURRICULUM_DATASET_DIR is required for curriculum RL." >&2
    exit 1
fi
if [ ! -d "$CURRICULUM_DATASET_DIR" ]; then
    echo "Curriculum RL dataset directory not found: $CURRICULUM_DATASET_DIR" >&2
    exit 1
fi
if [ ! -f "$VAL_FILES" ]; then
    echo "Curriculum RL val jsonl not found: $VAL_FILES" >&2
    exit 1
fi

resolve_latest_actor_hf_model() {
    local phase_dir="$1"
    local latest_file="$phase_dir/latest_checkpointed_iteration.txt"
    local latest_step=""
    local candidate=""

    if [ -f "$latest_file" ]; then
        latest_step="$(tr -d '[:space:]' < "$latest_file")"
        if [ -n "$latest_step" ]; then
            candidate="$phase_dir/global_step_${latest_step}/actor/huggingface"
            if [ -d "$candidate" ]; then
                resolve_transformers_model_dir "$candidate"
                return 0
            fi
        fi
    fi

    candidate="$(find "$phase_dir" -maxdepth 3 -type d -path "*/global_step_*/actor/huggingface" | sort -V | tail -1)"
    if [ -n "$candidate" ]; then
        resolve_transformers_model_dir "$candidate"
        return 0
    fi
    return 1
}

PHASE_LIST="$(python3 - "$PHASES" <<'PY'
import sys
from recipe.time_series_forecast.curriculum_utils import parse_curriculum_phase_list

phases = parse_curriculum_phase_list(sys.argv[1])
if not phases:
    raise SystemExit("RL_CURRICULUM_PHASES is empty.")
print("\n".join(phases))
PY
)"

CURRENT_MODEL_PATH="$INITIAL_MODEL_PATH"
PREVIOUS_PHASE_DIR=""

while IFS= read -r phase; do
    [ -n "$phase" ] || continue

    PHASE_TRAIN_FILE="$CURRICULUM_DATASET_DIR/train_${phase}.jsonl"
    if [ ! -f "$PHASE_TRAIN_FILE" ]; then
        echo "Missing curriculum train file for phase $phase: $PHASE_TRAIN_FILE" >&2
        exit 1
    fi

    PHASE_EXP_NAME="${BASE_EXP_NAME}_${phase}"
    PHASE_LOCAL_DIR="${BASE_LOCAL_DIR}/${phase}"
    PHASE_ROLLOUT_GPU_MEMORY_UTILIZATION="$(resolve_phase_rollout_gpu_memory_utilization "$phase")"

    if [ "${PRINT_CMD_ONLY:-0}" = "1" ] && [ -n "$PREVIOUS_PHASE_DIR" ]; then
        CURRENT_MODEL_PATH="${PREVIOUS_PHASE_DIR}/global_step_<latest>/actor/huggingface"
    fi

    echo "[CURRICULUM RL] phase=$phase"
    echo "  train=$PHASE_TRAIN_FILE"
    echo "  val=$VAL_FILES"
    echo "  model=$CURRENT_MODEL_PATH"
    echo "  save_dir=$PHASE_LOCAL_DIR"
    echo "  rollout_gpu_memory_utilization=$PHASE_ROLLOUT_GPU_MEMORY_UTILIZATION"

    if [ "${PRINT_CMD_ONLY:-0}" != "1" ]; then
        cleanup_ray_and_stale_vllm
    fi

    RL_MODEL_PATH="$CURRENT_MODEL_PATH" \
    RL_CURRICULUM_DATASET_DIR="$CURRICULUM_DATASET_DIR" \
    RL_CURRICULUM_PHASE="$phase" \
    RL_EXP_NAME="$PHASE_EXP_NAME" \
    RL_TRAINER_LOCAL_DIR="$PHASE_LOCAL_DIR" \
    RL_ROLLOUT_GPU_MEMORY_UTILIZATION="$PHASE_ROLLOUT_GPU_MEMORY_UTILIZATION" \
    RUN_MODE="$RUN_MODE" \
    bash "$SCRIPT_DIR/run_qwen3-1.7B.sh" "$@"

    PREVIOUS_PHASE_DIR="$PHASE_LOCAL_DIR"
    if [ "${PRINT_CMD_ONLY:-0}" = "1" ]; then
        continue
    fi

    CURRENT_MODEL_PATH="$(resolve_latest_actor_hf_model "$PHASE_LOCAL_DIR")" || {
        echo "Could not find actor/huggingface checkpoint after phase $phase in $PHASE_LOCAL_DIR" >&2
        exit 1
    }
done <<< "$PHASE_LIST"

if [ "${PRINT_CMD_ONLY:-0}" != "1" ]; then
    echo "[CURRICULUM RL] final_actor_hf_model=$CURRENT_MODEL_PATH"
fi
