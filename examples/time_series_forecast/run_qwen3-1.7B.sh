#!/bin/bash
set -euo pipefail

if [ "${TRACE:-0}" = "1" ]; then
    set -x
fi

export VLLM_USE_V1=1
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HYDRA_FULL_ERROR=1
# Ray 2.52.0 + current host opentelemetry stack occasionally crashes in worker startup.
# Disable open telemetry by default for this project; users can still re-enable explicitly.
export RAY_enable_open_telemetry="${RAY_enable_open_telemetry:-0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"
DEFAULT_PROFILE_PATH="$SCRIPT_DIR/configs/etth1_ot_qwen3_gpu012.sh"
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

enable_chain_debug_if_requested() {
    if [ -z "${DEBUG_CHAIN+x}" ]; then
        if [ "${RUN_MODE:-train}" = "train" ]; then
            DEBUG_CHAIN=1
        else
            DEBUG_CHAIN=0
        fi
    fi

    if [ "$DEBUG_CHAIN" = "1" ] || [ "${DEBUG_CHAIN,,}" = "true" ]; then
        export TS_CHAIN_DEBUG=1
        export TS_CHAIN_DEBUG_FILE="${TS_CHAIN_DEBUG_FILE:-$PROJECT_DIR/logs/debug/ts_chain_debug.jsonl}"
        export TS_MIN_DEBUG_DIR="${TS_MIN_DEBUG_DIR:-$PROJECT_DIR/logs/debug}"
        mkdir -p "$(dirname "$TS_CHAIN_DEBUG_FILE")" "$TS_MIN_DEBUG_DIR"
        echo "[CHAIN DEBUG] enabled, writing to: $TS_CHAIN_DEBUG_FILE"
        echo "[MIN DEBUG] validation debug dir: $TS_MIN_DEBUG_DIR"
    fi
}

enable_chain_debug_if_requested

if is_true "${RL_CLEANUP_STALE_VLLM_BEFORE_LAUNCH:-0}"; then
    cleanup_stale_vllm_processes
fi

require_env() {
    local name="$1"
    local hint="$2"
    if [ -z "${!name:-}" ]; then
        echo "Missing required config: $name" >&2
        echo "$hint" >&2
        exit 1
    fi
}

resolve_transformers_model_dir() {
    python3 - "$1" <<'PY'
import sys

from recipe.time_series_forecast.model_path_utils import resolve_transformers_model_dir

print(resolve_transformers_model_dir(sys.argv[1]))
PY
}

require_env RL_CONFIG_PATH "Set RL_CONFIG_PATH in PROFILE_PATH or override it in the launch command."
require_env RL_CURRICULUM_DATASET_DIR "Set RL_CURRICULUM_DATASET_DIR in PROFILE_PATH or override it in the launch command."
require_env RL_REWARD_FN_PATH "Set RL_REWARD_FN_PATH in PROFILE_PATH or override it in the launch command."
require_env RL_REWARD_FN_NAME "Set RL_REWARD_FN_NAME in PROFILE_PATH or override it in the launch command."
require_env RL_MODEL_PATH "Set RL_MODEL_PATH to a loadable HuggingFace/Transformers checkpoint before launch."
require_env RL_PROJECT_NAME "Set RL_PROJECT_NAME in PROFILE_PATH or override it in the launch command."
require_env RL_EXP_NAME "Set RL_EXP_NAME in PROFILE_PATH or override it in the launch command."
require_env RL_TRAINER_LOCAL_DIR "Set RL_TRAINER_LOCAL_DIR in PROFILE_PATH or override it in the launch command."
require_env RL_NNODES "Set RL_NNODES in PROFILE_PATH or override it in the launch command."
require_env RL_NUM_GPUS "Set RL_NUM_GPUS in PROFILE_PATH or override it in the launch command."
require_env RL_TRAIN_BATCH_SIZE "Set RL_TRAIN_BATCH_SIZE in PROFILE_PATH or override it in the launch command."
require_env RL_PPO_MINI_BATCH_SIZE "Set RL_PPO_MINI_BATCH_SIZE in PROFILE_PATH or override it in the launch command."
require_env RL_PPO_MICRO_BATCH_SIZE "Set RL_PPO_MICRO_BATCH_SIZE in PROFILE_PATH or override it in the launch command."
require_env RL_LOGPROB_MICRO_BATCH_SIZE "Set RL_LOGPROB_MICRO_BATCH_SIZE in PROFILE_PATH or override it in the launch command."
require_env RL_ACTOR_PARAM_OFFLOAD "Set RL_ACTOR_PARAM_OFFLOAD in PROFILE_PATH or override it in the launch command."
require_env RL_ACTOR_OPTIMIZER_OFFLOAD "Set RL_ACTOR_OPTIMIZER_OFFLOAD in PROFILE_PATH or override it in the launch command."
require_env RL_REF_PARAM_OFFLOAD "Set RL_REF_PARAM_OFFLOAD in PROFILE_PATH or override it in the launch command."
require_env RL_ROLLOUT_GPU_MEMORY_UTILIZATION "Set RL_ROLLOUT_GPU_MEMORY_UTILIZATION in PROFILE_PATH or override it in the launch command."
require_env RL_ROLLOUT_LOAD_FORMAT "Set RL_ROLLOUT_LOAD_FORMAT in PROFILE_PATH or override it in the launch command."
require_env RL_ROLLOUT_N "Set RL_ROLLOUT_N in PROFILE_PATH or override it in the launch command."
require_env RL_ROLLOUT_TP "Set RL_ROLLOUT_TP in PROFILE_PATH or override it in the launch command."
require_env RL_FSDP_MODEL_DTYPE "Set RL_FSDP_MODEL_DTYPE in PROFILE_PATH or override it in the launch command."
require_env RL_FSDP_USE_TORCH_COMPILE "Set RL_FSDP_USE_TORCH_COMPILE in PROFILE_PATH or override it in the launch command."
require_env RL_MAX_PROMPT_LENGTH "Set RL_MAX_PROMPT_LENGTH in PROFILE_PATH or override it in the launch command."
require_env RL_MAX_RESPONSE_LENGTH "Set RL_MAX_RESPONSE_LENGTH in PROFILE_PATH or override it in the launch command."
require_env RL_DATALOADER_NUM_WORKERS "Set RL_DATALOADER_NUM_WORKERS in PROFILE_PATH or override it in the launch command."
require_env RL_ACTOR_MAX_TOKEN_LEN_PER_GPU "Set RL_ACTOR_MAX_TOKEN_LEN_PER_GPU in PROFILE_PATH or override it in the launch command."
require_env RL_ROLLOUT_MAX_MODEL_LEN "Set RL_ROLLOUT_MAX_MODEL_LEN in PROFILE_PATH or override it in the launch command."
require_env RL_ROLLOUT_MAX_BATCHED_TOKENS "Set RL_ROLLOUT_MAX_BATCHED_TOKENS in PROFILE_PATH or override it in the launch command."
require_env RL_ROLLOUT_MAX_NUM_SEQS "Set RL_ROLLOUT_MAX_NUM_SEQS in PROFILE_PATH or override it in the launch command."
require_env RL_ROLLOUT_ENFORCE_EAGER "Set RL_ROLLOUT_ENFORCE_EAGER in PROFILE_PATH or override it in the launch command."
require_env RL_ROLLOUT_ENABLE_PREFIX_CACHING "Set RL_ROLLOUT_ENABLE_PREFIX_CACHING in PROFILE_PATH or override it in the launch command."
require_env RL_ROLLOUT_ENABLE_CHUNKED_PREFILL "Set RL_ROLLOUT_ENABLE_CHUNKED_PREFILL in PROFILE_PATH or override it in the launch command."
require_env RL_ROLLOUT_FREE_CACHE_ENGINE "Set RL_ROLLOUT_FREE_CACHE_ENGINE in PROFILE_PATH or override it in the launch command."
require_env RL_SAVE_FREQ "Set RL_SAVE_FREQ in PROFILE_PATH or override it in the launch command."
require_env RL_TEST_FREQ "Set RL_TEST_FREQ in PROFILE_PATH or override it in the launch command."
require_env RL_TOTAL_EPOCHS "Set RL_TOTAL_EPOCHS in PROFILE_PATH or override it in the launch command."
require_env RL_LOGGER "Set RL_LOGGER in PROFILE_PATH or override it in the launch command."
require_env RL_LR "Set RL_LR in PROFILE_PATH or override it in the launch command."
require_env RL_LR_SCHEDULER_TYPE "Set RL_LR_SCHEDULER_TYPE in PROFILE_PATH or override it in the launch command."
require_env RL_KL_LOSS_COEF "Set RL_KL_LOSS_COEF in PROFILE_PATH or override it in the launch command."
require_env RL_ENTROPY_COEFF "Set RL_ENTROPY_COEFF in PROFILE_PATH or override it in the launch command."
require_env RL_NORM_ADV_BY_STD_IN_GRPO "Set RL_NORM_ADV_BY_STD_IN_GRPO in PROFILE_PATH or override it in the launch command."
require_env RL_TEMPERATURE "Set RL_TEMPERATURE in PROFILE_PATH or override it in the launch command."
require_env RL_VAL_TEMPERATURE "Set RL_VAL_TEMPERATURE in PROFILE_PATH or override it in the launch command."

CONFIG_PATH="$RL_CONFIG_PATH"
RL_DATASET_DIR="$RL_CURRICULUM_DATASET_DIR"
REWARD_FN_PATH="$RL_REWARD_FN_PATH"
REWARD_FN_NAME="$RL_REWARD_FN_NAME"
MODEL_PATH="$RL_MODEL_PATH"
CURRICULUM_PHASE="${RL_CURRICULUM_PHASE:-}"
if [ "${PRINT_CMD_ONLY:-0}" != "1" ] || [[ "$MODEL_PATH" != *"<latest>"* ]]; then
    MODEL_PATH="$(resolve_transformers_model_dir "$MODEL_PATH")" || {
        echo "RL model path is not loadable: $MODEL_PATH" >&2
        exit 1
    }
fi
if [ ! -d "$RL_DATASET_DIR" ]; then
    echo "RL dataset directory not found: $RL_DATASET_DIR" >&2
    exit 1
fi
TRAIN_FILES="$RL_DATASET_DIR/train.jsonl"
VAL_FILES="$RL_DATASET_DIR/val.jsonl"
if [ ! -f "$VAL_FILES" ]; then
    echo "RL val jsonl not found: $VAL_FILES" >&2
    exit 1
fi
RL_METADATA_PATH="$RL_DATASET_DIR/metadata.json"
python3 - "$RL_METADATA_PATH" <<'PY'
import sys
from recipe.time_series_forecast.dataset_identity import (
    DATASET_KIND_RL_JSONL,
    require_multivariate_etth1_metadata,
    validate_metadata_file,
)

metadata_path = sys.argv[1]
payload, _ = validate_metadata_file(metadata_path, expected_kind=DATASET_KIND_RL_JSONL)
require_multivariate_etth1_metadata(payload, metadata_path=metadata_path)
print(
    f"[RL DATASET] kind={payload.get('dataset_kind')} stage={payload.get('pipeline_stage', '')} "
    f"metadata={metadata_path}"
)
PY
TRAIN_FILES="$(python3 - "$TRAIN_FILES" "$RL_METADATA_PATH" "$RUN_MODE" "$CURRICULUM_PHASE" <<'PY'
import sys
from recipe.time_series_forecast.curriculum_utils import resolve_curriculum_train_file
from recipe.time_series_forecast.dataset_identity import load_metadata

train_file, metadata_path, run_mode, curriculum_phase = sys.argv[1:5]
payload = load_metadata(metadata_path)
resolved = resolve_curriculum_train_file(
    train_file=train_file,
    metadata_payload=payload,
    run_mode=run_mode,
    curriculum_phase=curriculum_phase,
    allow_noncurriculum_train=False,
)
print(str(resolved))
PY
)"
if [ ! -f "$TRAIN_FILES" ]; then
    echo "Resolved RL train jsonl not found: $TRAIN_FILES" >&2
    exit 1
fi
if [ -n "$CURRICULUM_PHASE" ]; then
    echo "[RL CURRICULUM] phase=$CURRICULUM_PHASE train=$TRAIN_FILES"
fi

PROJECT_NAME="$RL_PROJECT_NAME"
EXP_NAME="$RL_EXP_NAME"
TRAINER_LOCAL_DIR="$RL_TRAINER_LOCAL_DIR"
NNODES="$RL_NNODES"
NUM_GPUS="$RL_NUM_GPUS"
TRAIN_BATCH_SIZE="$RL_TRAIN_BATCH_SIZE"
PPO_MINI_BATCH_SIZE="$RL_PPO_MINI_BATCH_SIZE"
PPO_MICRO_BATCH_SIZE="$RL_PPO_MICRO_BATCH_SIZE"
LOGPROB_MICRO_BATCH_SIZE="$RL_LOGPROB_MICRO_BATCH_SIZE"
ACTOR_PARAM_OFFLOAD="$RL_ACTOR_PARAM_OFFLOAD"
ACTOR_OPTIMIZER_OFFLOAD="$RL_ACTOR_OPTIMIZER_OFFLOAD"
REF_PARAM_OFFLOAD="$RL_REF_PARAM_OFFLOAD"
ROLLOUT_GPU_MEMORY_UTILIZATION="$RL_ROLLOUT_GPU_MEMORY_UTILIZATION"
ROLLOUT_LOAD_FORMAT="$RL_ROLLOUT_LOAD_FORMAT"
ROLLOUT_N="$RL_ROLLOUT_N"
ROLLOUT_TP="$RL_ROLLOUT_TP"
FSDP_MODEL_DTYPE="$RL_FSDP_MODEL_DTYPE"
FSDP_USE_TORCH_COMPILE="$RL_FSDP_USE_TORCH_COMPILE"
MAX_PROMPT_LENGTH="$RL_MAX_PROMPT_LENGTH"
MAX_RESPONSE_LENGTH="$RL_MAX_RESPONSE_LENGTH"
DATALOADER_NUM_WORKERS="$RL_DATALOADER_NUM_WORKERS"
ACTOR_MAX_TOKEN_LEN_PER_GPU="$RL_ACTOR_MAX_TOKEN_LEN_PER_GPU"
ROLLOUT_MAX_MODEL_LEN="$RL_ROLLOUT_MAX_MODEL_LEN"
ROLLOUT_MAX_BATCHED_TOKENS="$RL_ROLLOUT_MAX_BATCHED_TOKENS"
ROLLOUT_MAX_NUM_SEQS="$RL_ROLLOUT_MAX_NUM_SEQS"
ROLLOUT_ENFORCE_EAGER="$RL_ROLLOUT_ENFORCE_EAGER"
ROLLOUT_ENABLE_PREFIX_CACHING="$RL_ROLLOUT_ENABLE_PREFIX_CACHING"
ROLLOUT_ENABLE_CHUNKED_PREFILL="$RL_ROLLOUT_ENABLE_CHUNKED_PREFILL"
ROLLOUT_FREE_CACHE_ENGINE="$RL_ROLLOUT_FREE_CACHE_ENGINE"
SAVE_FREQ="$RL_SAVE_FREQ"
TEST_FREQ="$RL_TEST_FREQ"
TOTAL_EPOCHS="$RL_TOTAL_EPOCHS"
LOGGER="$RL_LOGGER"
LR="$RL_LR"
LR_SCHEDULER_TYPE="$RL_LR_SCHEDULER_TYPE"
KL_LOSS_COEF="$RL_KL_LOSS_COEF"
ENTROPY_COEFF="$RL_ENTROPY_COEFF"
NORM_ADV_BY_STD_IN_GRPO="$RL_NORM_ADV_BY_STD_IN_GRPO"
TEMPERATURE="$RL_TEMPERATURE"
VAL_TEMPERATURE="$RL_VAL_TEMPERATURE"

if [ "${ROLLOUT_ENABLE_CHUNKED_PREFILL}" != "True" ] && [ -n "${ROLLOUT_MAX_MODEL_LEN:-}" ]; then
    if [ "${ROLLOUT_MAX_BATCHED_TOKENS}" -lt "${ROLLOUT_MAX_MODEL_LEN}" ]; then
        echo "Invalid rollout config: enable_chunked_prefill=False requires max_num_batched_tokens >= max_model_len."
        echo "Current values: max_num_batched_tokens=${ROLLOUT_MAX_BATCHED_TOKENS}, max_model_len=${ROLLOUT_MAX_MODEL_LEN}."
        echo "Set RL_ROLLOUT_ENABLE_CHUNKED_PREFILL=True or increase RL_ROLLOUT_MAX_BATCHED_TOKENS."
        exit 1
    fi
fi

CMD=(
    python3 -m arft.main_agent_ppo
    "algorithm.adv_estimator=grpo"
    "data.train_files=$TRAIN_FILES"
    "data.val_files=$VAL_FILES"
    "data.prompt_key=raw_prompt"
    "data.train_batch_size=$TRAIN_BATCH_SIZE"
    "data.max_prompt_length=$MAX_PROMPT_LENGTH"
    "data.max_response_length=$MAX_RESPONSE_LENGTH"
    "data.dataloader_num_workers=$DATALOADER_NUM_WORKERS"
    "data.filter_overlong_prompts=False"
    "data.truncation=error"
    "data.return_raw_chat=True"
    "reward.custom_reward_function.path=$REWARD_FN_PATH"
    "reward.custom_reward_function.name=$REWARD_FN_NAME"
    "actor_rollout_ref.model.path=$MODEL_PATH"
    "actor_rollout_ref.actor.optim.lr=$LR"
    "actor_rollout_ref.actor.optim.lr_scheduler_type=$LR_SCHEDULER_TYPE"
    "actor_rollout_ref.actor.checkpoint.save_contents=['model','optimizer','extra','hf_model']"
    "actor_rollout_ref.actor.checkpoint.load_contents=['model','optimizer','extra']"
    "actor_rollout_ref.model.use_remove_padding=True"
    "actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$ACTOR_MAX_TOKEN_LEN_PER_GPU"
    "actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MINI_BATCH_SIZE"
    "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$PPO_MICRO_BATCH_SIZE"
    "actor_rollout_ref.actor.use_kl_loss=True"
    "actor_rollout_ref.actor.kl_loss_coef=$KL_LOSS_COEF"
    "actor_rollout_ref.actor.kl_loss_type=low_var_kl"
    "actor_rollout_ref.actor.entropy_coeff=$ENTROPY_COEFF"
    "actor_rollout_ref.model.enable_gradient_checkpointing=True"
    "actor_rollout_ref.actor.fsdp_config.model_dtype=$FSDP_MODEL_DTYPE"
    "actor_rollout_ref.actor.fsdp_config.use_torch_compile=$FSDP_USE_TORCH_COMPILE"
    "actor_rollout_ref.actor.fsdp_config.param_offload=$ACTOR_PARAM_OFFLOAD"
    "actor_rollout_ref.actor.fsdp_config.optimizer_offload=$ACTOR_OPTIMIZER_OFFLOAD"
    "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$LOGPROB_MICRO_BATCH_SIZE"
    "actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TP"
    "actor_rollout_ref.rollout.name=vllm"
    "actor_rollout_ref.rollout.load_format=$ROLLOUT_LOAD_FORMAT"
    "actor_rollout_ref.rollout.gpu_memory_utilization=$ROLLOUT_GPU_MEMORY_UTILIZATION"
    "actor_rollout_ref.rollout.max_num_batched_tokens=$ROLLOUT_MAX_BATCHED_TOKENS"
    "actor_rollout_ref.rollout.max_num_seqs=$ROLLOUT_MAX_NUM_SEQS"
    "actor_rollout_ref.rollout.enforce_eager=$ROLLOUT_ENFORCE_EAGER"
    "actor_rollout_ref.rollout.enable_prefix_caching=$ROLLOUT_ENABLE_PREFIX_CACHING"
    "actor_rollout_ref.rollout.enable_chunked_prefill=$ROLLOUT_ENABLE_CHUNKED_PREFILL"
    "actor_rollout_ref.rollout.free_cache_engine=$ROLLOUT_FREE_CACHE_ENGINE"
    "actor_rollout_ref.rollout.temperature=$TEMPERATURE"
    "actor_rollout_ref.rollout.response_length=$MAX_RESPONSE_LENGTH"
    "actor_rollout_ref.rollout.val_kwargs.temperature=$VAL_TEMPERATURE"
    "actor_rollout_ref.rollout.n=$ROLLOUT_N"
    "actor_rollout_ref.rollout.agent.agent_flow_config_path=$CONFIG_PATH"
    "actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=$ACTOR_MAX_TOKEN_LEN_PER_GPU"
    "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$LOGPROB_MICRO_BATCH_SIZE"
    "actor_rollout_ref.ref.fsdp_config.model_dtype=$FSDP_MODEL_DTYPE"
    "actor_rollout_ref.ref.fsdp_config.use_torch_compile=$FSDP_USE_TORCH_COMPILE"
    "actor_rollout_ref.ref.fsdp_config.param_offload=$REF_PARAM_OFFLOAD"
    "algorithm.use_kl_in_reward=False"
    "algorithm.norm_adv_by_std_in_grpo=$NORM_ADV_BY_STD_IN_GRPO"
    "trainer.logger=$LOGGER"
    "trainer.default_local_dir=$TRAINER_LOCAL_DIR"
    "trainer.project_name=$PROJECT_NAME"
    "trainer.experiment_name=$EXP_NAME"
    "trainer.n_gpus_per_node=$NUM_GPUS"
    "trainer.nnodes=$NNODES"
    "trainer.val_before_train=False"
    "trainer.log_val_generations=10"
    "trainer.save_freq=$SAVE_FREQ"
    "trainer.test_freq=$TEST_FREQ"
    "trainer.total_epochs=$TOTAL_EPOCHS"
)

if [ -n "${ROLLOUT_MAX_MODEL_LEN:-}" ]; then
    CMD+=("actor_rollout_ref.rollout.max_model_len=$ROLLOUT_MAX_MODEL_LEN")
fi

if declare -p RL_EXTRA_ARGS >/dev/null 2>&1; then
    CMD+=("${RL_EXTRA_ARGS[@]}")
fi

CMD+=("$@")

echo "FINAL MAX_PROMPT_LENGTH=$MAX_PROMPT_LENGTH"
echo "FINAL MAX_RESPONSE_LENGTH=$MAX_RESPONSE_LENGTH"
echo "FINAL ACTOR_MAX_TOKEN_LEN_PER_GPU=$ACTOR_MAX_TOKEN_LEN_PER_GPU"
echo "FINAL ROLLOUT_MAX_MODEL_LEN=${ROLLOUT_MAX_MODEL_LEN:-<unset>}"
echo "FINAL ACTOR_PARAM_OFFLOAD=$ACTOR_PARAM_OFFLOAD"
echo "FINAL ACTOR_OPTIMIZER_OFFLOAD=$ACTOR_OPTIMIZER_OFFLOAD"
echo "FINAL REF_PARAM_OFFLOAD=$REF_PARAM_OFFLOAD"
echo "FINAL ROLLOUT_GPU_MEMORY_UTILIZATION=$ROLLOUT_GPU_MEMORY_UTILIZATION"
echo "FINAL TEMPERATURE=$TEMPERATURE"
echo "FINAL VAL_TEMPERATURE=$VAL_TEMPERATURE"

mkdir -p "$PROJECT_DIR/artifacts/reports"
FINAL_CMD_DUMP_PATH="$PROJECT_DIR/artifacts/reports/final_launch_cmd.txt"
{
    printf 'timestamp=%s\n' "$(date -Is)"
    printf 'profile_path=%s\n' "$RESOLVED_PROFILE_PATH"
    printf 'exp_name=%s\n' "$EXP_NAME"
    printf 'model_path=%s\n' "$MODEL_PATH"
    printf 'cmd='
    printf '%q ' "${CMD[@]}"
    printf '\n'
} > "$FINAL_CMD_DUMP_PATH"
echo "[LAUNCH CMD] dumped to: $FINAL_CMD_DUMP_PATH"

if [ "${PRINT_CMD_ONLY:-0}" = "1" ]; then
    printf '%q ' "${CMD[@]}"
    printf '\n'
    exit 0
fi

"${CMD[@]}"
