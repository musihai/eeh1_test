#!/bin/bash
set -euo pipefail

if [ "${TRACE:-0}" = "1" ]; then
    set -x
fi

export VLLM_USE_V1=1
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HYDRA_FULL_ERROR=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"

DEBUG_CHAIN="${DEBUG_CHAIN:-0}"
if [ "$DEBUG_CHAIN" = "1" ] || [ "${DEBUG_CHAIN,,}" = "true" ]; then
    export TS_CHAIN_DEBUG=1
    export TS_CHAIN_DEBUG_FILE="${TS_CHAIN_DEBUG_FILE:-$PROJECT_DIR/logs/debug/ts_chain_debug.jsonl}"
    mkdir -p "$(dirname "$TS_CHAIN_DEBUG_FILE")"
    echo "[CHAIN DEBUG] enabled, writing to: $TS_CHAIN_DEBUG_FILE"
fi

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
if [ -z "$PROFILE_PATH" ]; then
    echo "PROFILE_PATH is required. Set PROFILE_PATH to your single source-of-truth profile file."
    exit 1
fi

RESOLVED_PROFILE_PATH="$(resolve_profile_path "$PROFILE_PATH")" || {
    echo "PROFILE_PATH not found: $PROFILE_PATH"
    exit 1
}
# shellcheck disable=SC1090
source "$RESOLVED_PROFILE_PATH"

CONFIG_PATH="${CONFIG_PATH:-${RL_CONFIG_PATH:-$PROJECT_DIR/recipe/time_series_forecast/base.yaml}}"
TRAIN_FILES="${TRAIN_FILES:-${RL_TRAIN_FILES:-$PROJECT_DIR/dataset/ett_rl_etth1_paper_aligned_ot_20260315_151424/train.jsonl}}"
VAL_FILES="${VAL_FILES:-${RL_VAL_FILES:-$PROJECT_DIR/dataset/ett_rl_etth1_paper_aligned_ot_20260315_151424/val.jsonl}}"
REWARD_FN_PATH="${REWARD_FN_PATH:-${RL_REWARD_FN_PATH:-$PROJECT_DIR/recipe/time_series_forecast/reward.py}}"
REWARD_FN_NAME="${REWARD_FN_NAME:-${RL_REWARD_FN_NAME:-compute_score}}"
MODEL_PATH="${MODEL_PATH:-}"
if [ -z "$MODEL_PATH" ]; then
    echo "MODEL_PATH is required. Export MODEL_PATH to your base chat model checkpoint."
    exit 1
fi

PROJECT_NAME="${PROJECT_NAME:-${RL_PROJECT_NAME:-TimeSeriesForecast}}"
EXP_NAME="${EXP_NAME:-${RL_EXP_NAME:-etth1_ot_qwen3_1_7b}}"
TRAINER_LOCAL_DIR="${TRAINER_LOCAL_DIR:-${RL_TRAINER_LOCAL_DIR:-$PROJECT_DIR/artifacts/checkpoints/rl/$EXP_NAME}}"
NNODES="${NNODES:-${RL_NNODES:-1}}"
if [ -z "${NUM_GPUS:-}" ] && [ -n "${RL_NUM_GPUS:-}" ]; then
    NUM_GPUS="$RL_NUM_GPUS"
fi
if [ -z "${NUM_GPUS:-}" ] && [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    IFS=',' read -r -a _VISIBLE_GPUS <<< "$CUDA_VISIBLE_DEVICES"
    NUM_GPUS="${#_VISIBLE_GPUS[@]}"
fi
NUM_GPUS="${NUM_GPUS:-4}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-${RL_TRAIN_BATCH_SIZE:-$NUM_GPUS}}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-${RL_PPO_MINI_BATCH_SIZE:-$TRAIN_BATCH_SIZE}}"
PPO_MICRO_BATCH_SIZE="${PPO_MICRO_BATCH_SIZE:-${RL_PPO_MICRO_BATCH_SIZE:-1}}"
LOGPROB_MICRO_BATCH_SIZE="${LOGPROB_MICRO_BATCH_SIZE:-${RL_LOGPROB_MICRO_BATCH_SIZE:-1}}"
ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-${RL_ROLLOUT_GPU_MEMORY_UTILIZATION:-0.35}}"
ROLLOUT_LOAD_FORMAT="${ROLLOUT_LOAD_FORMAT:-${RL_ROLLOUT_LOAD_FORMAT:-safetensors}}"
ROLLOUT_N="${ROLLOUT_N:-${RL_ROLLOUT_N:-1}}"
ROLLOUT_TP="${ROLLOUT_TP:-${RL_ROLLOUT_TP:-1}}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-${RL_MAX_PROMPT_LENGTH:-}}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-${RL_MAX_RESPONSE_LENGTH:-}}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-${RL_DATALOADER_NUM_WORKERS:-0}}"
ACTOR_MAX_TOKEN_LEN_PER_GPU="${ACTOR_MAX_TOKEN_LEN_PER_GPU:-${RL_ACTOR_MAX_TOKEN_LEN_PER_GPU:-8192}}"
ROLLOUT_MAX_MODEL_LEN="${ROLLOUT_MAX_MODEL_LEN:-${RL_ROLLOUT_MAX_MODEL_LEN:-}}"
ROLLOUT_MAX_BATCHED_TOKENS="${ROLLOUT_MAX_BATCHED_TOKENS:-${RL_ROLLOUT_MAX_BATCHED_TOKENS:-4096}}"
ROLLOUT_MAX_NUM_SEQS="${ROLLOUT_MAX_NUM_SEQS:-${RL_ROLLOUT_MAX_NUM_SEQS:-32}}"
ROLLOUT_ENFORCE_EAGER="${ROLLOUT_ENFORCE_EAGER:-${RL_ROLLOUT_ENFORCE_EAGER:-True}}"
ROLLOUT_ENABLE_PREFIX_CACHING="${ROLLOUT_ENABLE_PREFIX_CACHING:-${RL_ROLLOUT_ENABLE_PREFIX_CACHING:-False}}"
ROLLOUT_ENABLE_CHUNKED_PREFILL="${ROLLOUT_ENABLE_CHUNKED_PREFILL:-${RL_ROLLOUT_ENABLE_CHUNKED_PREFILL:-False}}"
ROLLOUT_FREE_CACHE_ENGINE="${ROLLOUT_FREE_CACHE_ENGINE:-${RL_ROLLOUT_FREE_CACHE_ENGINE:-False}}"
SAVE_FREQ="${SAVE_FREQ:-${RL_SAVE_FREQ:-10}}"
TEST_FREQ="${TEST_FREQ:-${RL_TEST_FREQ:-5}}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-${RL_TOTAL_EPOCHS:-10}}"
LOGGER="${LOGGER:-${RL_LOGGER:-[\"console\"]}}"
LR="${LR:-${RL_LR:-1e-6}}"
LR_SCHEDULER_TYPE="${LR_SCHEDULER_TYPE:-${RL_LR_SCHEDULER_TYPE:-cosine}}"
KL_LOSS_COEF="${KL_LOSS_COEF:-${RL_KL_LOSS_COEF:-0.01}}"
TEMPERATURE="${TEMPERATURE:-${RL_TEMPERATURE:-}}"
VAL_TEMPERATURE="${VAL_TEMPERATURE:-${RL_VAL_TEMPERATURE:-}}"

for REQUIRED_KEY in MAX_PROMPT_LENGTH MAX_RESPONSE_LENGTH TEMPERATURE VAL_TEMPERATURE; do
    if [ -z "${!REQUIRED_KEY:-}" ]; then
        echo "Missing required config: $REQUIRED_KEY"
        echo "Set it in PROFILE_PATH (recommended) or export $REQUIRED_KEY / RL_${REQUIRED_KEY} before launch."
        exit 1
    fi
done

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
    "data.filter_overlong_prompts=True"
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
    "actor_rollout_ref.actor.entropy_coeff=0"
    "actor_rollout_ref.model.enable_gradient_checkpointing=True"
    "actor_rollout_ref.actor.fsdp_config.param_offload=True"
    "actor_rollout_ref.actor.fsdp_config.optimizer_offload=True"
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
    "actor_rollout_ref.ref.fsdp_config.param_offload=True"
    "algorithm.use_kl_in_reward=False"
    "algorithm.norm_adv_by_std_in_grpo=False"
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
echo "FINAL TEMPERATURE=$TEMPERATURE"
echo "FINAL VAL_TEMPERATURE=$VAL_TEMPERATURE"

mkdir -p "$PROJECT_DIR/debug_logs"
FINAL_CMD_DUMP_PATH="$PROJECT_DIR/debug_logs/final_launch_cmd.txt"
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
