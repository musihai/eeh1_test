#!/bin/bash
# shellcheck shell=bash

PROFILE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$PROFILE_DIR/../../.." && pwd)"

RUN_MODE="${RUN_MODE:-train}"

export MODEL_PATH="${MODEL_PATH:-/data/linyujie/models/Qwen3-1.7B}"
export TRAIN_GPU_IDS="${TRAIN_GPU_IDS:-0,1,2}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$TRAIN_GPU_IDS}"
export NUM_GPUS="${NUM_GPUS:-3}"

export SERVER_GPU_ID="${SERVER_GPU_ID:-3}"
export MODEL_SERVICE_PORT="${MODEL_SERVICE_PORT:-8994}"
export MODEL_SERVICE_URL="${MODEL_SERVICE_URL:-http://127.0.0.1:${MODEL_SERVICE_PORT}}"

export RL_CONFIG_PATH="${RL_CONFIG_PATH:-$PROJECT_DIR/recipe/time_series_forecast/base.yaml}"
export RL_TRAIN_FILES="${RL_TRAIN_FILES:-$PROJECT_DIR/dataset/ett_rl_etth1_paper_aligned_ot_20260315_151424/train.jsonl}"
export RL_VAL_FILES="${RL_VAL_FILES:-$PROJECT_DIR/dataset/ett_rl_etth1_paper_aligned_ot_20260315_151424/val.jsonl}"
export RL_PROJECT_NAME="${RL_PROJECT_NAME:-TimeSeriesForecast}"
export RL_ROLLOUT_TP="${RL_ROLLOUT_TP:-1}"
export RL_ROLLOUT_GPU_MEMORY_UTILIZATION="${RL_ROLLOUT_GPU_MEMORY_UTILIZATION:-0.35}"
export RL_ROLLOUT_LOAD_FORMAT="${RL_ROLLOUT_LOAD_FORMAT:-safetensors}"
export RL_MAX_PROMPT_LENGTH="${RL_MAX_PROMPT_LENGTH:-4096}"
export RL_DATALOADER_NUM_WORKERS="${RL_DATALOADER_NUM_WORKERS:-0}"
export RL_ACTOR_MAX_TOKEN_LEN_PER_GPU="${RL_ACTOR_MAX_TOKEN_LEN_PER_GPU:-8192}"
export RL_ROLLOUT_MAX_BATCHED_TOKENS="${RL_ROLLOUT_MAX_BATCHED_TOKENS:-8192}"
export RL_ROLLOUT_MAX_NUM_SEQS="${RL_ROLLOUT_MAX_NUM_SEQS:-32}"
export RL_ROLLOUT_ENFORCE_EAGER="${RL_ROLLOUT_ENFORCE_EAGER:-True}"
export RL_ROLLOUT_ENABLE_PREFIX_CACHING="${RL_ROLLOUT_ENABLE_PREFIX_CACHING:-False}"
export RL_ROLLOUT_ENABLE_CHUNKED_PREFILL="${RL_ROLLOUT_ENABLE_CHUNKED_PREFILL:-False}"
export RL_ROLLOUT_FREE_CACHE_ENGINE="${RL_ROLLOUT_FREE_CACHE_ENGINE:-False}"
export RL_LR="${RL_LR:-1e-6}"
export RL_LR_SCHEDULER_TYPE="${RL_LR_SCHEDULER_TYPE:-cosine}"
export RL_KL_LOSS_COEF="${RL_KL_LOSS_COEF:-0.01}"
export RL_TEMPERATURE="${RL_TEMPERATURE:-0.6}"
export RL_VAL_TEMPERATURE="${RL_VAL_TEMPERATURE:-0.2}"
export RL_NNODES="${RL_NNODES:-1}"

export SFT_TRAIN_FILES="${SFT_TRAIN_FILES:-$PROJECT_DIR/dataset/ett_sft_etth1_runtime_ot_paper200/train.parquet}"
export SFT_VAL_FILES="${SFT_VAL_FILES:-$PROJECT_DIR/dataset/ett_sft_etth1_runtime_ot_paper200/val.parquet}"
export SFT_SAVE_DIR="${SFT_SAVE_DIR:-$PROJECT_DIR/artifacts/checkpoints/sft/time_series_forecast_sft}"
export SFT_PROJECT_NAME="${SFT_PROJECT_NAME:-TimeSeriesForecast-SFT}"
export SFT_NUM_GPUS="${SFT_NUM_GPUS:-$NUM_GPUS}"
export SFT_MAX_LENGTH="${SFT_MAX_LENGTH:-8192}"
export SFT_MAX_TOKEN_LEN_PER_GPU="${SFT_MAX_TOKEN_LEN_PER_GPU:-8192}"
export SFT_LR="${SFT_LR:-1e-5}"
export SFT_LR_SCHEDULER_TYPE="${SFT_LR_SCHEDULER_TYPE:-cosine}"

case "$RUN_MODE" in
    smoke)
        export SFT_EXPERIMENT_NAME="${SFT_EXPERIMENT_NAME:-qwen3-1.7b-etth1-ot-sft-smoke-gpu012}"
        export SFT_TRAIN_BATCH_SIZE="${SFT_TRAIN_BATCH_SIZE:-6}"
        export SFT_MICRO_BATCH_SIZE="${SFT_MICRO_BATCH_SIZE:-1}"
        export SFT_TOTAL_EPOCHS="${SFT_TOTAL_EPOCHS:-1}"
        export SFT_SAVE_FREQ="${SFT_SAVE_FREQ:-after_each_epoch}"
        export SFT_TEST_FREQ="${SFT_TEST_FREQ:-after_each_epoch}"
        export SFT_LOGGER="${SFT_LOGGER:-[\"console\"]}"
        SFT_EXTRA_ARGS=(
            "data.train_max_samples=24"
            "data.val_max_samples=8"
        )

        export RL_EXP_NAME="${RL_EXP_NAME:-etth1_ot_qwen3_1_7b_rl_smoke_gpu012}"
        export RL_TRAIN_BATCH_SIZE="${RL_TRAIN_BATCH_SIZE:-3}"
        export RL_PPO_MINI_BATCH_SIZE="${RL_PPO_MINI_BATCH_SIZE:-3}"
        export RL_PPO_MICRO_BATCH_SIZE="${RL_PPO_MICRO_BATCH_SIZE:-1}"
        export RL_LOGPROB_MICRO_BATCH_SIZE="${RL_LOGPROB_MICRO_BATCH_SIZE:-1}"
        export RL_ROLLOUT_N="${RL_ROLLOUT_N:-1}"
        export RL_MAX_RESPONSE_LENGTH="${RL_MAX_RESPONSE_LENGTH:-4096}"
        export RL_SAVE_FREQ="${RL_SAVE_FREQ:-1}"
        export RL_TEST_FREQ="${RL_TEST_FREQ:-5}"
        export RL_TOTAL_EPOCHS="${RL_TOTAL_EPOCHS:-1}"
        export RL_LOGGER="${RL_LOGGER:-[\"console\"]}"
        RL_EXTRA_ARGS=(
            "data.train_max_samples=3"
            "data.val_max_samples=1"
            "trainer.total_training_steps=1"
        )
        ;;
    train)
        export SFT_EXPERIMENT_NAME="${SFT_EXPERIMENT_NAME:-qwen3-1.7b-etth1-ot-sft-gpu012}"
        export SFT_TRAIN_BATCH_SIZE="${SFT_TRAIN_BATCH_SIZE:-18}"
        export SFT_MICRO_BATCH_SIZE="${SFT_MICRO_BATCH_SIZE:-2}"
        export SFT_TOTAL_EPOCHS="${SFT_TOTAL_EPOCHS:-1}"
        export SFT_SAVE_FREQ="${SFT_SAVE_FREQ:-after_each_epoch}"
        export SFT_TEST_FREQ="${SFT_TEST_FREQ:-after_each_epoch}"
        export SFT_LOGGER="${SFT_LOGGER:-[\"console\"]}"
        SFT_EXTRA_ARGS=()

        export RL_EXP_NAME="${RL_EXP_NAME:-etth1_ot_qwen3_1_7b_rl_gpu012}"
        export RL_TRAIN_BATCH_SIZE="${RL_TRAIN_BATCH_SIZE:-3}"
        export RL_PPO_MINI_BATCH_SIZE="${RL_PPO_MINI_BATCH_SIZE:-3}"
        export RL_PPO_MICRO_BATCH_SIZE="${RL_PPO_MICRO_BATCH_SIZE:-1}"
        export RL_LOGPROB_MICRO_BATCH_SIZE="${RL_LOGPROB_MICRO_BATCH_SIZE:-1}"
        export RL_ROLLOUT_N="${RL_ROLLOUT_N:-2}"
        export RL_MAX_RESPONSE_LENGTH="${RL_MAX_RESPONSE_LENGTH:-4096}"
        export RL_SAVE_FREQ="${RL_SAVE_FREQ:-50}"
        export RL_TEST_FREQ="${RL_TEST_FREQ:-5}"
        export RL_TOTAL_EPOCHS="${RL_TOTAL_EPOCHS:-1}"
        export RL_LOGGER="${RL_LOGGER:-[\"console\"]}"
        RL_EXTRA_ARGS=()
        ;;
    *)
        echo "Unsupported RUN_MODE: $RUN_MODE"
        return 1 2>/dev/null || exit 1
        ;;
esac
