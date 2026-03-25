#!/bin/bash
# shellcheck shell=bash

PROFILE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$PROFILE_DIR/../../.." && pwd)"

# 这是当前 time-series 正式实验的单一主配置文件。
# 使用方式：
# 1. 永久修改：直接改本文件里的默认值
# 2. 临时覆盖：在命令前加环境变量，例如
#    RL_MODEL_PATH=/abs/path/to/hf bash examples/time_series_forecast/run_qwen3-1.7B_curriculum.sh
#    RUN_MODE=val_only DEBUG_CHAIN=1 bash examples/time_series_forecast/run_qwen3-1.7B.sh

# 运行模式：train / smoke / val_only
RUN_MODE="${RUN_MODE:-train}"

# =========================
# 基础模型与 GPU 分配
# =========================
# 基座模型路径
export MODEL_PATH="${MODEL_PATH:-/data/linyujie/models/Qwen3-1.7B}"
# 训练使用的 GPU 列表
export TRAIN_GPU_IDS="${TRAIN_GPU_IDS:-0,1,2}"
# 实际训练进程可见的 GPU
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$TRAIN_GPU_IDS}"
# 训练使用的 GPU 数量
export NUM_GPUS="${NUM_GPUS:-3}"

# =========================
# 预测服务配置
# =========================
# 预测服务默认单独占用的 GPU
export SERVER_GPU_ID="${SERVER_GPU_ID:-3}"
# 预测服务端口
export MODEL_SERVICE_PORT="${MODEL_SERVICE_PORT:-8994}"
# 预测服务地址
export MODEL_SERVICE_URL="${MODEL_SERVICE_URL:-http://127.0.0.1:${MODEL_SERVICE_PORT}}"

# =========================
# RL / 共享数据配置
# =========================
# agent flow 配置文件
export RL_CONFIG_PATH="${RL_CONFIG_PATH:-$PROJECT_DIR/recipe/time_series_forecast/base.yaml}"
# curriculum RL 数据目录
export RL_CURRICULUM_DATASET_DIR="${RL_CURRICULUM_DATASET_DIR:-$PROJECT_DIR/dataset/ett_rl_etth1_mv1}"
# RL 初始化 checkpoint；当前默认指向 mv1 重新训练得到的 SFT warm start
export RL_MODEL_PATH="${RL_MODEL_PATH:-$PROJECT_DIR/artifacts/checkpoints/sft/time_series_forecast_sft_mv1_tsfix_20260324/global_step_66/hf_merged}"
# RL 训练集与验证集；train 在 curriculum wrapper 中会自动替换成分阶段文件
export RL_TRAIN_FILES="${RL_TRAIN_FILES:-$RL_CURRICULUM_DATASET_DIR/train.jsonl}"
export RL_VAL_FILES="${RL_VAL_FILES:-$RL_CURRICULUM_DATASET_DIR/val.jsonl}"
# curriculum 阶段顺序
export RL_CURRICULUM_PHASES="${RL_CURRICULUM_PHASES:-stage1,stage12,stage123}"
# RL 项目名
export RL_PROJECT_NAME="${RL_PROJECT_NAME:-TimeSeriesForecast-Formal}"
# rollout tensor parallel 大小
export RL_ROLLOUT_TP="${RL_ROLLOUT_TP:-1}"
# FSDP 初始化 dtype
export RL_FSDP_MODEL_DTYPE="${RL_FSDP_MODEL_DTYPE:-bfloat16}"
# 是否启用 torch compile
export RL_FSDP_USE_TORCH_COMPILE="${RL_FSDP_USE_TORCH_COMPILE:-False}"
# vLLM 显存利用率
export RL_ROLLOUT_GPU_MEMORY_UTILIZATION="${RL_ROLLOUT_GPU_MEMORY_UTILIZATION:-0.25}"
# 模型加载格式
export RL_ROLLOUT_LOAD_FORMAT="${RL_ROLLOUT_LOAD_FORMAT:-safetensors}"
# prompt 最大长度
export RL_MAX_PROMPT_LENGTH="${RL_MAX_PROMPT_LENGTH:-9216}"
# dataloader worker 数
export RL_DATALOADER_NUM_WORKERS="${RL_DATALOADER_NUM_WORKERS:-0}"
# PPO 每卡最大 token 长度
export RL_ACTOR_MAX_TOKEN_LEN_PER_GPU="${RL_ACTOR_MAX_TOKEN_LEN_PER_GPU:-12288}"
# rollout 最大模型长度
export RL_ROLLOUT_MAX_MODEL_LEN="${RL_ROLLOUT_MAX_MODEL_LEN:-12288}"
# vLLM 每批最大 token 数
export RL_ROLLOUT_MAX_BATCHED_TOKENS="${RL_ROLLOUT_MAX_BATCHED_TOKENS:-12288}"
# vLLM 最大并发序列数
export RL_ROLLOUT_MAX_NUM_SEQS="${RL_ROLLOUT_MAX_NUM_SEQS:-3}"
# 是否强制 eager
export RL_ROLLOUT_ENFORCE_EAGER="${RL_ROLLOUT_ENFORCE_EAGER:-True}"
# 是否打开 prefix cache
export RL_ROLLOUT_ENABLE_PREFIX_CACHING="${RL_ROLLOUT_ENABLE_PREFIX_CACHING:-False}"
# 是否打开 chunked prefill
export RL_ROLLOUT_ENABLE_CHUNKED_PREFILL="${RL_ROLLOUT_ENABLE_CHUNKED_PREFILL:-True}"
# 是否释放 cache engine
export RL_ROLLOUT_FREE_CACHE_ENGINE="${RL_ROLLOUT_FREE_CACHE_ENGINE:-False}"
# RL 学习率
export RL_LR="${RL_LR:-1e-6}"
# RL 学习率调度器
export RL_LR_SCHEDULER_TYPE="${RL_LR_SCHEDULER_TYPE:-cosine}"
# actor KL loss 系数
export RL_KL_LOSS_COEF="${RL_KL_LOSS_COEF:-0.01}"
# rollout 训练温度
export RL_TEMPERATURE="${RL_TEMPERATURE:-1.0}"
# rollout 验证温度
export RL_VAL_TEMPERATURE="${RL_VAL_TEMPERATURE:-0.0}"
# 节点数
export RL_NNODES="${RL_NNODES:-1}"

# =========================
# SFT 默认配置
# =========================
# SFT 默认使用的 GPU 列表
export SFT_GPU_IDS="${SFT_GPU_IDS:-0,1,2}"
# SFT 实际可见 GPU；需要临时改卡时优先覆盖这个变量
export SFT_CUDA_VISIBLE_DEVICES="${SFT_CUDA_VISIBLE_DEVICES:-$SFT_GPU_IDS}"
# 当前推荐的 SFT 数据目录
export SFT_DATASET_DIR="${SFT_DATASET_DIR:-$PROJECT_DIR/dataset/ett_sft_etth1_runtime_ot_teacher200_mv1_stepwise_r25_tsfix}"
# SFT 保存目录
export SFT_SAVE_DIR="${SFT_SAVE_DIR:-$PROJECT_DIR/artifacts/checkpoints/sft/time_series_forecast_sft_mv1_tsfix_20260324}"
# SFT 项目名
export SFT_PROJECT_NAME="${SFT_PROJECT_NAME:-TimeSeriesForecast-SFT-Formal}"
# SFT 使用的 GPU 数
export SFT_NUM_GPUS="${SFT_NUM_GPUS:-3}"
# SFT 最大样本长度
export SFT_MAX_LENGTH="${SFT_MAX_LENGTH:-9216}"
# SFT 每卡最大 token 长度
export SFT_MAX_TOKEN_LEN_PER_GPU="${SFT_MAX_TOKEN_LEN_PER_GPU:-9216}"
# SFT 学习率
export SFT_LR="${SFT_LR:-1e-5}"
# SFT 学习率调度器
export SFT_LR_SCHEDULER_TYPE="${SFT_LR_SCHEDULER_TYPE:-cosine}"

case "$RUN_MODE" in
    val_only)
        # =========================
        # RL val-only 默认值
        # =========================
        export RL_EXP_NAME="${RL_EXP_NAME:-etth1_ot_qwen3_1_7b_rl_mv1_valonly_20260324}"
        export RL_TRAIN_BATCH_SIZE="${RL_TRAIN_BATCH_SIZE:-$NUM_GPUS}"
        export RL_PPO_MINI_BATCH_SIZE="${RL_PPO_MINI_BATCH_SIZE:-$RL_TRAIN_BATCH_SIZE}"
        export RL_PPO_MICRO_BATCH_SIZE="${RL_PPO_MICRO_BATCH_SIZE:-1}"
        export RL_LOGPROB_MICRO_BATCH_SIZE="${RL_LOGPROB_MICRO_BATCH_SIZE:-1}"
        export RL_ROLLOUT_N="${RL_ROLLOUT_N:-1}"
        export RL_MAX_RESPONSE_LENGTH="${RL_MAX_RESPONSE_LENGTH:-3072}"
        export RL_SAVE_FREQ="${RL_SAVE_FREQ:-1}"
        export RL_TEST_FREQ="${RL_TEST_FREQ:-1}"
        export RL_TOTAL_EPOCHS="${RL_TOTAL_EPOCHS:-1}"
        export RL_LOGGER="${RL_LOGGER:-[\"console\"]}"
        RL_EXTRA_ARGS=(
            "trainer.val_only=True"
            "trainer.val_before_train=True"
            "data.train_max_samples=32"
            "data.val_max_samples=256"
            "trainer.total_training_steps=1"
        )
        ;;
    smoke)
        # =========================
        # SFT smoke 默认值
        # =========================
        export SFT_EXPERIMENT_NAME="${SFT_EXPERIMENT_NAME:-qwen3-1.7b-etth1-ot-sft-paper-strict-formal-smoke-gpu012}"
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

        # =========================
        # RL smoke 默认值
        # =========================
        export RL_EXP_NAME="${RL_EXP_NAME:-etth1_ot_qwen3_1_7b_rl_mv1_smoke_20260324}"
        export RL_TRAIN_BATCH_SIZE="${RL_TRAIN_BATCH_SIZE:-3}"
        export RL_PPO_MINI_BATCH_SIZE="${RL_PPO_MINI_BATCH_SIZE:-3}"
        export RL_PPO_MICRO_BATCH_SIZE="${RL_PPO_MICRO_BATCH_SIZE:-1}"
        export RL_LOGPROB_MICRO_BATCH_SIZE="${RL_LOGPROB_MICRO_BATCH_SIZE:-1}"
        export RL_ROLLOUT_N="${RL_ROLLOUT_N:-8}"
        export RL_MAX_RESPONSE_LENGTH="${RL_MAX_RESPONSE_LENGTH:-3072}"
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
        # =========================
        # SFT train 默认值
        # =========================
        export SFT_EXPERIMENT_NAME="${SFT_EXPERIMENT_NAME:-qwen3-1.7b-etth1-sft-mv1-tsfix}"
        export SFT_TRAIN_BATCH_SIZE="${SFT_TRAIN_BATCH_SIZE:-12}"
        export SFT_MICRO_BATCH_SIZE="${SFT_MICRO_BATCH_SIZE:-1}"
        export SFT_TOTAL_EPOCHS="${SFT_TOTAL_EPOCHS:-1}"
        export SFT_SAVE_FREQ="${SFT_SAVE_FREQ:-after_each_epoch}"
        export SFT_TEST_FREQ="${SFT_TEST_FREQ:-after_each_epoch}"
        export SFT_LOGGER="${SFT_LOGGER:-[\"console\"]}"
        SFT_EXTRA_ARGS=()

        # =========================
        # RL train 默认值
        # =========================
        export RL_EXP_NAME="${RL_EXP_NAME:-etth1_ot_qwen3_1_7b_rl_mv1_20260324}"
        export RL_TRAIN_BATCH_SIZE="${RL_TRAIN_BATCH_SIZE:-3}"
        export RL_PPO_MINI_BATCH_SIZE="${RL_PPO_MINI_BATCH_SIZE:-3}"
        export RL_PPO_MICRO_BATCH_SIZE="${RL_PPO_MICRO_BATCH_SIZE:-1}"
        export RL_LOGPROB_MICRO_BATCH_SIZE="${RL_LOGPROB_MICRO_BATCH_SIZE:-1}"
        export RL_ROLLOUT_N="${RL_ROLLOUT_N:-8}"
        export RL_MAX_RESPONSE_LENGTH="${RL_MAX_RESPONSE_LENGTH:-3072}"
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
