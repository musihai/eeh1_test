#!/bin/bash
# shellcheck shell=bash

PROFILE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$PROFILE_DIR/../../.." && pwd)"

# ETTh1 正式实验的唯一默认配置入口。
# 规则：
# - 想永久修改默认值，直接改本文件。
# - 想只覆盖某一次运行，在启动命令前加环境变量。

# 运行模式：正式训练 / 冒烟测试 / 仅验证
RUN_MODE="${RUN_MODE:-train}"

# =========================
# 路径配置
# =========================
# SFT 与 RL 在试运行时默认使用的 Qwen 基础模型。
export BASE_MODEL_PATH="${BASE_MODEL_PATH:-/data/linyujie/models/Qwen3-1.7B}"
# SFT 初始化模型；默认直接使用基座模型。
export SFT_MODEL_PATH="${SFT_MODEL_PATH:-$BASE_MODEL_PATH}"
# RL 热启动权重目录；正式 RL 前必须显式提供。
export RL_MODEL_PATH="${RL_MODEL_PATH:-}"
# RL 采样进程使用的 agent flow 配置文件。
export RL_CONFIG_PATH="${RL_CONFIG_PATH:-$PROJECT_DIR/recipe/time_series_forecast/base.yaml}"
# RL 自定义奖励函数文件路径。
export RL_REWARD_FN_PATH="${RL_REWARD_FN_PATH:-$PROJECT_DIR/recipe/time_series_forecast/reward.py}"
# `reward.py` 中实际调用的函数名。
export RL_REWARD_FN_NAME="${RL_REWARD_FN_NAME:-compute_score}"

# 数据集路径
# 分步式 SFT 的 parquet 目录。
export SFT_DATASET_DIR="${SFT_DATASET_DIR:-$PROJECT_DIR/dataset/ett_sft_etth1_runtime_ot_teacher200_paper_same4_stepwise}"
# 课程式 RL 的 jsonl 数据目录。
export RL_CURRICULUM_DATASET_DIR="${RL_CURRICULUM_DATASET_DIR:-$PROJECT_DIR/dataset/ett_rl_etth1_paper_aligned_ot_curriculum_same2}"
# 完整课程式 RL 的阶段顺序。
export RL_CURRICULUM_PHASES="${RL_CURRICULUM_PHASES:-stage1,stage12,stage123}"

# =========================
# 资源配置
# =========================
# 默认训练 GPU 列表。
export TRAIN_GPU_IDS="${TRAIN_GPU_IDS:-0,1,2}"
# 当前命令默认可见的 GPU。
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$TRAIN_GPU_IDS}"
# 默认使用的 GPU 数量。
export NUM_GPUS="${NUM_GPUS:-3}"
# SFT 默认可见的 GPU 列表；SFT 单独换卡时改这里。
export SFT_GPU_IDS="${SFT_GPU_IDS:-$TRAIN_GPU_IDS}"
# SFT 实际使用的 CUDA_VISIBLE_DEVICES；启动脚本优先读取这个值。
export SFT_CUDA_VISIBLE_DEVICES="${SFT_CUDA_VISIBLE_DEVICES:-$SFT_GPU_IDS}"
# SFT 训练器使用的 GPU 数量。
export SFT_NUM_GPUS="${SFT_NUM_GPUS:-$NUM_GPUS}"
# RL 训练器使用的 GPU 数量。
export RL_NUM_GPUS="${RL_NUM_GPUS:-$NUM_GPUS}"
# RL 训练器使用的节点数。
export RL_NNODES="${RL_NNODES:-1}"
# 说明：
# - 论文 RL 使用 4 张训练卡，global batch size=64，group size G=8，gradient accumulation=4。
# - 当前正式配置默认保留 1 张卡给 expert 服务，只用 3 张卡训练。
# - 在 verl 里必须满足 real_train_batch_size = data.train_batch_size * rollout.n，
#   且 real_train_batch_size 需要能被 RL_NUM_GPUS 整除。
# - 因此在 3 张训练卡下，64 条轨迹/iter 不能精确实现；正式默认值会取最接近论文、
#   同时仍保持 gradient accumulation=4 的配置。

# 预测服务
# expert 预测服务默认占用的 GPU。
export SERVER_GPU_ID="${SERVER_GPU_ID:-3}"
# expert 预测服务端口。
export MODEL_SERVICE_PORT="${MODEL_SERVICE_PORT:-8994}"
# expert 预测服务完整地址。
export MODEL_SERVICE_URL="${MODEL_SERVICE_URL:-http://127.0.0.1:${MODEL_SERVICE_PORT}}"

# =========================
# 实验标识
# =========================
# SFT 的日志项目名。
export SFT_PROJECT_NAME="${SFT_PROJECT_NAME:-TimeSeriesForecast-SFT-Formal}"
# RL 的日志项目名。
export RL_PROJECT_NAME="${RL_PROJECT_NAME:-TimeSeriesForecast-Formal}"

# =========================
# 训练超参数
# =========================
# SFT 通用超参
# SFT 最大样本长度。
export SFT_MAX_LENGTH="${SFT_MAX_LENGTH:-9216}"
# SFT 每卡最大 token 数。
export SFT_MAX_TOKEN_LEN_PER_GPU="${SFT_MAX_TOKEN_LEN_PER_GPU:-9216}"
# SFT 学习率。
export SFT_LR="${SFT_LR:-1e-5}"
# SFT 学习率调度策略。
export SFT_LR_SCHEDULER_TYPE="${SFT_LR_SCHEDULER_TYPE:-cosine}"

# RL 通用超参
# RL actor 学习率。
export RL_LR="${RL_LR:-1e-6}"
# RL 学习率调度策略。
export RL_LR_SCHEDULER_TYPE="${RL_LR_SCHEDULER_TYPE:-cosine}"
# actor PPO loss 的 KL 系数。
export RL_KL_LOSS_COEF="${RL_KL_LOSS_COEF:-0.04}"
# 训练采样阶段的温度。
export RL_TEMPERATURE="${RL_TEMPERATURE:-1.0}"
# 验证采样阶段的温度。
export RL_VAL_TEMPERATURE="${RL_VAL_TEMPERATURE:-0.0}"
# RL 数据加载时的输入提示最大长度。
export RL_MAX_PROMPT_LENGTH="${RL_MAX_PROMPT_LENGTH:-8192}"
# 采样与训练共用的回复最大长度。
export RL_MAX_RESPONSE_LENGTH="${RL_MAX_RESPONSE_LENGTH:-4096}"
# RL 数据加载进程数。
export RL_DATALOADER_NUM_WORKERS="${RL_DATALOADER_NUM_WORKERS:-0}"
# actor 每卡最大 token 预算。
export RL_ACTOR_MAX_TOKEN_LEN_PER_GPU="${RL_ACTOR_MAX_TOKEN_LEN_PER_GPU:-12288}"
# 采样模型的最大上下文长度。
export RL_ROLLOUT_MAX_MODEL_LEN="${RL_ROLLOUT_MAX_MODEL_LEN:-12288}"
# vLLM 单批最大 token 数。
export RL_ROLLOUT_MAX_BATCHED_TOKENS="${RL_ROLLOUT_MAX_BATCHED_TOKENS:-12288}"
# vLLM 最大并发序列数。
export RL_ROLLOUT_MAX_NUM_SEQS="${RL_ROLLOUT_MAX_NUM_SEQS:-3}"
# vLLM 张量并行大小。
export RL_ROLLOUT_TP="${RL_ROLLOUT_TP:-1}"
# vLLM 显存利用率目标。
export RL_ROLLOUT_GPU_MEMORY_UTILIZATION="${RL_ROLLOUT_GPU_MEMORY_UTILIZATION:-0.25}"
# vLLM 模型加载格式。
export RL_ROLLOUT_LOAD_FORMAT="${RL_ROLLOUT_LOAD_FORMAT:-safetensors}"
# 是否启用即时采样模式。
export RL_ROLLOUT_ENFORCE_EAGER="${RL_ROLLOUT_ENFORCE_EAGER:-True}"
# 是否启用 vLLM 前缀缓存。
export RL_ROLLOUT_ENABLE_PREFIX_CACHING="${RL_ROLLOUT_ENABLE_PREFIX_CACHING:-False}"
# 是否启用 vLLM 分块预填充。
export RL_ROLLOUT_ENABLE_CHUNKED_PREFILL="${RL_ROLLOUT_ENABLE_CHUNKED_PREFILL:-True}"
# 是否在采样后释放 vLLM 缓存引擎。
export RL_ROLLOUT_FREE_CACHE_ENGINE="${RL_ROLLOUT_FREE_CACHE_ENGINE:-False}"
# FSDP 模型数据类型。
export RL_FSDP_MODEL_DTYPE="${RL_FSDP_MODEL_DTYPE:-bfloat16}"
# 是否在 actor/ref 进程中启用 torch.compile。
export RL_FSDP_USE_TORCH_COMPILE="${RL_FSDP_USE_TORCH_COMPILE:-False}"

case "$RUN_MODE" in
    val_only)
        # 仅打印命令 / 试运行时默认使用的 SFT 实验名。
        export SFT_EXPERIMENT_NAME="${SFT_EXPERIMENT_NAME:-qwen3-1.7b-etth1-sft-paper-valonly-20260326}"
        # 仅打印命令 / 试运行时默认使用的 SFT 输出目录。
        export SFT_SAVE_DIR="${SFT_SAVE_DIR:-$PROJECT_DIR/artifacts/checkpoints/sft/$SFT_EXPERIMENT_NAME}"
        # SFT 全局批大小。
        export SFT_TRAIN_BATCH_SIZE="${SFT_TRAIN_BATCH_SIZE:-12}"
        # SFT 每卡微批大小。
        export SFT_MICRO_BATCH_SIZE="${SFT_MICRO_BATCH_SIZE:-1}"
        # SFT epoch 数。
        export SFT_TOTAL_EPOCHS="${SFT_TOTAL_EPOCHS:-1}"
        # SFT 权重保存频率。
        export SFT_SAVE_FREQ="${SFT_SAVE_FREQ:-after_each_epoch}"
        # SFT 验证频率。
        export SFT_TEST_FREQ="${SFT_TEST_FREQ:-after_each_epoch}"
        # SFT 日志后端列表。
        export SFT_LOGGER="${SFT_LOGGER:-[\"console\"]}"
        # SFT 额外 Hydra 参数。
        SFT_EXTRA_ARGS=()

        # RL 实验名。
        export RL_EXP_NAME="${RL_EXP_NAME:-etth1_ot_qwen3_1_7b_rl_paper_valonly_20260326}"
        # RL 输出目录。
        export RL_TRAINER_LOCAL_DIR="${RL_TRAINER_LOCAL_DIR:-$PROJECT_DIR/artifacts/checkpoints/rl/$RL_EXP_NAME}"
        # RL 全局批大小。
        export RL_TRAIN_BATCH_SIZE="${RL_TRAIN_BATCH_SIZE:-$RL_NUM_GPUS}"
        # RL PPO 小批大小。
        export RL_PPO_MINI_BATCH_SIZE="${RL_PPO_MINI_BATCH_SIZE:-$RL_TRAIN_BATCH_SIZE}"
        # RL PPO 每卡微批大小。
        export RL_PPO_MICRO_BATCH_SIZE="${RL_PPO_MICRO_BATCH_SIZE:-1}"
        # RL log-prob 每卡微批大小。
        export RL_LOGPROB_MICRO_BATCH_SIZE="${RL_LOGPROB_MICRO_BATCH_SIZE:-1}"
        # 每个提示词采样的轨迹数。
        export RL_ROLLOUT_N="${RL_ROLLOUT_N:-1}"
        # RL 权重保存频率。
        export RL_SAVE_FREQ="${RL_SAVE_FREQ:-1}"
        # RL 验证频率。
        export RL_TEST_FREQ="${RL_TEST_FREQ:-1}"
        # RL epoch 数。
        export RL_TOTAL_EPOCHS="${RL_TOTAL_EPOCHS:-1}"
        # RL 日志后端列表。
        export RL_LOGGER="${RL_LOGGER:-[\"console\"]}"
        # RL 仅验证模式的额外 Hydra 参数。
        RL_EXTRA_ARGS=(
            "trainer.val_only=True"
            "trainer.val_before_train=True"
            "data.train_max_samples=32"
            "data.val_max_samples=256"
            "trainer.total_training_steps=1"
        )
        ;;
    smoke)
        # SFT 实验名。
        export SFT_EXPERIMENT_NAME="${SFT_EXPERIMENT_NAME:-qwen3-1.7b-etth1-sft-paper-smoke-20260326-gpu012}"
        # SFT 输出目录。
        export SFT_SAVE_DIR="${SFT_SAVE_DIR:-$PROJECT_DIR/artifacts/checkpoints/sft/$SFT_EXPERIMENT_NAME}"
        # SFT 全局批大小。
        export SFT_TRAIN_BATCH_SIZE="${SFT_TRAIN_BATCH_SIZE:-6}"
        # SFT 每卡微批大小。
        export SFT_MICRO_BATCH_SIZE="${SFT_MICRO_BATCH_SIZE:-1}"
        # SFT epoch 数。
        export SFT_TOTAL_EPOCHS="${SFT_TOTAL_EPOCHS:-1}"
        # SFT 权重保存频率。
        export SFT_SAVE_FREQ="${SFT_SAVE_FREQ:-after_each_epoch}"
        # SFT 验证频率。
        export SFT_TEST_FREQ="${SFT_TEST_FREQ:-after_each_epoch}"
        # SFT 日志后端列表。
        export SFT_LOGGER="${SFT_LOGGER:-[\"console\"]}"
        # SFT 冒烟测试的额外 Hydra 参数。
        SFT_EXTRA_ARGS=(
            "data.train_max_samples=24"
            "data.val_max_samples=8"
        )

        # RL 实验名。
        export RL_EXP_NAME="${RL_EXP_NAME:-etth1_ot_qwen3_1_7b_rl_paper_smoke_20260326}"
        # RL 输出目录。
        export RL_TRAINER_LOCAL_DIR="${RL_TRAINER_LOCAL_DIR:-$PROJECT_DIR/artifacts/checkpoints/rl/$RL_EXP_NAME}"
        # RL 全局批大小。
        export RL_TRAIN_BATCH_SIZE="${RL_TRAIN_BATCH_SIZE:-3}"
        # RL PPO 小批大小。
        export RL_PPO_MINI_BATCH_SIZE="${RL_PPO_MINI_BATCH_SIZE:-3}"
        # RL PPO 每卡微批大小。
        export RL_PPO_MICRO_BATCH_SIZE="${RL_PPO_MICRO_BATCH_SIZE:-1}"
        # RL log-prob 每卡微批大小。
        export RL_LOGPROB_MICRO_BATCH_SIZE="${RL_LOGPROB_MICRO_BATCH_SIZE:-1}"
        # 每个提示词采样的轨迹数。
        export RL_ROLLOUT_N="${RL_ROLLOUT_N:-8}"
        # RL 权重保存频率。
        export RL_SAVE_FREQ="${RL_SAVE_FREQ:-1}"
        # RL 验证频率。
        export RL_TEST_FREQ="${RL_TEST_FREQ:-5}"
        # RL epoch 数。
        export RL_TOTAL_EPOCHS="${RL_TOTAL_EPOCHS:-1}"
        # RL 日志后端列表。
        export RL_LOGGER="${RL_LOGGER:-[\"console\"]}"
        # RL 冒烟测试的额外 Hydra 参数。
        RL_EXTRA_ARGS=(
            "data.train_max_samples=3"
            "data.val_max_samples=1"
            "trainer.total_training_steps=1"
        )
        ;;
    train)
        # SFT 实验名。
        export SFT_EXPERIMENT_NAME="${SFT_EXPERIMENT_NAME:-qwen3-1.7b-etth1-sft-paper-formal-20260326}"
        # SFT 输出目录。
        export SFT_SAVE_DIR="${SFT_SAVE_DIR:-$PROJECT_DIR/artifacts/checkpoints/sft/$SFT_EXPERIMENT_NAME}"
        # SFT 全局批大小。
        export SFT_TRAIN_BATCH_SIZE="${SFT_TRAIN_BATCH_SIZE:-12}"
        # SFT 每卡微批大小。
        export SFT_MICRO_BATCH_SIZE="${SFT_MICRO_BATCH_SIZE:-1}"
        # SFT epoch 数。
        export SFT_TOTAL_EPOCHS="${SFT_TOTAL_EPOCHS:-1}"
        # SFT 权重保存频率。
        export SFT_SAVE_FREQ="${SFT_SAVE_FREQ:-after_each_epoch}"
        # SFT 验证频率。
        export SFT_TEST_FREQ="${SFT_TEST_FREQ:-after_each_epoch}"
        # SFT 日志后端列表。
        export SFT_LOGGER="${SFT_LOGGER:-[\"console\"]}"
        # 正式 SFT 的额外 Hydra 参数。
        SFT_EXTRA_ARGS=()

        # RL 实验名。
        export RL_EXP_NAME="${RL_EXP_NAME:-etth1_ot_qwen3_1_7b_rl_paper_20260326}"
        # RL 输出目录。
        export RL_TRAINER_LOCAL_DIR="${RL_TRAINER_LOCAL_DIR:-$PROJECT_DIR/artifacts/checkpoints/rl/$RL_EXP_NAME}"
        # RL 训练迭代中的 prompt batch。
        # 论文目标是 64 trajectories / iter；在 3 张训练卡 + rollout_n=8 下无法精确整除，
        # 因此正式默认改成 9 prompts / iter => 72 trajectories / iter，作为最接近论文的可行值。
        export RL_TRAIN_BATCH_SIZE="${RL_TRAIN_BATCH_SIZE:-9}"
        # RL PPO 小批大小（prompt 视角）。
        # 取 3 后，verl 内部会做 * rollout_n / RL_NUM_GPUS 的归一化：
        # normalized_ppo_mini_batch_size = 3 * 8 / 3 = 8 trajectories / GPU-group。
        export RL_PPO_MINI_BATCH_SIZE="${RL_PPO_MINI_BATCH_SIZE:-3}"
        # RL PPO 每卡微批大小（trajectory 视角）。
        # 取 2 后，actor 侧等效 gradient accumulation = 8 / 2 = 4，与论文一致。
        export RL_PPO_MICRO_BATCH_SIZE="${RL_PPO_MICRO_BATCH_SIZE:-2}"
        # RL log-prob 每卡微批大小。
        export RL_LOGPROB_MICRO_BATCH_SIZE="${RL_LOGPROB_MICRO_BATCH_SIZE:-1}"
        # 每个提示词采样的轨迹数。
        export RL_ROLLOUT_N="${RL_ROLLOUT_N:-8}"
        # RL 权重保存频率。
        export RL_SAVE_FREQ="${RL_SAVE_FREQ:-50}"
        # RL 验证频率。
        export RL_TEST_FREQ="${RL_TEST_FREQ:-20}"
        # RL epoch 数。
        export RL_TOTAL_EPOCHS="${RL_TOTAL_EPOCHS:-1}"
        # RL 日志后端列表。
        export RL_LOGGER="${RL_LOGGER:-[\"console\"]}"
        # 正式 RL 的额外 Hydra 参数。
        RL_EXTRA_ARGS=()
        ;;
    *)
        echo "Unsupported RUN_MODE: $RUN_MODE"
        return 1 2>/dev/null || exit 1
        ;;
esac
