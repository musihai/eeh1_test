# Cast-R1-TS

截至 `2026-03-25`，当前仓库代码主线已经修到下面这版：

- ETTh1 输入链路改回 `multivariate`，不再是旧的 OT-only 单变量 prompt / model request。
- diagnostic stage 改成 `plan-driven subset tools`，不再固定全 5-tool。
- `paper_strict` 只约束 `<think><answer>` 协议，不再顺手禁掉 Turn 3 的 `local_refine` 监督。
- reward 改成 `MSE-first + weak structural tie-break`，不会再让结构项主导 expert 排序。
- Turn 3 refinement prompt 现在显式给出 `Final Allowed Forecast Timestamp`，不再只靠“数到 96 行”来停，专门约束 over-generation。
- 最终 `<answer>` 协议现在兼容两种格式：
  - 优先 `YYYY-MM-DD HH:MM:SS value`
  - 兼容 plain value-only
  - 不允许混用

当前最近一轮相关回归测试状态：

- `48 passed`

## 当前要点

代码已经是新的，但磁盘上的部分旧数据集 / checkpoint 仍然来自旧链路。

所以要区分两类用途：

- `协议 / agent flow / I/O debug`
  可以直接复用当前仓库里的 multivariate debug 数据和现有 checkpoint。
- `正式 paper-aligned 指标`
  需要按当前代码重新全量重建 teacher-eval / step-wise SFT / curriculum RL 数据，再重新训练。

当前建议直接用 `mv1` 这套正式重建后的资源：

- RL curriculum 数据：
  `dataset/ett_rl_etth1_mv1/`
- 当前已验证可加载的 SFT warm start：
  `artifacts/checkpoints/sft/time_series_forecast_sft_mv1_tsfix_20260324/global_step_66/hf_merged`
- 但下一轮正式训练不要直接从这个 checkpoint 进入 RL：
  需要先用当前最新 prompt 重新生成 step-wise SFT parquet，再重训一版新的 warm start
- 本地 multivariate experts：
  `recipe/time_series_forecast/models/patchtst/`
  `recipe/time_series_forecast/models/itransformer/`
  `recipe/time_series_forecast/models/chronos-2/`

## 最新验证结果

最近两轮 `val_only=32` 的关键差异：

- 只修 `Turn 3 token budget` 之后：
  `logs/debug/mv1tsfix_val32_budgetfix_20260324/eval_step_aggregate.jsonl`
  - `final_answer_accept_ratio = 0.75`
  - `missing_answer_close_tag_count = 7`
  - `pred_len_gt_96_ratio = 0.25`
- 再加上 `Final Allowed Forecast Timestamp` 约束之后：
  `logs/debug/mv1tsfix_val32_terminalfix_20260324/eval_step_aggregate.jsonl`
  - `final_answer_accept_ratio = 0.9375`
  - `missing_answer_close_tag_count = 0`
  - `pred_len_gt_96_ratio = 0.0625`
  - `validation_reward_mean = 0.3016`

这说明：

- `Turn 3` 的大面积截断问题已经修掉了。
- 还剩少量 `arima` over-generation 残差，但已经不是旧的 budget/clipping 主因。
- 下一轮正式训练应该从 `step-wise SFT` 重建重新开始，让新 prompt 被 checkpoint 真正学进去。

## 环境

```bash
cd /data/linyujie/Cast-R1-TS-main/Cast-R1-TS-main
source /data/linyujie/miniconda3/etc/profile.d/conda.sh
conda activate cast-r1-ts
export PYTHONPATH=$PWD:$PYTHONPATH
```

## 4x RTX 5090 推荐配置

当前机器是 `4 x RTX 5090 32GB`。对 `Qwen3-1.7B + 单独预测服务`，当前推荐：

- `GPU 0,1,2`：RL / val-only / SFT
- `GPU 3`：model server

调试时不要开太大，也不要小到没有代表性。当前推荐的 moderate debug 参数：

- `NUM_GPUS=3`
- `SERVER_GPU_ID=3`
- `RL_MAX_PROMPT_LENGTH=9216`
- `RL_MAX_RESPONSE_LENGTH=3072`
- `RL_FSDP_MODEL_DTYPE=bfloat16`
- `RL_FSDP_USE_TORCH_COMPILE=False`
- `RL_ACTOR_MAX_TOKEN_LEN_PER_GPU=12288`
- `RL_ROLLOUT_MAX_MODEL_LEN=12288`
- `RL_ROLLOUT_GPU_MEMORY_UTILIZATION=0.25`
- `RL_ROLLOUT_MAX_BATCHED_TOKENS=12288`
- `RL_ROLLOUT_MAX_NUM_SEQS=3`
- `data.val_max_samples=4`
- `data.train_max_samples=4`

这套配置的目的不是追求吞吐，而是保证：

- 每步输入输出能完整落日志
- 显存占用不过高
- 一次能看到不止 1 条样本

补充说明：

- 当前 multivariate RL prompt 的实测 token 长度大约在 `8.1k - 8.3k`
- 当前 paper-style Turn 3 `96` 行 timestamp-value 最终答案，实测 token 长度中位数约 `2718`，`p95` 约 `2730`
- 旧的 `MAX_PROMPT_LENGTH=4096` / `ROLLOUT_MAX_MODEL_LEN=8192` 会把样本全部过滤掉
- 旧的 `RL_MAX_RESPONSE_LENGTH=2048` 会把 Turn 3 在 `~1408` token 处截断，直接导致 `missing_answer_close_tag`
- 上面这组参数是按 `4 x RTX 5090 32GB` 和 `Qwen3-1.7B` 调过的一组“刚好够用”的值，不建议再缩回旧值

## 从头检查一轮最小真实链路

### Step 1. 启动预测服务

```bash
cd /data/linyujie/Cast-R1-TS-main/Cast-R1-TS-main
source /data/linyujie/miniconda3/etc/profile.d/conda.sh
conda activate cast-r1-ts
export PYTHONPATH=$PWD:$PYTHONPATH

CUDA_VISIBLE_DEVICES=3 \
python recipe/time_series_forecast/model_server.py --port 8994
```

健康检查：

```bash
curl -s http://127.0.0.1:8994/health
```

期望：

- `models_loaded.patchtst = true`
- `models_loaded.itransformer = true`
- `models_loaded.chronos2 = true`

### Step 2. 跑一轮 val-only debug

```bash
cd /data/linyujie/Cast-R1-TS-main/Cast-R1-TS-main
source /data/linyujie/miniconda3/etc/profile.d/conda.sh
conda activate cast-r1-ts
export PYTHONPATH=$PWD:$PYTHONPATH

ray stop --force

RUN_MODE=val_only \
TRAIN_GPU_IDS=0,1,2 \
NUM_GPUS=3 \
MODEL_SERVICE_PORT=8994 \
MODEL_SERVICE_URL=http://127.0.0.1:8994 \
RL_MODEL_PATH=$PWD/artifacts/checkpoints/sft/time_series_forecast_sft_mv1_tsfix_20260324/global_step_66/hf_merged \
RL_TRAIN_FILES=$PWD/dataset/ett_rl_etth1_mv1/train.jsonl \
RL_VAL_FILES=$PWD/dataset/ett_rl_etth1_mv1/val.jsonl \
RL_MAX_PROMPT_LENGTH=9216 \
RL_MAX_RESPONSE_LENGTH=3072 \
RL_FSDP_MODEL_DTYPE=bfloat16 \
RL_FSDP_USE_TORCH_COMPILE=False \
RL_ACTOR_MAX_TOKEN_LEN_PER_GPU=12288 \
RL_ROLLOUT_MAX_MODEL_LEN=12288 \
RL_ROLLOUT_GPU_MEMORY_UTILIZATION=0.25 \
RL_ROLLOUT_MAX_BATCHED_TOKENS=12288 \
RL_ROLLOUT_MAX_NUM_SEQS=3 \
RL_EXP_NAME=etth1_mv1_valonly_debug_20260324 \
RL_TRAINER_LOCAL_DIR=$PWD/artifacts/checkpoints/rl/etth1_mv1_valonly_debug_20260324 \
DEBUG_CHAIN=1 \
RUN_TAG=mv1_valonly_debug_$(date +%Y%m%d_%H%M%S) \
TS_CHAIN_DEBUG_FILE=$PWD/logs/debug/$RUN_TAG/ts_chain_debug.jsonl \
TS_MIN_EVAL_AGG_FILE=$PWD/logs/debug/$RUN_TAG/eval_step_aggregate.jsonl \
TS_MIN_EVAL_SAMPLE_FILE=$PWD/logs/debug/$RUN_TAG/eval_step_samples.jsonl \
bash examples/time_series_forecast/run_qwen3-1.7B.sh \
  trainer.val_before_train=True \
  trainer.val_only=True \
  trainer.log_val_generations=4 \
  data.train_max_samples=4 \
  data.val_max_samples=4 \
  trainer.total_training_steps=1
```

### Step 3. 核对每一步输入输出

看这三个文件：

- `logs/debug/<RUN_TAG>/ts_chain_debug.jsonl`
- `logs/debug/<RUN_TAG>/eval_step_aggregate.jsonl`
- `logs/debug/<RUN_TAG>/eval_step_samples.jsonl`

重点核对：

- Turn 1 user prompt 里要有：
  - multivariate historical window
  - `### Diagnostic Plan`
  - 当前 turn 暴露的 diagnostic tool 子集
- Turn 1 assistant tool calls：
  - 只能调用当前 batch 暴露的 feature tools
  - 不应该直接调用 `predict_time_series`
- Turn 2 routing：
  - 只能调用一次 `predict_time_series`
  - `prediction_model_defaulted_ratio` 应为 `0.0`
  - `prediction_tool_error_count` 应为 `0`
- Turn 3 refinement：
  - 不应该再调用工具
  - 最终输出必须是 `<think>...</think><answer>...</answer>`
  - `<answer>` 可为 `timestamp-value` 或 `value-only`
  - 不能混用两种格式

聚合结果建议至少看：

- `final_answer_accept_ratio`
- `strict_length_match_ratio`
- `prediction_call_count_mean`
- `illegal_turn3_tool_call_count_mean`
- `selected_model_distribution`
- `prediction_requested_model_distribution`
- `format_failure_reason_distribution`

## 正式重建前的说明

如果要做正式 paper-aligned rerun，不要直接把旧目录当正式结果：

- `dataset/ett_rl_etth1_paper_same2/`
- `dataset/ett_rl_etth1_paper_aligned_ot_curriculum_same2/`
- `dataset/ett_sft_etth1_runtime_ot_teacher200_paper_same4_stepwise_heuristicroute/`
- `artifacts/checkpoints/sft/time_series_forecast_sft_paper_strict_heuristicroute_4gpu_20260323/`

这些目录里有相当一部分是在旧的 univariate / pre-planning / pre-refine-fix / pre-reward-fix 链路下生成的。

`mv1` 这条正式重建链路里，已经完成的部分是：

1. teacher-eval 重建完成
2. curated SFT 重建完成
3. step-wise SFT parquet 重建完成
4. `mv1_tsfix` SFT 重训完成

但由于 `2026-03-25` 又新增了 `Final Allowed Forecast Timestamp` 的 Turn 3 prompt 约束，下一轮正式训练不要直接接着 RL，应该从下面这一步重新开始：

1. 用 `dataset/ett_sft_etth1_runtime_teacher200_mv1/*.jsonl` 重新生成新的 step-wise SFT parquet
2. 用新的 parquet 重训一版 `mv1_terminalfix` warm start
3. 先跑 `val_only`
4. 再进入 curriculum RL

补充说明：

- 新训练完的 SFT checkpoint，默认一定会有
  `global_step_*/huggingface/`
- `global_step_*/hf_merged/` 只有在你额外执行 merge 之后才会出现
- 所以如果你刚训完 SFT，后续 `val_only / RL` 最稳妥的是先直接用
  `global_step_*/huggingface`
  如果后面再手工 merge 成 `hf_merged`，两种路径都可用

## 当前保留的关键主线代码

- `recipe/time_series_forecast/build_etth1_rl_dataset.py`
- `recipe/time_series_forecast/build_etth1_high_quality_sft.py`
- `recipe/time_series_forecast/build_etth1_sft_dataset.py`
- `recipe/time_series_forecast/diagnostic_policy.py`
- `recipe/time_series_forecast/prompts.py`
- `recipe/time_series_forecast/reward.py`
- `recipe/time_series_forecast/reward_protocol.py`
- `recipe/time_series_forecast/time_series_forecast_agent_flow.py`
- `recipe/time_series_forecast/model_server.py`
- `recipe/time_series_forecast/retrain_expert_models_train_split.py`
- `examples/time_series_forecast/run_qwen3-1.7B_sft.sh`
- `examples/time_series_forecast/run_qwen3-1.7B.sh`
- `examples/time_series_forecast/run_qwen3-1.7B_curriculum.sh`
- `examples/time_series_forecast/configs/etth1_ot_qwen3_gpu012.sh`
