# Cast-R1-TS（ETTh1 单变量 `OT`）

这个仓库当前维护的是 `ETTh1` 单变量 `OT` 预测链路，目标是把实现收敛到 [Cast-R1 论文 PDF](/data/linyujie/Cast-R1-TS-main/Cast-R1-TS-main/Tao%20等%20-%202026%20-%20Cast-R1%20Learning%20Tool-Augmented%20Sequential%20Decision%20Policies%20for%20Time%20Series%20Forecasting.pdf) 的主线：

- `Turn 1`：诊断 / 特征提取
- `Turn 2`：单模型 routing
- `Turn 3`：agent 基于 selected forecast 做 reasoning / refinement / final output

当前推荐训练主线是：

`工具服务 -> 基础 RL jsonl -> 第一轮 teacher eval -> curriculum RL jsonl -> 第二轮 teacher200 -> SFT -> RL`

## 当前实现状态

### 已对齐的部分

- stage-aware prompt 已落地：`Turn 1` 只能特征工具，`Turn 2` 只能 `predict_time_series`，`Turn 3` 不允许继续调用工具。
- `Turn 2` 是单模型 routing，不再把“同时拿多个 forecast 再比较”当主流程。
- `Turn 3` 明确要求 final answer 基于 selected model prediction，只允许小幅、证据驱动的 refinement。
- runtime 在拿到 prediction 之后会拒绝后续 tool invocation，并记录 `illegal_turn3_tool_call_count`。
- reward 已回到论文式 composite reward：
  - normalized + log-transformed MSE 主项
  - turning point / structural alignment
  - trend / seasonality consistency
  - format validity
  - length consistency
- SFT builder 不再固定五个 feature tools 全调用，改成 state-dependent 选择。
- RL builder 已补齐离线元信息：
  - `reference_teacher_error`
  - `reference_teacher_error_band`
  - `normalized_permutation_entropy`
  - `normalized_permutation_entropy_band`
  - `quality_issue_flag`
  - `difficulty_stage`

### 还没有完全对齐论文的部分

下面这些已经明确识别出来，但这次还没继续改：

1. runtime memory 仍然是紧凑状态实现，主要依赖 `history_analysis + selected prediction`，还不是显式的结构化 memory object。
2. SFT 的 `Turn 3` 目标现在已经显式写出：
   - `turn3_target_type`
   - `turn3_trigger_reason`
   - `refine_ops_signature`
   - `selected_feature_tool_signature`
   但当前的 refine target 仍然是基于规则器合成的局部修正，不是论文原文给出的显式数值修正器。
3. `teacher200` 的 curator 现在已经加入了按 `best_model` 的平衡抽样，但还不是完整的 curriculum-aware curator；难度分桶和 teacher-error 分层还没有一并进入 selection。
4. RL jsonl 已经带了 difficulty metadata，但在线 trainer 还没有接入按 `difficulty_stage` / `entropy_band` 分桶采样。
5. 调试聚合已经能看到工具链计数、工具调用序列和 refinement 结果，但更细的 turn-level reasoning 仍然要靠 chain debug 排查。

如果你想继续往论文靠，这五项就是下一轮的主方向。

## 环境

```bash
cd /data/linyujie/Cast-R1-TS-main/Cast-R1-TS-main
conda activate cast-r1-ts
export PYTHONPATH=$PWD:$PYTHONPATH
```

如果当前 shell 不能直接 `conda activate`，下面所有命令都可以改成 `conda run -n cast-r1-ts ...`。

## 当前默认配置

统一 profile：

```bash
examples/time_series_forecast/configs/etth1_ot_qwen3_gpu012.sh
```

当前 profile / launcher 默认值已经和这份 README 对齐：

- RL 默认 train 文件：`dataset/ett_rl_etth1_paper_aligned_ot_curriculum_same2/train_stage123.jsonl`
- RL 稳定显存配置：`gpu_memory_utilization=0.22`、`max_model_len=8192`、`max_batched_tokens=4096`
- RL 稳定采样配置：`rollout.n=1`、`temperature=0.3`、`response_length=3072`
- SFT 默认 parquet：`dataset/ett_sft_etth1_runtime_ot_teacher200_paper_same2/{train,val}.parquet`

关键脚本：

- 工具服务：[start_model_server.sh](recipe/time_series_forecast/start_model_server.sh)
- RL 数据构造：[build_etth1_rl_dataset.py](recipe/time_series_forecast/build_etth1_rl_dataset.py)
- 高质量 teacher / SFT 构造：[build_etth1_high_quality_sft.py](recipe/time_series_forecast/build_etth1_high_quality_sft.py)
- SFT parquet 构造：[build_etth1_sft_dataset.py](recipe/time_series_forecast/build_etth1_sft_dataset.py)
- SFT 训练：[run_qwen3-1.7B_sft.sh](examples/time_series_forecast/run_qwen3-1.7B_sft.sh)
- RL 训练：[run_qwen3-1.7B.sh](examples/time_series_forecast/run_qwen3-1.7B.sh)

默认约定：

- `GPU3` 跑工具服务
- `GPU0,1,2` 跑 SFT / RL
- 基础模型默认是 `/data/linyujie/models/Qwen3-1.7B`
- `Chronos2 / PatchTST / iTransformer` 通过工具服务提供
- `ARIMA` 是本地解析专家，不会显示在 `/models` HTTP 列表里

## 完整流程

### 0. 进入项目

```bash
cd /data/linyujie/Cast-R1-TS-main/Cast-R1-TS-main
conda activate cast-r1-ts
export PYTHONPATH=$PWD:$PYTHONPATH
```

### 1. 启动工具服务

这一步对 RL 训练是必需的；对 Step 3 / Step 5 的 teacher eval 则不是必需，因为当前默认推荐走本地多 GPU teacher 评估。

如果你准备后面直接跑 RL，单独开一个终端执行：

```bash
PROFILE_PATH=examples/time_series_forecast/configs/etth1_ot_qwen3_gpu012.sh \
bash recipe/time_series_forecast/start_model_server.sh 3 8994
```

服务健康检查：

```bash
curl http://127.0.0.1:8994/health
curl http://127.0.0.1:8994/models
```

说明：

- `/models` 里只会列出远程加载的 `chronos2 / patchtst / itransformer`
- `arima` 走本地实现，不依赖这个列表

### 2. 生成基础 RL jsonl

这一步先生成不带 teacher error 的基础 RL 数据。输出目录我建议新开一套，别继续混旧产物。

```bash
python recipe/time_series_forecast/build_etth1_rl_dataset.py \
  --csv-path dataset/ETT-small/ETTh1.csv \
  --output-dir dataset/ett_rl_etth1_paper_same2 \
  --lookback-window 96 \
  --forecast-horizon 96 \
  --target-column OT \
  --train-rows 12251 \
  --val-rows 1913 \
  --test-rows 3256
```

输出：

- `dataset/ett_rl_etth1_paper_same2/train.jsonl`
- `dataset/ett_rl_etth1_paper_same2/train_stage1.jsonl`
- `dataset/ett_rl_etth1_paper_same2/train_stage12.jsonl`
- `dataset/ett_rl_etth1_paper_same2/train_stage123.jsonl`
- `dataset/ett_rl_etth1_paper_same2/val.jsonl`
- `dataset/ett_rl_etth1_paper_same2/test.jsonl`

说明：

- `train_stage1.jsonl`：只含 `curriculum_stage=easy`
- `train_stage12.jsonl`：含 `easy + medium`
- `train_stage123.jsonl`：完整 train split
- 在 Step 2 没有 teacher metadata 时，stage 主要由 entropy 决定；正式 curriculum 以 Step 4 生成的 staged files 为准

### 3. 第一轮 teacher eval / teacher200（为 teacher error 提供来源）

先基于基础 RL jsonl 跑一轮 teacher scoring，生成 `teacher_eval.jsonl` 和第一版 teacher200。
当前推荐直接走本地多 GPU teacher 评估，不再默认依赖 HTTP 模型服务。

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python recipe/time_series_forecast/build_etth1_high_quality_sft.py \
  --train-jsonl  dataset/ett_rl_etth1_paper_same2/train.jsonl \
  --val-jsonl  dataset/ett_rl_etth1_paper_same2/val.jsonl \
  --test-jsonl  dataset/ett_rl_etth1_paper_same2/test.jsonl \
  --output-dir dataset/ett_sft_etth1_runtime_teacher200_paper_same2 \
  --train-target-samples 200 \
  --val-target-samples 64 \
  --test-target-samples 128 \
  --train-candidate-samples 600 \
  --val-candidate-samples 192 \
  --test-candidate-samples 256 \
  --models patchtst,chronos2,itransformer,arima \
  --predictor-mode local \
  --predictor-device cuda \
  --num-workers 4 \
  --local-batch-size 256 \
  --resume-teacher-eval \
  --max-concurrency 4
```

说明：

- `--resume-teacher-eval` 默认开启，会复用已有的 `*_teacher_eval.jsonl`，重复执行时不重算已完成样本。
- `--local-batch-size` 会对 `patchtst` / `itransformer` 使用 batched inference；`chronos2` / `arima` 仍按单样本路径走。
- `--num-workers 2` 会把样本分给两个本地 worker，并把可见 GPU 分成两组；如果 4 张卡显存都足够，可以尝试 `--num-workers 4`。
- 在 `predictor-mode=local` 且 `num-workers=1` 时，builder 会按模型 round-robin 分配 GPU；在 `num-workers>1` 时，先按 worker 分设备组，再由每个 worker 自己给模型分配设备。
- 如果你想手动指定设备来源，可以额外传：
  - `--predictor-devices cuda:0,cuda:1,cuda:2,cuda:3`

这一步会产出两类文件：

- `*_teacher_eval.jsonl`：teacher 评分和 best-model 信息
- `train/val/test.parquet`：第一版 SFT parquet

关键文件：

- `dataset/ett_sft_etth1_runtime_teacher200_paper_same2/train_teacher_eval.jsonl`
- `dataset/ett_sft_etth1_runtime_teacher200_paper_same2/val_teacher_eval.jsonl`
- `dataset/ett_sft_etth1_runtime_teacher200_paper_same2/test_teacher_eval.jsonl`

这一步现在默认会写两套 teacher eval：

- `*_teacher_eval.jsonl`：全量 split 的 teacher metadata，给 curriculum 回灌使用
- `*_teacher_eval_curated.jsonl`：按目标样本数裁出来的 curated 子集

建议立刻检查：

- `*_teacher_eval.jsonl` 的行数应接近对应 split 的样本数
- `metadata.json` 里的 `train_teacher_distribution` / `val_teacher_distribution` 不应该只剩一个模型
- `metadata.json` 里的 `resume_teacher_eval`、`num_workers`、`local_batch_size`、`local_worker_device_groups` 应和你的命令一致

### 4. 用 teacher metadata 重建 curriculum RL jsonl

这一步把 `reference_teacher_error` 等元信息灌回 RL jsonl，给 curriculum 和后续分析使用。

```bash
python recipe/time_series_forecast/build_etth1_rl_dataset.py \
  --csv-path dataset/ETT-small/ETTh1.csv \
  --output-dir dataset/ett_rl_etth1_paper_aligned_ot_curriculum_same2 \
  --lookback-window 96 \
  --forecast-horizon 96 \
  --target-column OT \
  --train-rows 12251 \
  --val-rows 1913 \
  --test-rows 3256 \
  --train-teacher-metadata-jsonl dataset/ett_sft_etth1_runtime_teacher200_paper_same2/train_teacher_eval.jsonl \
  --val-teacher-metadata-jsonl dataset/ett_sft_etth1_runtime_teacher200_paper_same2/val_teacher_eval.jsonl \
  --test-teacher-metadata-jsonl dataset/ett_sft_etth1_runtime_teacher200_paper_same2/test_teacher_eval.jsonl
```

这一步新增加的字段包括：

- `reference_teacher_error`
- `reference_teacher_error_band`
- `normalized_permutation_entropy`
- `normalized_permutation_entropy_band`
- `quality_issue_flag`
- `difficulty_stage`
- `curriculum_stage`
- `curriculum_band`

同时会为 train split 额外写出：

- `train_stage1.jsonl`
- `train_stage12.jsonl`
- `train_stage123.jsonl`

这一步会检查 teacher metadata 覆盖率；如果你还在拿旧的稀疏 `teacher_eval.jsonl` 来灌 curriculum，会直接报错。正常情况下应看到类似：

```text
[RL-DATA] split=train teacher metadata coverage: 12060/12060 (100.00%)
```

### 5. 第二轮 teacher200（推荐正式用于 SFT 的版本）

现在再基于带 curriculum metadata 的 RL jsonl 重跑一轮 teacher200。这样后续 SFT / RL 使用的上游数据就是同一套口径。

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python recipe/time_series_forecast/build_etth1_high_quality_sft.py \
  --train-jsonl dataset/ett_rl_etth1_paper_aligned_ot_curriculum_same2/train.jsonl \
  --val-jsonl dataset/ett_rl_etth1_paper_aligned_ot_curriculum_same2/val.jsonl \
  --test-jsonl dataset/ett_rl_etth1_paper_aligned_ot_curriculum_same2/test.jsonl \
  --output-dir dataset/ett_sft_etth1_runtime_ot_teacher200_paper_same2 \
  --train-target-samples 200 \
  --val-target-samples 64 \
  --test-target-samples 128 \
  --train-candidate-samples 600 \
  --val-candidate-samples 192 \
  --test-candidate-samples 256 \
  --models patchtst,chronos2,itransformer,arima \
  --predictor-mode local \
  --predictor-device cuda \
  --train-eval-samples 600 \
  --val-eval-samples 192 \
  --test-eval-samples 256 \
  --num-workers 4 \
  --local-batch-size 256 \
  --resume-teacher-eval \
  --max-concurrency 4
```

推荐后续统一使用这个目录：

- `dataset/ett_sft_etth1_runtime_ot_teacher200_paper_same2/train.parquet`
- `dataset/ett_sft_etth1_runtime_ot_teacher200_paper_same2/val.parquet`
- `dataset/ett_sft_etth1_runtime_ot_teacher200_paper_same2/test.parquet`

这一步建议检查：

- `metadata.json` 里的 `train_teacher_distribution` / `val_teacher_distribution` 是否仍然塌到单模型
- `metadata.json` 里的 `train_teacher_eval_records=600`、`val_teacher_eval_records=192`、`test_teacher_eval_records=256`
- `metadata.json` 里的 `local_worker_device_groups` 是否符合预期
- `train.parquet` / `val.parquet` 是否带有：
  - `turn3_target_type`
  - `turn3_trigger_reason`
  - `refine_ops_signature`
  - `selected_feature_tool_signature`
- `metadata.json` 里的 `train_turn3_target_type_distribution_before_balance` 和 `train_turn3_target_type_distribution`
  是否显示 train parquet 已做轻量 keep/refine 重平衡

### 6. 跑 SFT

```bash
MODEL_PATH=/data/linyujie/models/Qwen3-1.7B \
SAVE_DIR=$PWD/artifacts/checkpoints/sft/time_series_forecast_sft_teacher200_paper_same2 \
EXPERIMENT_NAME=time_series_forecast_sft_teacher200_paper_same2 \
PROFILE_PATH=examples/time_series_forecast/configs/etth1_ot_qwen3_gpu012.sh \
RUN_MODE=train \
SFT_TRAIN_FILES=$PWD/dataset/ett_sft_etth1_runtime_ot_teacher200_paper_same2/train.parquet \
SFT_VAL_FILES=$PWD/dataset/ett_sft_etth1_runtime_ot_teacher200_paper_same2/val.parquet \
bash examples/time_series_forecast/run_qwen3-1.7B_sft.sh \
  trainer.resume_mode=disable
```

SFT 输出目录：

```bash
artifacts/checkpoints/sft/time_series_forecast_sft_teacher200_paper_same2/
```

如果你直接调用 `examples/time_series_forecast/run_qwen3-1.7B_sft.sh` 而不额外覆盖数据路径，脚本默认也会指向这套 `paper_same2` parquet。

RL 要用的模型路径必须指向：

```bash
artifacts/checkpoints/sft/time_series_forecast_sft_teacher200_paper_same2/<global_step_x>/huggingface
```

说明：

- 当前 `build_etth1_sft_dataset.py` 默认会对 `train.parquet` 做轻量 `turn3_target_type` 重平衡
- 默认目标是让 `local_refine` 在 train split 中至少达到约 `30%`
- `val.parquet` / `test.parquet` 不做这个重平衡，保留自然分布

### 7. RL smoke test

这个只用于确认链路和显存配置没问题，不用于判断方案是否成功。

```bash
cd /data/linyujie/Cast-R1-TS-main/Cast-R1-TS-main
conda activate cast-r1-ts
export PYTHONPATH=$PWD:$PYTHONPATH

unset PYTORCH_CUDA_ALLOC_CONF
ray stop --force

DEBUG_CHAIN=1 \
TS_CHAIN_DEBUG_FILE=$PWD/logs/debug/ts_chain_debug_smoke_v2.jsonl \
RL_TRAIN_FILES=$PWD/dataset/ett_rl_etth1_paper_aligned_ot_curriculum_same2/train.jsonl \
RL_VAL_FILES=$PWD/dataset/ett_rl_etth1_paper_aligned_ot_curriculum_same2/val.jsonl \
RL_TEMPERATURE=0.3 \
RL_ROLLOUT_GPU_MEMORY_UTILIZATION=0.22 \
RL_ROLLOUT_MAX_MODEL_LEN=8192 \
RL_ROLLOUT_MAX_BATCHED_TOKENS=4096 \
RL_ROLLOUT_N=1 \
RL_MAX_RESPONSE_LENGTH=3072 \
RL_ACTOR_MAX_TOKEN_LEN_PER_GPU=6144 \
MODEL_PATH=$PWD/artifacts/checkpoints/sft/time_series_forecast_sft_teacher200_paper_same2/global_step_11/huggingface \
PROFILE_PATH=examples/time_series_forecast/configs/etth1_ot_qwen3_gpu012.sh \
RUN_MODE=train \
RL_EXP_NAME=etth1_ot_qwen3_1_7b_rl_smoke_paper_same2 \
bash examples/time_series_forecast/run_qwen3-1.7B.sh \
  trainer.total_training_steps=20 \
  trainer.test_freq=5 \
  trainer.resume_mode=disable \
  trainer.log_val_generations=0
```

把上面的 `global_step_11` 改成你实际存在的 SFT 导出目录。

### 8. RL 机制验证

这一步才是当前实现该跑的主验证命令。推荐先按 curriculum 顺序使用：

1. `train_stage1.jsonl`
2. `train_stage12.jsonl`
3. `train_stage123.jsonl`

如果你只是做一次机制验证，也可以直接先用 `train_stage123.jsonl`。推荐先用 `rollout.n=1` 做低方差验证。

如果你直接调用 `examples/time_series_forecast/run_qwen3-1.7B.sh` 而不额外覆盖 `RL_TRAIN_FILES`，脚本默认也会指向 `train_stage123.jsonl`。

```bash
cd /data/linyujie/Cast-R1-TS-main/Cast-R1-TS-main
conda activate cast-r1-ts
export PYTHONPATH=$PWD:$PYTHONPATH

unset PYTORCH_CUDA_ALLOC_CONF
ray stop --force

DEBUG_CHAIN=0 \
TS_CHAIN_DEBUG_FILE=$PWD/logs/debug/ts_chain_debug_eval20_paper_same2.jsonl \
RL_TRAIN_FILES=$PWD/dataset/ett_rl_etth1_paper_aligned_ot_curriculum_same2/train_stage123.jsonl \
RL_VAL_FILES=$PWD/dataset/ett_rl_etth1_paper_aligned_ot_curriculum_same2/val.jsonl \
RL_TEMPERATURE=0.3 \
RL_ROLLOUT_GPU_MEMORY_UTILIZATION=0.22 \
RL_ROLLOUT_MAX_MODEL_LEN=8192 \
RL_ROLLOUT_MAX_BATCHED_TOKENS=4096 \
RL_ROLLOUT_N=1 \
RL_MAX_RESPONSE_LENGTH=3072 \
RL_ACTOR_MAX_TOKEN_LEN_PER_GPU=6144 \
MODEL_PATH=$PWD/artifacts/checkpoints/sft/time_series_forecast_sft_teacher200_paper_same2/global_step_11/huggingface \
PROFILE_PATH=examples/time_series_forecast/configs/etth1_ot_qwen3_gpu012.sh \
RUN_MODE=train \
RL_EXP_NAME=etth1_ot_qwen3_1_7b_rl_paper_same2 \
bash examples/time_series_forecast/run_qwen3-1.7B.sh \
  trainer.total_training_steps=120 \
  trainer.test_freq=20 \
  trainer.save_freq=40 \
  trainer.resume_mode=disable \
  trainer.log_val_generations=0
```

如果你只是要续跑同一个实验，把 `trainer.total_training_steps` 调大，并去掉或改掉 `trainer.resume_mode=disable`。
如果这一步也要抓逐 turn 的链路问题，把 `DEBUG_CHAIN=0` 改成 `1`。

## 训练后看什么

### 当前“反思修正”监督是怎么构造的

当前实现里，SFT builder 会先拿 `selected_prediction_model` 的 teacher forecast 作为 `Turn 3` 的 base forecast，然后离线构造两类 target：

- `validated_keep`：base forecast 通过检查，最终答案直接保留它
- `local_refine`：base forecast 存在局部问题，离线规则器生成一个小幅修正后的 target

当前会显式写进 parquet / metadata 的字段主要有：

- `turn3_target_type`
- `turn3_trigger_reason`
- `refine_ops_signature`
- `refine_gain_mse`
- `refine_gain_mae`

当前 SFT 里 agent 学到的“反思修正”主要是两层：

- 轨迹层：何时应该保留 selected forecast，何时应该做局部修正
- 文本层：`<think>` 中如何解释“为什么保持不变”或“为什么做小幅修正”

需要注意的是：当前 `local_refine` 仍然是确定性规则器合成的 target，不是论文原文定义的学习式数值修正器。这一块是当前实现和论文之间仍然存在的工程差异。

### 关键日志

- 聚合验证日志：`logs/debug/eval_step_aggregate.jsonl`
- 样本切片日志：`logs/debug/eval_step_samples.jsonl`
- chain debug：`$TS_CHAIN_DEBUG_FILE`
- 最终启动命令：`debug_logs/final_launch_cmd.txt`

### 当前最值得盯的字段

聚合级：

- `validation_reward_mean`
- `orig_mse_mean`
- `norm_mse_mean`
- `selected_model_distribution`
- `generation_stop_reason_distribution`
- `final_answer_accept_ratio`
- `tool_call_count_mean`
- `history_analysis_count_mean`
- `no_tool_call_ratio`
- `tool_call_sequence_distribution`

样本级：

- `selected_model`
- `generation_stop_reason`
- `selected_forecast_orig_mse`
- `final_vs_selected_mse`
- `refinement_delta_orig_mse`
- `final_answer_reject_reason`
- `tool_call_count`
- `history_analysis_count`
- `tool_call_sequence`

说明：

- `selected_forecast_*` 和 `final_vs_selected_*` 已经进入 reward extra info
- 如果 `no_tool_call_ratio` 很高，优先怀疑 agent 没有真正进入 Cast-R1 工具链
- 如果 `tool_call_sequence_distribution` 里大多数是 `none`，先不要分析 forecasting 精度，先修协议和数据
- 更细的字段优先去 `eval_step_samples.jsonl` 或 chain debug 看

## 常见问题

### 1. `MODEL_PATH` 写错

RL 的 `MODEL_PATH` 必须指向：

```bash
.../global_step_x/huggingface
```

不能只写到 `global_step_x`。

### 2. `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`

当前 vLLM memory pool 和这个选项不兼容。跑 RL 前先：

```bash
unset PYTORCH_CUDA_ALLOC_CONF
```

### 3. `Engine core initialization failed`

如果你把 `RL_ROLLOUT_GPU_MEMORY_UTILIZATION` 压得很低，却没设置 `RL_ROLLOUT_MAX_MODEL_LEN`，vLLM 会回退到模型原生上下文长度来预留 KV cache，Qwen3-1.7B 这里会按 `40960` 处理，常见结果就是启动阶段直接失败。

建议固定带上：

```bash
RL_ROLLOUT_MAX_MODEL_LEN=8192
```

### 4. OOM

优先按这个顺序降：

1. `RL_ROLLOUT_GPU_MEMORY_UTILIZATION`
2. `RL_ROLLOUT_N`
3. `RL_ROLLOUT_MAX_BATCHED_TOKENS`
4. `RL_MAX_RESPONSE_LENGTH`
5. `RL_ACTOR_MAX_TOKEN_LEN_PER_GPU`

### 5. 为什么 `/models` 里看不到 `arima`

`arima` 走本地推理路径，不通过远程模型服务加载，所以不会出现在 `/models` 列表。

### 6. 我已经有旧的 `teacher200` / SFT / RL checkpoint 了，还能接着用吗

不建议。

当前 reward、SFT 轨迹和 RL 元信息都已经变了，旧产物至少要按下面顺序重建：

1. RL jsonl
2. teacher eval / teacher200
3. SFT
4. RL

## 清理 Ray

```bash
ray stop
ray stop --force
rm -rf /tmp/ray
```

如果同一台机器上还有其他人在用 Ray，先确认不会影响别人。

## 当前最重要的目录

- 原始数据：`dataset/ETT-small/ETTh1.csv`
- 基础 RL 数据：`dataset/ett_rl_etth1_paper_same2/`
- curriculum RL 数据：`dataset/ett_rl_etth1_paper_aligned_ot_curriculum_same2/`
- 第一轮 teacher200：`dataset/ett_sft_etth1_runtime_teacher200_paper_same2/`
- 推荐正式 teacher200：`dataset/ett_sft_etth1_runtime_ot_teacher200_paper_same2/`
- SFT checkpoint：`artifacts/checkpoints/sft/`
- RL checkpoint：`artifacts/checkpoints/rl/`
- 调试日志：`logs/debug/`
