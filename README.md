# Cast-R1-TS（ETTh1 单变量 `OT`）

当前仓库的推荐主线是：

`工具服务 -> 基础 RL jsonl -> 第一轮 teacher eval -> curriculum RL jsonl -> 第二轮 teacher200 -> SFT -> RL`

## 先说结论

这轮修复之后，不需要强制重跑整个流程。

当前推荐最小重跑范围：

1. 重新跑 Step 5：第二轮 teacher200，生成新的 `same3` parquet
2. 重新跑 Step 6：SFT
3. 重新跑 Step 7 / Step 8：RL

一般不需要重跑 Step 2 到 Step 4，原因是：

- 这次最关键的修复集中在 `teacher200 -> SFT -> RL` 这一段
- `ETTh1.csv` 本身带真实时间戳，所以固定 synthetic timestamp anchor 不会改变这条主线的数据
- `SFT_DATASET_DIR` 路径约束是 launcher 行为修复，不影响已经生成好的 RL jsonl

只有在下面两种情况下，才建议从 Step 2 全部重跑：

- 你怀疑当前 `dataset/ett_rl_etth1_paper_aligned_ot_curriculum_same2/` 本身就不是用正确 teacher metadata 构建出来的
- 你想做一套完全干净、从头一致的新产物

## 环境

```bash
cd /data/linyujie/Cast-R1-TS-main/Cast-R1-TS-main
conda activate cast-r1-ts
export PYTHONPATH=$PWD:$PYTHONPATH
```

如果当前 shell 不能直接 `conda activate`，下面所有命令都可以改成 `conda run -n cast-r1-ts ...`。

统一 profile：

```bash
examples/time_series_forecast/configs/etth1_ot_qwen3_gpu012.sh
```

## 现在的关键约束

- `run_qwen3-1.7B_sft.sh` 不再猜数据集路径
- 跑 SFT 时必须显式设置 `SFT_DATASET_DIR`
- `SFT_DATASET_DIR` 对应目录里必须有 `train.parquet`、`val.parquet`、`metadata.json`
- `build_etth1_high_quality_sft.py` 现在默认 `Turn 3` 标注出错就直接失败，不再静默降级混入 parquet
- RL 启动前建议先 `ray stop --force`

## Step 1. 启动工具服务

如果后面要跑 RL，单独开一个终端执行：

```bash
PROFILE_PATH=examples/time_series_forecast/configs/etth1_ot_qwen3_gpu012.sh \
bash recipe/time_series_forecast/start_model_server.sh 3 8994
```

检查：

```bash
curl http://127.0.0.1:8994/health
curl http://127.0.0.1:8994/models
```

## Step 2 到 Step 4. 只有需要全量重建时才跑

### Step 2. 生成基础 RL jsonl

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

### Step 3. 第一轮 teacher eval

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python recipe/time_series_forecast/build_etth1_high_quality_sft.py \
  --train-jsonl dataset/ett_rl_etth1_paper_same2/train.jsonl \
  --val-jsonl dataset/ett_rl_etth1_paper_same2/val.jsonl \
  --test-jsonl dataset/ett_rl_etth1_paper_same2/test.jsonl \
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
  --max-turn3-annotation-error-count 0 \
  --max-turn3-annotation-error-ratio 0.0 \
  --max-concurrency 4
```

### Step 4. 用 teacher metadata 重建 curriculum RL jsonl

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

## 推荐现在直接从这里开始

### Step 5. 第二轮 teacher200

这一步现在必须重跑。

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python recipe/time_series_forecast/build_etth1_high_quality_sft.py \
  --train-jsonl dataset/ett_rl_etth1_paper_aligned_ot_curriculum_same2/train.jsonl \
  --val-jsonl dataset/ett_rl_etth1_paper_aligned_ot_curriculum_same2/val.jsonl \
  --test-jsonl dataset/ett_rl_etth1_paper_aligned_ot_curriculum_same2/test.jsonl \
  --output-dir dataset/ett_sft_etth1_runtime_ot_teacher200_paper_same3 \
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
  --train-min-local-refine-ratio 0.30 \
  --num-workers 4 \
  --local-batch-size 256 \
  --resume-teacher-eval \
  --max-turn3-annotation-error-count 0 \
  --max-turn3-annotation-error-ratio 0.0 \
  --max-concurrency 4
```

跑完至少检查这几项：

- `dataset/ett_sft_etth1_runtime_ot_teacher200_paper_same3/train.parquet`
- `dataset/ett_sft_etth1_runtime_ot_teacher200_paper_same3/val.parquet`
- `dataset/ett_sft_etth1_runtime_ot_teacher200_paper_same3/metadata.json`
- `metadata.json` 里的 `train_turn3_annotation_error_count=0`
- `metadata.json` 里的 `train_turn3_annotation_error_ratio=0.0`

如果这里失败并且生成了 `*_turn3_annotation_errors.jsonl`，先修这个，不要继续往下跑。

### Step 6. 跑 SFT

这一步现在必须重跑。

```bash
cd /data/linyujie/Cast-R1-TS-main/Cast-R1-TS-main
conda activate cast-r1-ts
export PYTHONPATH=$PWD:$PYTHONPATH

export MODEL_PATH=/data/linyujie/models/Qwen3-1.7B
export SAVE_DIR=$PWD/artifacts/checkpoints/sft/time_series_forecast_sft_teacher200_paper_same3
export EXPERIMENT_NAME=time_series_forecast_sft_teacher200_paper_same3
export PROFILE_PATH=examples/time_series_forecast/configs/etth1_ot_qwen3_gpu012.sh
export RUN_MODE=train
export SFT_DATASET_DIR=$PWD/dataset/ett_sft_etth1_runtime_ot_teacher200_paper_same3

PRINT_CMD_ONLY=1 bash examples/time_series_forecast/run_qwen3-1.7B_sft.sh \
  trainer.resume_mode=disable

unset PRINT_CMD_ONLY

bash examples/time_series_forecast/run_qwen3-1.7B_sft.sh \
  trainer.resume_mode=disable
```

先确认打印出来的是：

- `train.parquet` 和 `val.parquet` 都来自 `ett_sft_etth1_runtime_ot_teacher200_paper_same3`
- `trainer.default_local_dir` 指向 `time_series_forecast_sft_teacher200_paper_same3`

SFT 输出目录：

```bash
artifacts/checkpoints/sft/time_series_forecast_sft_teacher200_paper_same3/
```

RL 用的模型路径必须指向：

```bash
artifacts/checkpoints/sft/time_series_forecast_sft_teacher200_paper_same3/<global_step_x>/huggingface
```

### Step 7. RL smoke

这一步建议先跑。

```bash
cd /data/linyujie/Cast-R1-TS-main/Cast-R1-TS-main
conda activate cast-r1-ts
export PYTHONPATH=$PWD:$PYTHONPATH

unset PYTORCH_CUDA_ALLOC_CONF
ray stop --force

unset EXP_NAME TRAIN_FILES VAL_FILES MODEL_PATH RL_EXP_NAME

export DEBUG_CHAIN=1
export TS_CHAIN_DEBUG_FILE=$PWD/logs/debug/ts_chain_debug_smoke.jsonl
export RL_TRAIN_FILES=$PWD/dataset/ett_rl_etth1_paper_aligned_ot_curriculum_same2/train.jsonl
export RL_VAL_FILES=$PWD/dataset/ett_rl_etth1_paper_aligned_ot_curriculum_same2/val.jsonl
export RL_TEMPERATURE=0.3
export RL_ROLLOUT_GPU_MEMORY_UTILIZATION=0.22
export RL_ROLLOUT_MAX_MODEL_LEN=8192
export RL_ROLLOUT_MAX_BATCHED_TOKENS=4096
export RL_ROLLOUT_N=1
export RL_MAX_RESPONSE_LENGTH=3072
export RL_ACTOR_MAX_TOKEN_LEN_PER_GPU=6144
export MODEL_PATH=$PWD/artifacts/checkpoints/sft/time_series_forecast_sft_teacher200_paper_same3/global_step_6/huggingface
export PROFILE_PATH=examples/time_series_forecast/configs/etth1_ot_qwen3_gpu012.sh
export RUN_MODE=train
export RL_EXP_NAME=etth1_ot_qwen3_1_7b_rl_smoke_paper_same3

PRINT_CMD_ONLY=1 bash examples/time_series_forecast/run_qwen3-1.7B.sh \
  trainer.total_training_steps=20 \
  trainer.test_freq=5 \
  trainer.resume_mode=disable \
  trainer.log_val_generations=0

unset PRINT_CMD_ONLY

bash examples/time_series_forecast/run_qwen3-1.7B.sh \
  trainer.total_training_steps=20 \
  trainer.test_freq=5 \
  trainer.resume_mode=disable \
  trainer.log_val_generations=0
```

先确认：

- `data.train_files=...train.jsonl`
- `data.val_files=...val.jsonl`
- `actor_rollout_ref.model.path=.../huggingface`

不要假设一定是 `global_step_6`。先看：

```bash
cat artifacts/checkpoints/sft/time_series_forecast_sft_teacher200_paper_same3/latest_checkpointed_iteration.txt
```

然后把上面命令里的 step 改成真实值。

### Step 8. RL 正式训练

smoke 正常后再跑。

```bash
cd /data/linyujie/Cast-R1-TS-main/Cast-R1-TS-main
conda activate cast-r1-ts
export PYTHONPATH=$PWD:$PYTHONPATH

unset PYTORCH_CUDA_ALLOC_CONF
ray stop --force

unset EXP_NAME TRAIN_FILES VAL_FILES MODEL_PATH RL_EXP_NAME

export DEBUG_CHAIN=0
export TS_CHAIN_DEBUG_FILE=$PWD/logs/debug/ts_chain_debug_eval20_paper_same3.jsonl
export RL_TRAIN_FILES=$PWD/dataset/ett_rl_etth1_paper_aligned_ot_curriculum_same2/train_stage123.jsonl
export RL_VAL_FILES=$PWD/dataset/ett_rl_etth1_paper_aligned_ot_curriculum_same2/val.jsonl
export RL_TEMPERATURE=0.3
export RL_ROLLOUT_GPU_MEMORY_UTILIZATION=0.22
export RL_ROLLOUT_MAX_MODEL_LEN=8192
export RL_ROLLOUT_MAX_BATCHED_TOKENS=4096
export RL_ROLLOUT_N=1
export RL_MAX_RESPONSE_LENGTH=3072
export RL_ACTOR_MAX_TOKEN_LEN_PER_GPU=6144
export MODEL_PATH=$PWD/artifacts/checkpoints/sft/time_series_forecast_sft_teacher200_paper_same3/global_step_6/huggingface
export PROFILE_PATH=examples/time_series_forecast/configs/etth1_ot_qwen3_gpu012.sh
export RUN_MODE=train
export RL_EXP_NAME=etth1_ot_qwen3_1_7b_rl_paper_same3

PRINT_CMD_ONLY=1 bash examples/time_series_forecast/run_qwen3-1.7B.sh \
  trainer.total_training_steps=120 \
  trainer.test_freq=20 \
  trainer.save_freq=40 \
  trainer.resume_mode=disable \
  trainer.log_val_generations=0

unset PRINT_CMD_ONLY

bash examples/time_series_forecast/run_qwen3-1.7B.sh \
  trainer.total_training_steps=120 \
  trainer.test_freq=20 \
  trainer.save_freq=40 \
  trainer.resume_mode=disable \
  trainer.log_val_generations=0
```

## 当前最值得看什么

关键日志：

- `logs/debug/eval_step_aggregate.jsonl`
- `logs/debug/eval_step_samples.jsonl`
- `$TS_CHAIN_DEBUG_FILE`
- `debug_logs/final_launch_cmd.txt`

RL smoke / 正式训练时优先看：

- `workflow_status`
- `final_answer_parse_mode`
- `final_answer_reject_reason`
- `tool_call_count`
- `tool_call_sequence`

如果 `workflow_status` 长期是 `not_attempted`，或者 `refinement` 轮还在继续调工具，先不要分析预测精度，先修协议和轨迹。

## 当前最重要的目录

- 原始数据：`dataset/ETT-small/ETTh1.csv`
- 基础 RL 数据：`dataset/ett_rl_etth1_paper_same2/`
- curriculum RL 数据：`dataset/ett_rl_etth1_paper_aligned_ot_curriculum_same2/`
- 第一轮 teacher200：`dataset/ett_sft_etth1_runtime_teacher200_paper_same2/`
- 推荐正式 teacher200：`dataset/ett_sft_etth1_runtime_ot_teacher200_paper_same3/`
- SFT checkpoint：`artifacts/checkpoints/sft/`
- RL checkpoint：`artifacts/checkpoints/rl/`
- 调试日志：`logs/debug/`
