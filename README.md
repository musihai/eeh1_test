# Cast-R1-TS

当前仓库已经清理到一条新的正式复现实验主线，只保留原始 `ETTh1.csv` 和代码。

截至 `2026-03-23`，这条主线已经实际跑通到：

- 基础 RL 数据重建完成
- teacher-curated 数据重建完成
- curriculum RL 数据重建完成
- step-wise SFT 数据重建完成
- 正式 SFT 完成，checkpoint: `artifacts/checkpoints/sft/time_series_forecast_sft_paper_strict_formal_20260323/global_step_33/huggingface`
- RL `val_only` 完成，debug 目录: `logs/debug/diag_paper_strict_formal_20260323_valonly/`

## 正式实验命名

本轮正式实验统一使用：

- rerun tag: `paper_strict_formal_20260323`
- SFT project: `TimeSeriesForecast-SFT-Formal`
- SFT experiment: `qwen3-1.7b-etth1-ot-sft-paper-strict-formal-20260323`
- SFT save dir: `artifacts/checkpoints/sft/time_series_forecast_sft_paper_strict_formal_20260323/`
- RL project: `TimeSeriesForecast-Formal`
- RL experiment: `etth1_ot_qwen3_1_7b_rl_paper_strict_formal_20260323`

当前 profile 默认已经切到这套命名：

- [examples/time_series_forecast/configs/etth1_ot_qwen3_gpu012.sh](/data/linyujie/Cast-R1-TS-main/Cast-R1-TS-main/examples/time_series_forecast/configs/etth1_ot_qwen3_gpu012.sh)

注意：

- `RL_MODEL_PATH` 不再默认指向旧 SFT checkpoint
- 正式 RL 启动前必须显式 export 新的 SFT `huggingface` checkpoint 路径

## 环境

```bash
cd /data/linyujie/Cast-R1-TS-main/Cast-R1-TS-main
conda activate cast-r1-ts
export PYTHONPATH=$PWD:$PYTHONPATH
```

说明：

- 下文所有 `python` / `ray` / launcher 命令都默认在 `cast-r1-ts` 环境内执行
- `run_qwen3-1.7B_sft.sh` 和 `run_qwen3-1.7B.sh` 内部会直接调用 `python3` / `ray`，所以不要在系统环境里直接跑

## 当前仓库状态

这轮 clean rerun 开始前，已经删除：

- 旧生成数据集
- 旧 SFT / RL checkpoints
- 旧 debug 日志
- 非主线旧脚本 `build_etth1_sft_subset.py`
- 非主线旧脚本 `retrain_expert_models_train_split.py`

保留项只有：

- 原始数据：`dataset/ETT-small/ETTh1.csv`
- 主线代码与测试

## 正式流程

### Step 1. 重建基础 RL 数据

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

核对项：

- `train / val / test` 行数应为 `12060 / 1722 / 3065`
- `metadata.json` 的 `pipeline_stage` 应为 `base_rl`
- `raw_prompt` 必须包含 3-stage workflow 指令
- `ground_truth` 必须是带真实时间戳的 96 行序列

### Step 2. 重建 teacher-curated 数据

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
  --train-min-local-refine-ratio 0.0 \
  --models patchtst,chronos2,itransformer,arima \
  --predictor-mode local \
  --predictor-device cuda \
  --predictor-devices cuda:0,cuda:1,cuda:2,cuda:3 \
  --num-workers 4 \
  --local-batch-size 256 \
  --resume-teacher-eval \
  --max-turn3-annotation-error-count 0 \
  --max-turn3-annotation-error-ratio 0.0 \
  --max-concurrency 4
```

核对项：

- 产物目录：`dataset/ett_sft_etth1_runtime_teacher200_paper_same2/`
- `train_curated.jsonl / val_curated.jsonl / test_curated.jsonl` 数量应为 `200 / 64 / 128`
- `metadata.json` 的 `pipeline_stage` 应为 `teacher200_runtime_sft`
- `turn3_annotation_error_count` 和 `turn3_annotation_error_ratio` 必须为 `0`
- 当前主线是 `paper_strict`，所以 `local_refine` 不应再被当作默认目标分布

### Step 3. 用 teacher metadata 重建 curriculum RL 数据

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

核对项：

- `metadata.json` 的 `pipeline_stage` 应为 `curriculum_rl`
- `teacher_metadata_coverage_ratio` 应接近 `1.0`
- `train_stage1 / train_stage12 / train_stage123` 应正常生成

### Step 4. 重建 step-wise SFT 数据

```bash
python recipe/time_series_forecast/build_etth1_sft_dataset.py \
  --train-jsonl dataset/ett_sft_etth1_runtime_teacher200_paper_same2/train_curated.jsonl \
  --val-jsonl dataset/ett_sft_etth1_runtime_teacher200_paper_same2/val_curated.jsonl \
  --test-jsonl dataset/ett_sft_etth1_runtime_teacher200_paper_same2/test_curated.jsonl \
  --output-dir dataset/ett_sft_etth1_runtime_ot_teacher200_paper_same4_stepwise \
  --turn3-target-mode paper_strict \
  --train-min-local-refine-ratio 0.0
```

核对项：

- refinement 行必须记录 `turn3_target_mode = paper_strict`
- refinement 行必须记录 `turn3_target_type = validated_keep`
- `refine_ops_signature` 应为 `none`
- `turn3_protocol_valid_ratio` 应为 `1.0`

### Step 5. 跑正式 SFT

```bash
export PROFILE_PATH=examples/time_series_forecast/configs/etth1_ot_qwen3_gpu012.sh
export RUN_MODE=train
export CUDA_VISIBLE_DEVICES=0,1,2
export MODEL_PATH=/data/linyujie/models/Qwen3-1.7B
export SFT_DATASET_DIR=$PWD/dataset/ett_sft_etth1_runtime_ot_teacher200_paper_same4_stepwise

bash examples/time_series_forecast/run_qwen3-1.7B_sft.sh
```

默认命名：

- save dir: `artifacts/checkpoints/sft/time_series_forecast_sft_paper_strict_formal_20260323/`
- experiment: `qwen3-1.7b-etth1-ot-sft-paper-strict-formal-20260323`

核对项：

- `trainer.default_local_dir` 必须写到新的 formal SFT 目录
- `train.parquet / val.parquet` 必须来自 `same4_stepwise`
- 保存出的 checkpoint 必须包含 `hf_model`

本轮实际结果：

- checkpoint tracker: `artifacts/checkpoints/sft/time_series_forecast_sft_paper_strict_formal_20260323/latest_checkpointed_iteration.txt = 33`
- 最新 checkpoint: `artifacts/checkpoints/sft/time_series_forecast_sft_paper_strict_formal_20260323/global_step_33/`
- 最终 `val/loss = 0.0018108173971995711`

### Step 6. 先跑 RL val-only 检查

先找到最新 SFT checkpoint 的 `huggingface` 路径，然后显式设置：

```bash
export RL_MODEL_PATH=/abs/path/to/latest/huggingface
```

再跑：

```bash
ray stop --force

export RUN_MODE=smoke
export PROFILE_PATH=examples/time_series_forecast/configs/etth1_ot_qwen3_gpu012.sh
export CUDA_VISIBLE_DEVICES=0,1,2
export DEBUG_CHAIN=1
export TS_CHAIN_DEBUG_FILE=$PWD/logs/debug/diag_paper_strict_formal_20260323_valonly/ts_chain_debug_smoke.jsonl
export TS_MIN_EVAL_AGG_FILE=$PWD/logs/debug/diag_paper_strict_formal_20260323_valonly/eval_step_aggregate.jsonl
export TS_MIN_EVAL_SAMPLE_FILE=$PWD/logs/debug/diag_paper_strict_formal_20260323_valonly/eval_step_samples.jsonl

bash examples/time_series_forecast/run_qwen3-1.7B.sh \
  trainer.val_before_train=True \
  trainer.val_only=True
```

核对项：

- `final_answer_accept_ratio`
- `strict_length_match_ratio`
- `required_step_budget_mean`
- `prediction_step_index`
- `final_answer_step_index`
- `generation_finish_reason_distribution`

本轮实际结果：

- `final_answer_accept_ratio = 1.0`
- `strict_length_match_ratio = 1.0`
- `required_step_budget_mean = 4.0`
- `prediction_step_index = 2.0`
- `final_answer_step_index = 3.0`
- `generation_finish_reason_distribution = {"stop": 1}`
- `selected_forecast_exact_copy_ratio = 1.0`

### Step 7. 跑正式 curriculum RL

```bash
ray stop --force

export RUN_MODE=train
export PROFILE_PATH=examples/time_series_forecast/configs/etth1_ot_qwen3_gpu012.sh
export CUDA_VISIBLE_DEVICES=0,1,2
export RL_MODEL_PATH=/abs/path/to/latest/huggingface

bash examples/time_series_forecast/run_qwen3-1.7B_curriculum.sh
```

默认命名：

- RL project: `TimeSeriesForecast-Formal`
- RL experiment: `etth1_ot_qwen3_1_7b_rl_paper_strict_formal_20260323`

执行逻辑：

- `stage1`: `train_stage1.jsonl`
- `stage12`: `train_stage12.jsonl`
- `stage123`: `train_stage123.jsonl`
- 每个 phase 会把上一 phase 最新 `actor/huggingface` checkpoint 作为下一 phase 初始化

注意：

- 对 `curriculum_rl` 数据集，正式 `train` 模式不再允许直接用根目录 `train.jsonl`
- 如果你直接跑 `run_qwen3-1.7B.sh` 且没有显式设置 `RL_CURRICULUM_PHASE`，脚本会拒绝启动，避免再次绕开 curriculum

## 当前必须保留的主线代码

- `recipe/time_series_forecast/build_etth1_rl_dataset.py`
- `recipe/time_series_forecast/build_etth1_high_quality_sft.py`
- `recipe/time_series_forecast/build_etth1_sft_dataset.py`
- `recipe/time_series_forecast/time_series_forecast_agent_flow.py`
- `recipe/time_series_forecast/prompts.py`
- `recipe/time_series_forecast/reward.py`
- `recipe/time_series_forecast/model_server.py`
- `recipe/time_series_forecast/task_protocol.py`
- `recipe/time_series_forecast/dataset_identity.py`
- `examples/time_series_forecast/run_qwen3-1.7B_sft.sh`
- `examples/time_series_forecast/run_qwen3-1.7B.sh`
- `examples/time_series_forecast/configs/etth1_ot_qwen3_gpu012.sh`
