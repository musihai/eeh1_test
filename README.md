# Cast-R1-TS

当前仓库只保留一条正式链路：`ETTh1 multivariate -> teacher-curated SFT -> step-wise SFT -> curriculum RL`。  
不再推荐也不再记录旧的 `mv1/tsfix/heuristicroute/toolschema` 实验链路。

## 1. 环境

```bash
cd /data/linyujie/Cast-R1-TS-main/Cast-R1-TS-main
source /data/linyujie/miniconda3/etc/profile.d/conda.sh
conda activate cast-r1-ts
export PYTHONPATH=$PWD:$PYTHONPATH
```

默认 GPU 约定：

- `GPU 0,1,2`：SFT / RL
- `GPU 3`：预测服务

## 2. 当前正式目录

- RL base dataset:
  `dataset/ett_rl_etth1_paper_same2`
- teacher-curated SFT dataset:
  `dataset/ett_sft_etth1_runtime_teacher200_paper_same2`
- step-wise SFT dataset:
  `dataset/ett_sft_etth1_runtime_ot_teacher200_paper_same4_stepwise`
- curriculum RL dataset:
  `dataset/ett_rl_etth1_paper_aligned_ot_curriculum_same2`
- profile:
  `examples/time_series_forecast/configs/etth1_ot_qwen3_gpu012.sh`

说明：

- 当前仓库不再内置一个保证存在的 RL warm start checkpoint。
- 运行 RL 前，必须显式提供 `RL_MODEL_PATH`，指向一个真实存在的 HuggingFace 格式目录，例如：
  `artifacts/checkpoints/sft/<your_sft_run>/global_step_xx/huggingface`
- 正式 step-wise SFT 只接受这一组 metadata 约束：
  `sft_stage_mode=full`、`turn3_target_mode=paper_strict`、`routing_label_source=reference_teacher`
- 当前 formal RL 默认值已经改为：
  `RL_KL_LOSS_COEF=0.01`
  `RL_NORM_ADV_BY_STD_IN_GRPO=False`
- `dataset/ett_sft_etth1_runtime_ot_teacher200_paper_same4_refinement_decision_*`
  和
  `artifacts/checkpoints/sft/qwen3-1.7b-etth1-refinement-decision-*`
  都是实验分支，不作为正式 RL warm start 输入

### 2.1 清理基线

- 当前正式链路建议只保留：
  `dataset/ETT-small`
  `dataset/ett_rl_etth1_paper_same2`
  `dataset/ett_sft_etth1_runtime_teacher200_paper_same2`
  `dataset/ett_sft_etth1_runtime_ot_teacher200_paper_same4_stepwise`
  `dataset/ett_rl_etth1_paper_aligned_ot_curriculum_same2`
- 当前仓库内建议只保留一个 formal SFT warm start 目录：
  `artifacts/checkpoints/sft/qwen3-1.7b-etth1-sft-paper-20260330_224014`
- 可以整体删除的旧实验产物包括：
  `dataset/ett_sft_etth1_runtime_ot_teacher200_paper_same4_*`
  里除 `..._stepwise` 之外的所有目录
- 可以整体删除的旧权重包括：
  `artifacts/checkpoints/sft/`
  里除最新 formal SFT 外的所有目录
- 可以整体删除的旧 RL 输出包括：
  `artifacts/checkpoints/rl/` 下全部目录
- 可以整体删除的运行报告与 probe 输出包括：
  `artifacts/reports/` 下除少量手写 `.md` 说明外的所有目录和日志文件

## 3. 一次性准备

预测服务依赖的本地 expert 权重放在：

- `recipe/time_series_forecast/models/patchtst`
- `recipe/time_series_forecast/models/itransformer`
- `recipe/time_series_forecast/models/chronos-2`

基座模型默认是：

- `/data/linyujie/models/Qwen3-1.7B`

## 4. 重建完整数据链路

### 4.1 构建 RL base dataset

```bash
python -m recipe.time_series_forecast.build_etth1_rl_dataset \
  --output-dir dataset/ett_rl_etth1_paper_same2
```

产物：

- `dataset/ett_rl_etth1_paper_same2/train.jsonl`
- `dataset/ett_rl_etth1_paper_same2/val.jsonl`
- `dataset/ett_rl_etth1_paper_same2/test.jsonl`
- `dataset/ett_rl_etth1_paper_same2/metadata.json`

### 4.2 构建 teacher-curated SFT dataset

推荐直接用本地 predictor，单卡放在 `GPU 3`：

```bash
CUDA_VISIBLE_DEVICES=3 \
python -m recipe.time_series_forecast.build_etth1_high_quality_sft \
  --train-jsonl dataset/ett_rl_etth1_paper_same2/train.jsonl \
  --val-jsonl dataset/ett_rl_etth1_paper_same2/val.jsonl \
  --test-jsonl dataset/ett_rl_etth1_paper_same2/test.jsonl \
  --output-dir dataset/ett_sft_etth1_runtime_teacher200_paper_same2 \
  --predictor-mode local \
  --predictor-device cuda:0 \
  --predictor-devices cuda:0
```

产物：

- `train_curated.jsonl / val_curated.jsonl / test_curated.jsonl`
- `train_teacher_eval.jsonl / val_teacher_eval.jsonl / test_teacher_eval.jsonl`
- `metadata.json`

### 4.3 构建 step-wise SFT dataset

```bash
python -m recipe.time_series_forecast.build_etth1_sft_dataset \
  --train-jsonl dataset/ett_sft_etth1_runtime_teacher200_paper_same2/train_curated.jsonl \
  --val-jsonl dataset/ett_sft_etth1_runtime_teacher200_paper_same2/val_curated.jsonl \
  --test-jsonl dataset/ett_sft_etth1_runtime_teacher200_paper_same2/test_curated.jsonl \
  --routing-label-source reference_teacher \
  --train-min-local-refine-ratio 0.10 \
  --train-turn3-rebalance-mode oversample_local_refine \
  --output-dir dataset/ett_sft_etth1_runtime_ot_teacher200_paper_same4_stepwise
```

产物：

- `train.parquet / val.parquet / test.parquet`
- `metadata.json`

正式链路要求：

- `--routing-label-source` 必须是 `reference_teacher`
- 如果 4.2 保持 `teacher200` 紧凑高质量集合，4.3 推荐同时使用：
  `--train-min-local-refine-ratio 0.10`
  `--train-turn3-rebalance-mode oversample_local_refine`
- 不要继续使用 `downsample_keep`；它会把 `200` 个 train source window 下采样到几十个
- 产物 `metadata.json` 必须满足：
  `sft_stage_mode=full`、`turn3_target_mode=paper_strict`、`routing_label_source=reference_teacher`

建议构建后立即检查：

```bash
python - <<'PY'
import json
from pathlib import Path

meta = json.loads(
    Path("dataset/ett_sft_etth1_runtime_ot_teacher200_paper_same4_stepwise/metadata.json").read_text()
)
for key in (
    "sft_stage_mode",
    "turn3_target_mode",
    "routing_label_source",
    "train_turn3_rebalance_mode",
    "train_min_local_refine_ratio",
    "train_source_samples_before_balance",
    "train_source_samples",
):
    print(f"{key}={meta.get(key)}")
PY
```

预期输出：

- `sft_stage_mode=full`
- `turn3_target_mode=paper_strict`
- `routing_label_source=reference_teacher`
- `train_turn3_rebalance_mode=oversample_local_refine`
- `train_min_local_refine_ratio=0.1`
- `train_source_samples_before_balance=200`
- `train_source_samples=200`

如果这里仍然出现 `train_turn3_rebalance_mode=downsample_keep`
或者 `train_source_samples` 明显小于 `200`，说明 4.3 还没有按正式推荐配置重建。

如果这个目录是旧版本构建出来的，重新执行 4.3 后再检查一次 `metadata.json`；
只要其中还出现 `routing_label_source=heuristic`，就不要继续拿它训 formal SFT / RL。

### 4.4 构建 curriculum RL dataset

```bash
python -m recipe.time_series_forecast.build_etth1_rl_dataset \
  --output-dir dataset/ett_rl_etth1_paper_aligned_ot_curriculum_same2 \
  --train-teacher-metadata-jsonl dataset/ett_sft_etth1_runtime_teacher200_paper_same2/train_teacher_eval.jsonl \
  --val-teacher-metadata-jsonl dataset/ett_sft_etth1_runtime_teacher200_paper_same2/val_teacher_eval.jsonl \
  --test-teacher-metadata-jsonl dataset/ett_sft_etth1_runtime_teacher200_paper_same2/test_teacher_eval.jsonl
```

产物：

- `train_stage1.jsonl`
- `train_stage12.jsonl`
- `train_stage123.jsonl`
- `val.jsonl / test.jsonl`
- `metadata.json`

## 5. 启动预测服务

```bash
PROFILE_PATH=examples/time_series_forecast/configs/etth1_ot_qwen3_gpu012.sh \
bash recipe/time_series_forecast/start_model_server.sh 3 8994
```

健康检查：

```bash
curl -s http://127.0.0.1:8994/health
curl -s http://127.0.0.1:8994/models
```

要求：

- `patchtst` 已加载
- `itransformer` 已加载
- `chronos2` 已加载

## 6. 训练命令

这一节只保留一套可直接复制的命令。
默认值统一放在：

- `examples/time_series_forecast/configs/etth1_ot_qwen3_gpu012.sh`

这个文件现在按三块组织：

- `Paths`：模型、数据、输出路径
- `Resources`：GPU、服务端口、节点数
- `Training hyperparameters`：batch、lr、temperature、rollout 等

当前 formal RL 默认值已经按“尽量贴论文、但兼容当前 3 张训练卡 + 1 张服务卡”的原则收好：

- 论文目标：`global batch size=64`、`G=8`、`gradient accumulation=4`
- 当前硬件下，verl 需要满足 `data.train_batch_size * rollout.n` 能被训练 GPU 数整除
- 因此 `64` 在 3 张训练卡下不能精确实现
- 当前默认 formal RL 取：
  `RL_TRAIN_BATCH_SIZE=9`、`RL_ROLLOUT_N=8`
  也就是 `72 trajectories / iter`
- 同时取：
  `RL_PPO_MINI_BATCH_SIZE=3`、`RL_PPO_MICRO_BATCH_SIZE=2`
  这样 actor 侧等效 `gradient accumulation=4`
- 另外固定取：
  `RL_KL_LOSS_COEF=0.01`
  `RL_NORM_ADV_BY_STD_IN_GRPO=False`

如果你想临时改成别的值，优先只改 profile；如果只是一次实验，直接用命令前缀覆盖。

单次实验如果想临时覆盖默认值，直接在命令前加环境变量即可，例如：

```bash
SFT_TRAIN_BATCH_SIZE=8 RL_LR=5e-7 bash ...
```

命令里显式传入的环境变量都会覆盖 profile 默认值。

### 6.1 可选：启用 SwanLab

如果你想在训练时记录 SFT loss 曲线和 RL 关键指标，先安装并设置 SwanLab：

```bash
pip install swanlab

export SWANLAB_API_KEY=你的_swanlab_api_key
export SWANLAB_MODE=cloud
```

说明：

- logger 后端名字是 `swanlab`
- `SWANLAB_LOG_DIR` 是本地日志目录；如果不设置，默认写到当前目录下的 `swanlog`
- 如果只想本地记录，可以把 `SWANLAB_MODE=local`

### 6.2 先定义公共变量

```bash
PROJECT_DIR=/data/linyujie/Cast-R1-TS-main/Cast-R1-TS-main
PROFILE_PATH=$PROJECT_DIR/examples/time_series_forecast/configs/etth1_ot_qwen3_gpu012.sh
BASE_MODEL_PATH=/data/linyujie/models/Qwen3-1.7B

SFT_DATASET_DIR=$PROJECT_DIR/dataset/ett_sft_etth1_runtime_ot_teacher200_paper_same4_stepwise
RL_CURRICULUM_DATASET_DIR=$PROJECT_DIR/dataset/ett_rl_etth1_paper_aligned_ot_curriculum_same2

RUN_TAG=$(date +%Y%m%d_%H%M%S)
SFT_RUN_NAME=qwen3-1.7b-etth1-sft-paper-$RUN_TAG
RL_RUN_NAME=etth1_ot_qwen3_1_7b_rl_paper_$RUN_TAG

SFT_SMOKE_SAVE_DIR=$PROJECT_DIR/artifacts/checkpoints/sft/${SFT_RUN_NAME}_smoke
SFT_FORMAL_SAVE_DIR=$PROJECT_DIR/artifacts/checkpoints/sft/$SFT_RUN_NAME
RL_TRAINER_LOCAL_DIR=$PROJECT_DIR/artifacts/checkpoints/rl/$RL_RUN_NAME
SWANLAB_LOG_DIR=$PROJECT_DIR/swanlog/$RUN_TAG
RL_DEBUG_DIR=$PROJECT_DIR/logs/debug/$RUN_TAG
TS_CHAIN_DEBUG_FILE=$RL_DEBUG_DIR/ts_chain_debug.jsonl
TS_MIN_DEBUG_DIR=$RL_DEBUG_DIR
```

说明：

- `BASE_MODEL_PATH`：Qwen3-1.7B 基座模型
- `SFT_DATASET_DIR`：正式 step-wise SFT parquet 目录
  即 4.3 重新构建后的
  `dataset/ett_sft_etth1_runtime_ot_teacher200_paper_same4_stepwise`
- `RL_CURRICULUM_DATASET_DIR`：curriculum RL jsonl 目录
- `SFT_SMOKE_SAVE_DIR`：SFT smoke 输出目录
- `SFT_FORMAL_SAVE_DIR`：正式 SFT 输出目录
- `RL_TRAINER_LOCAL_DIR`：RL 输出目录
- `SWANLAB_LOG_DIR`：当前实验的 SwanLab 本地日志目录
- `RL_DEBUG_DIR`：当前 RL run 的独立调试目录
- `TS_CHAIN_DEBUG_FILE` / `TS_MIN_DEBUG_DIR`：当前 RL run 的链路调试输出

不要把 `SFT_DATASET_DIR` 指到任何
`dataset/ett_sft_etth1_runtime_ot_teacher200_paper_same4_refinement_decision_*`
目录；这些是实验型数据集，不是正式链路输入。

### 6.3 训练前只打印命令

SFT：

```bash
PROFILE_PATH=$PROFILE_PATH \
PRINT_CMD_ONLY=1 \
SFT_MODEL_PATH=$BASE_MODEL_PATH \
SFT_DATASET_DIR=$SFT_DATASET_DIR \
SFT_SAVE_DIR=$SFT_FORMAL_SAVE_DIR \
SFT_EXPERIMENT_NAME=$SFT_RUN_NAME \
SFT_LOGGER='["console","swanlab"]' \
SWANLAB_LOG_DIR=$SWANLAB_LOG_DIR \
bash examples/time_series_forecast/run_qwen3-1.7B_sft.sh
```

RL：

```bash
PROFILE_PATH=$PROFILE_PATH \
PRINT_CMD_ONLY=1 \
RUN_MODE=val_only \
RL_CURRICULUM_PHASE=stage123 \
RL_CURRICULUM_DATASET_DIR=$RL_CURRICULUM_DATASET_DIR \
RL_MODEL_PATH=$BASE_MODEL_PATH \
RL_TRAINER_LOCAL_DIR=$RL_TRAINER_LOCAL_DIR \
RL_EXP_NAME=${RL_RUN_NAME}_valonly \
RL_LOGGER='["console","swanlab"]' \
SWANLAB_LOG_DIR=$SWANLAB_LOG_DIR \
bash examples/time_series_forecast/run_qwen3-1.7B.sh
```

说明：

- SFT 这一步会校验 `metadata.json`、`train.parquet`、`val.parquet` 和 Turn-3 protocol
- RL 这一步会校验 `metadata.json`、`train_stage123.jsonl`、`val.jsonl` 和 `RL_MODEL_PATH`
- 这里只是 dry-run，所以 RL 临时使用 `RL_MODEL_PATH=$BASE_MODEL_PATH`

### 6.4 跑 SFT

SFT smoke：

```bash
PROFILE_PATH=$PROFILE_PATH \
RUN_MODE=smoke \
SFT_MODEL_PATH=$BASE_MODEL_PATH \
SFT_DATASET_DIR=$SFT_DATASET_DIR \
SFT_SAVE_DIR=$SFT_SMOKE_SAVE_DIR \
SFT_EXPERIMENT_NAME=${SFT_RUN_NAME}_smoke \
SFT_LOGGER='["console","swanlab"]' \
SWANLAB_LOG_DIR=$SWANLAB_LOG_DIR \
bash examples/time_series_forecast/run_qwen3-1.7B_sft.sh
```

正式 SFT：

```bash
PROFILE_PATH=$PROFILE_PATH \
SFT_MODEL_PATH=$BASE_MODEL_PATH \
SFT_DATASET_DIR=$SFT_DATASET_DIR \
SFT_SAVE_DIR=$SFT_FORMAL_SAVE_DIR \
SFT_EXPERIMENT_NAME=$SFT_RUN_NAME \
SFT_LOGGER='["console","swanlab"]' \
SWANLAB_LOG_DIR=$SWANLAB_LOG_DIR \
bash examples/time_series_forecast/run_qwen3-1.7B_sft.sh
```

说明：

- 不要让 smoke 和正式 SFT 共用同一个 `SFT_SAVE_DIR`
- `verl` 的 SFT trainer 默认 `resume_mode=auto`，如果目录里已经有 `global_step_*`，正式训练会自动从 smoke checkpoint 继续
- 如果你必须复用一个旧目录，请显式追加：
  `trainer.resume_mode=disable`

SFT 在 SwanLab 里建议重点看：

- `train/loss`
- `val/loss`
- `train/grad_norm`
- `train/lr`

### 6.5 从 Formal SFT 输出里取 RL warm start

SFT 完成后，执行：

```bash
RL_MODEL_PATH=$(find "$SFT_FORMAL_SAVE_DIR" -maxdepth 2 -type d -path "*/global_step_*/huggingface" | sort -V | tail -1)
echo "$RL_MODEL_PATH"
```

要求：

- `RL_MODEL_PATH` 不能为空
- `RL_MODEL_PATH` 必须指向真实存在的 `global_step_*/huggingface`
- `RL_MODEL_PATH` 必须来自 6.4 这一步正式 SFT 的输出
- 不要再使用任何实验型 checkpoint 作为 warm start，例如：
  `qwen3-1.7b-etth1-refinement-decision-*`
  `*keepok*`
  `*fixgating*`
  `*supportcard*`

如果你想再确认一次 warm start 来源，可以检查：

```bash
python - <<'PY'
import os
from pathlib import Path

path = Path(os.environ["RL_MODEL_PATH"])
print(path)
print("exists=", path.exists())
print("is_hf_dir=", (path / "config.json").exists() and (path / "tokenizer_config.json").exists())
PY
```

### 6.6 跑 RL

如果要重新起一轮 RL，不要复用旧的 `RUN_TAG / RL_RUN_NAME / RL_TRAINER_LOCAL_DIR`。
先刷新一遍：

```bash
RUN_TAG=$(date +%Y%m%d_%H%M%S)
RL_RUN_NAME=etth1_ot_qwen3_1_7b_rl_paper_$RUN_TAG
RL_TRAINER_LOCAL_DIR=$PROJECT_DIR/artifacts/checkpoints/rl/$RL_RUN_NAME
SWANLAB_LOG_DIR=$PROJECT_DIR/swanlog/$RUN_TAG
RL_DEBUG_DIR=$PROJECT_DIR/logs/debug/$RUN_TAG
TS_CHAIN_DEBUG_FILE=$RL_DEBUG_DIR/ts_chain_debug.jsonl
TS_MIN_DEBUG_DIR=$RL_DEBUG_DIR
```

RL val-only：

```bash
PROFILE_PATH=$PROFILE_PATH \
DEBUG_CHAIN=1 \
RUN_MODE=val_only \
RL_CURRICULUM_PHASE=stage123 \
RL_CURRICULUM_DATASET_DIR=$RL_CURRICULUM_DATASET_DIR \
RL_MODEL_PATH=$RL_MODEL_PATH \
RL_TRAINER_LOCAL_DIR=$RL_TRAINER_LOCAL_DIR \
RL_EXP_NAME=${RL_RUN_NAME}_valonly \
RL_LOGGER='["console","swanlab"]' \
SWANLAB_LOG_DIR=$SWANLAB_LOG_DIR \
TS_CHAIN_DEBUG_FILE=$TS_CHAIN_DEBUG_FILE \
TS_MIN_DEBUG_DIR=$TS_MIN_DEBUG_DIR \
bash examples/time_series_forecast/run_qwen3-1.7B.sh \
  trainer.resume_mode=disable
```

RL smoke：

```bash
PROFILE_PATH=$PROFILE_PATH \
DEBUG_CHAIN=1 \
RUN_MODE=smoke \
RL_CURRICULUM_PHASE=stage1 \
RL_CURRICULUM_DATASET_DIR=$RL_CURRICULUM_DATASET_DIR \
RL_MODEL_PATH=$RL_MODEL_PATH \
RL_TRAINER_LOCAL_DIR=$RL_TRAINER_LOCAL_DIR \
RL_EXP_NAME=${RL_RUN_NAME}_smoke \
RL_LOGGER='["console","swanlab"]' \
SWANLAB_LOG_DIR=$SWANLAB_LOG_DIR \
TS_CHAIN_DEBUG_FILE=$TS_CHAIN_DEBUG_FILE \
TS_MIN_DEBUG_DIR=$TS_MIN_DEBUG_DIR \
bash examples/time_series_forecast/run_qwen3-1.7B.sh \
  trainer.resume_mode=disable
```

正式 curriculum RL：

```bash
conda run -n cast-r1-ts ray stop --force

PROFILE_PATH=$PROFILE_PATH \
DEBUG_CHAIN=1 \
RL_CURRICULUM_DATASET_DIR=$RL_CURRICULUM_DATASET_DIR \
RL_MODEL_PATH=$RL_MODEL_PATH \
RL_TRAINER_LOCAL_DIR=$RL_TRAINER_LOCAL_DIR \
RL_EXP_NAME=$RL_RUN_NAME \
RL_LOGGER='["console","swanlab"]' \
SWANLAB_LOG_DIR=$SWANLAB_LOG_DIR \
TS_CHAIN_DEBUG_FILE=$TS_CHAIN_DEBUG_FILE \
TS_MIN_DEBUG_DIR=$TS_MIN_DEBUG_DIR \
bash examples/time_series_forecast/run_qwen3-1.7B_curriculum.sh \
  trainer.resume_mode=disable
```

说明：

- `run_qwen3-1.7B_curriculum.sh` 会自动依次跑 `stage1 -> stage12 -> stage123`
- 每个命令里显式传入的环境变量都会覆盖 profile 默认值
- formal RL 当前默认每 `8` 个 step 做一次验证
- RL 真正开始前，不要再使用 `RL_MODEL_PATH=$BASE_MODEL_PATH`
- 重新起 RL 时一定换新的 `RUN_TAG`，否则 checkpoint、SwanLab 和 debug 文件很容易和旧 run 混在一起
- `trainer.resume_mode=disable` 是为了避免误接上旧 smoke 或旧 formal 目录

RL 在 SwanLab 里建议分两组看。

训练过程指标：

- `reward_mean`
- `critic/rewards/mean`
- `critic/advantages/mean`
- `response/aborted_ratio`
- `response_length_non_aborted/clip_ratio`

验证效果指标：

- `val-agg/validation_reward_mean`
- `val-agg/orig_mse_mean`
- `val-agg/norm_mse_mean`
- `val-agg/final_answer_accept_ratio`
- `val-agg/strict_length_match_ratio`
- `val-agg/selected_forecast_orig_mse_mean`
- `val-agg/final_vs_selected_mse_mean`
- `val-agg/refinement_changed_ratio`
- `val-agg/refinement_improved_ratio`
- `val-agg/refinement_degraded_ratio`
- `val-agg/patchtst_share`
- `val-agg/itransformer_share`
- `val-agg/arima_share`
- `val-agg/chronos2_share`

说明：

- 训练 `reward_mean` 主要看优化过程是否稳定
- 选 checkpoint 优先看验证指标，尤其是 `val-agg/validation_reward_mean`、`val-agg/orig_mse_mean`、`val-agg/final_answer_accept_ratio`
- 当前 RL 还会把验证样例表记到 `val/generations`

## 9. 调试输出

如果要记录完整链路：

```bash
DEBUG_CHAIN=1 \
RUN_MODE=val_only \
RL_CURRICULUM_PHASE=stage123 \
RUN_TAG=paper_valonly_$(date +%Y%m%d_%H%M%S) \
TS_CHAIN_DEBUG_FILE=$PWD/logs/debug/$RUN_TAG/ts_chain_debug.jsonl \
bash examples/time_series_forecast/run_qwen3-1.7B.sh
```

关键文件：

- `logs/debug/<RUN_TAG>/ts_chain_debug.jsonl`
- `logs/debug/<RUN_TAG>/eval_step_aggregate.jsonl`
- `logs/debug/<RUN_TAG>/eval_step_samples.jsonl`

## 10. 当前代码主线

- `recipe/time_series_forecast/build_etth1_rl_dataset.py`
- `recipe/time_series_forecast/build_etth1_high_quality_sft.py`
- `recipe/time_series_forecast/build_etth1_sft_dataset.py`
- `recipe/time_series_forecast/model_server.py`
- `recipe/time_series_forecast/utils.py`
- `recipe/time_series_forecast/prompts.py`
- `recipe/time_series_forecast/reward.py`
- `recipe/time_series_forecast/time_series_forecast_agent_flow.py`
- `examples/time_series_forecast/configs/etth1_ot_qwen3_gpu012.sh`
- `examples/time_series_forecast/run_qwen3-1.7B.sh`
- `examples/time_series_forecast/run_qwen3-1.7B_sft.sh`
- `examples/time_series_forecast/run_qwen3-1.7B_curriculum.sh`
