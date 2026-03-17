# Cast-R1-TS（ETTh1 单变量 `OT`）

这个分支基于作者原始仓库 [Xiaoyu-Tao/Cast-R1-TS](https://github.com/Xiaoyu-Tao/Cast-R1-TS) 整理，当前只保留 `ETTh1` 单变量 `OT` 预测所需改动，训练主线是：

`工具服务 -> 高质量 teacher200 -> SFT -> RL`

当前默认运行时已经切到 compact 协议：

- Turn 1：提特征
- Turn 2：调 `predict_time_series`
- Turn 3：输出 `<think>...</think><answer>...</answer>`
- runtime 不再把完整多轮对话当作 memory；只保留紧凑状态：`history_analysis` 和 `prediction_tool_output`
- 工具轮不再保留冗长 assistant prose，只保留实际 tool effects
- `predict_time_series` 的 tool 输出改为“单个起始时间戳 + 频率 + 纯数值 forecast values”
- 最终 `<answer>` 仍保留时间戳，保证 reward 和现有评估逻辑兼容
- reward 默认使用严格 ablation：只保留格式、长度和 MSE，关闭 change point / season-trend 附加项
- 正式 RL 默认使用严格最终答案格式：必须输出完整 `<answer>...</answer>`，不再接受截断 `<answer>` 或裸 forecast block

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

默认约定：

- `GPU3` 跑工具服务
- `GPU0,1,2` 跑 SFT / RL
- 基础模型默认是 `/data/linyujie/models/Qwen3-1.7B`
- 默认 logger 只用 `console`，不会主动连接 `swanlab`
- RL 默认按多轮 agent 的实际长度预算配置：`max_prompt_length=8192`、`actor_max_token_len_per_gpu=16384`
- RL 默认强制 `actor_rollout_ref.rollout.load_format=safetensors`，不要再用默认 `dummy`
- RL 默认 `test_freq=5`
- RL 默认 `data.dataloader_num_workers=0`，避免 `StatefulDataLoader` 在训练结束时出现 worker 被系统回收的退出噪声

关键脚本：

- 工具服务：[start_model_server.sh](/data/linyujie/Cast-R1-TS-main/Cast-R1-TS-main/recipe/time_series_forecast/start_model_server.sh)
- 高质量 SFT 构建：[build_etth1_high_quality_sft.py](/data/linyujie/Cast-R1-TS-main/Cast-R1-TS-main/recipe/time_series_forecast/build_etth1_high_quality_sft.py)
- SFT 训练：[run_qwen3-1.7B_sft.sh](/data/linyujie/Cast-R1-TS-main/Cast-R1-TS-main/examples/time_series_forecast/run_qwen3-1.7B_sft.sh)
- RL 训练：[run_qwen3-1.7B.sh](/data/linyujie/Cast-R1-TS-main/Cast-R1-TS-main/examples/time_series_forecast/run_qwen3-1.7B.sh)

## 推荐直接执行的流程

### 1. 启动工具服务

单独开一个终端执行：

```bash
PROFILE_PATH=examples/time_series_forecast/configs/etth1_ot_qwen3_gpu012.sh \
bash recipe/time_series_forecast/start_model_server.sh
```

说明：

- 这个命令会一直占用终端，这是正常的。
- 服务起来后可以检查：

```bash
curl http://127.0.0.1:8994/health
curl http://127.0.0.1:8994/models
```

### 2. 生成高质量 `teacher200_gpu`

如果你之前已经生成过 `dataset/ett_sft_etth1_runtime_ot_teacher200_gpu/`，在 compact 协议或 memory 策略更新后建议重新生成一次。

这一步现在还承担严格 ablation 的 teacher 选择。如果你之前生成的 `teacher200_gpu` 来自旧 reward，请重新生成，避免 teacher 选择目标和当前 RL reward 不一致。

如果你误删了上游 RL jsonl，可以先恢复：

```bash
python recipe/time_series_forecast/build_etth1_rl_dataset.py
```

恢复结果会写到：

- `dataset/ett_rl_etth1_paper_aligned_ot_20260315_151424/train.jsonl`
- `dataset/ett_rl_etth1_paper_aligned_ot_20260315_151424/val.jsonl`
- `dataset/ett_rl_etth1_paper_aligned_ot_20260315_151424/test.jsonl`

```bash
python recipe/time_series_forecast/build_etth1_high_quality_sft.py \
  --output-dir dataset/ett_sft_etth1_runtime_ot_teacher200_gpu \
  --model-service-url http://127.0.0.1:8994 \
  --train-target-samples 200 \
  --val-target-samples 64 \
  --test-target-samples 128 \
  --train-candidate-samples 600 \
  --val-candidate-samples 192 \
  --test-candidate-samples 256 \
  --models patchtst,chronos2,itransformer \
  --predictor-mode service \
  --max-concurrency 4
```

生成结果：

- `dataset/ett_sft_etth1_runtime_ot_teacher200_gpu/train.parquet`
- `dataset/ett_sft_etth1_runtime_ot_teacher200_gpu/val.parquet`
- `dataset/ett_sft_etth1_runtime_ot_teacher200_gpu/test.parquet`

### 3. 跑 SFT

推荐直接用高质量 `teacher200_gpu`：

```bash
MODEL_PATH=/data/linyujie/models/Qwen3-1.7B \
SAVE_DIR=$PWD/checkpoints/time_series_forecast_sft_teacher200_v2 \
PROFILE_PATH=examples/time_series_forecast/configs/etth1_ot_qwen3_gpu012.sh \
RUN_MODE=train \
SFT_TRAIN_FILES=$PWD/dataset/ett_sft_etth1_runtime_ot_teacher200_gpu/train.parquet \
SFT_VAL_FILES=$PWD/dataset/ett_sft_etth1_runtime_ot_teacher200_gpu/val.parquet \
bash examples/time_series_forecast/run_qwen3-1.7B_sft.sh \
  trainer.resume_mode=disable
```

如果你只是想先试通链路，不生成 teacher200，也可以直接用 profile 默认的 `paper200`：

```bash
MODEL_PATH=/data/linyujie/models/Qwen3-1.7B \
PROFILE_PATH=examples/time_series_forecast/configs/etth1_ot_qwen3_gpu012.sh \
RUN_MODE=train \
bash examples/time_series_forecast/run_qwen3-1.7B_sft.sh
```

SFT 输出目录：

```bash
checkpoints/time_series_forecast_sft/
```

后续 RL 需要用到的目录是：

```bash
checkpoints/time_series_forecast_sft/<exp>/global_step_x/huggingface
```

### 4. 跑 RL

把 `MODEL_PATH` 换成**本轮最新 SFT** 导出的 HuggingFace 目录。不要继续复用 compact 协议更新前的旧 checkpoint：

如果你要**续跑当前已经到 100 step 的实验**，把总步数调大，并显式打开验证频率：

```bash
MODEL_PATH=/data/linyujie/Cast-R1-TS-main/Cast-R1-TS-main/checkpoints/time_series_forecast_sft_compact_state_v1/global_step_11/huggingface \
PROFILE_PATH=examples/time_series_forecast/configs/etth1_ot_qwen3_gpu012.sh \
RUN_MODE=train \
bash examples/time_series_forecast/run_qwen3-1.7B.sh \
  trainer.total_training_steps=200 \
  trainer.test_freq=5
```

如果你要**新开一轮 RL 实验**，不要和当前 `etth1_ot_qwen3_1_7b_rl_gpu012` 混用，用新的实验名并禁用自动续跑：

```bash
DEBUG_CHAIN=1 \
TS_CHAIN_DEBUG_FILE=/tmp/ts_chain_debug.jsonl\
MODEL_PATH=/data/linyujie/Cast-R1-TS-main/Cast-R1-TS-main/checkpoints/time_series_forecast_sft_teacher200_v2/global_step_11/huggingface \
PROFILE_PATH=examples/time_series_forecast/configs/etth1_ot_qwen3_gpu012.sh \
RUN_MODE=train \
RL_EXP_NAME=etth1_ot_qwen3_1_7b_rl_gpu012_eval5_fresh \
bash examples/time_series_forecast/run_qwen3-1.7B.sh \
  trainer.total_training_steps=100 \
  trainer.test_freq=5 \
  trainer.resume_mode=disable
```

RL 输出目录：

```bash
checkpoints/TimeSeriesForecast/
```

说明：

- RL 默认 `resume_mode=auto`。同一个 `RL_EXP_NAME` 下再次运行，不会从头覆盖，而是会自动续跑最近的 checkpoint。
- 如果当前实验已经跑到 `100 step`，再写 `trainer.total_training_steps=100` 通常不会继续训练；续跑时要把总步数设得更大，比如 `200`。
- 如果你想完全重新开始，请换一个新的 `RL_EXP_NAME`，或者加 `trainer.resume_mode=disable`。
- 当前这条 ETTh1/OT 训练链路里，`vLLM + load_format=dummy` 会导致 rollout 输出异常长的乱码文本，表现为 `response_length=3072`、`reward=0`。
- 现在脚本已经默认改成 `safetensors`，不需要再手工额外覆盖；只有在你明确要测试别的加载方式时，才需要自己改 `RL_ROLLOUT_LOAD_FORMAT`。
- 如果训练已经跑到目标 step 并写出 `global_step_x`，但最后只剩下 dataloader / vLLM 的退出日志，先以 checkpoint 是否成功落盘为准；当前默认配置已经把这类退出期噪声尽量收到了最小。

## 清理 Ray

训练结束（或卡死）后，先停掉 Ray 集群，再清理临时目录，避免下次启动时继承旧 actor / worker 状态：

```bash
# 优雅停止（等待 actor 退出）
ray stop

# 如果 ray stop 挂住，强制杀掉所有 Ray 进程
ray stop --force

# 清理 Ray session 临时目录（日志、socket、plasma store 等）
rm -rf /tmp/ray
```

> 注意：如果同一台机器上还有其他人在用 Ray，`ray stop --force` 会影响所有人；确认没有冲突再执行。

---

## 可选 smoke

如果你只想先确认整条链能起：

### SFT smoke

```bash
MODEL_PATH=/data/linyujie/models/Qwen3-1.7B \
PROFILE_PATH=examples/time_series_forecast/configs/etth1_ot_qwen3_gpu012.sh \
RUN_MODE=smoke \
bash examples/time_series_forecast/run_qwen3-1.7B_sft.sh
```

### RL smoke

```bash
MODEL_PATH=/data/linyujie/models/Qwen3-1.7B \
PROFILE_PATH=examples/time_series_forecast/configs/etth1_ot_qwen3_gpu012.sh \
RUN_MODE=smoke \
bash examples/time_series_forecast/run_qwen3-1.7B.sh
```

## 当前最重要的几个目录

- 原始数据：`dataset/ETT-small/ETTh1.csv`
- RL 数据：`dataset/ett_rl_etth1_paper_aligned_ot_20260315_151424/`
- 默认 SFT 子集：`dataset/ett_sft_etth1_runtime_ot_paper200/`
- 推荐高质量 SFT：`dataset/ett_sft_etth1_runtime_ot_teacher200_gpu/`

## 一句话注意事项

- 如果你重建过 `teacher200_gpu` 或更新过 compact 协议，就重新跑一次 SFT，不要直接接旧的 RL checkpoint。
- 工具服务命令是前台常驻，不会自动返回 shell。
- 正式 RL 之前，优先保证 SFT 已经成功导出 `huggingface` 目录。
- 如果你确实要启用 `swanlab`，手动覆盖 `SFT_LOGGER='["console","swanlab"]'` 或 `RL_LOGGER='["console","swanlab"]'`。
