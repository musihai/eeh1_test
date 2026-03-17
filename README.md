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
- RL 默认长度预算（按当前 profile）：`max_prompt_length=4096`、`actor_max_token_len_per_gpu=8192`
- RL 默认强制 `actor_rollout_ref.rollout.load_format=safetensors`，不要再用默认 `dummy`
- RL 默认 `test_freq=5`
- RL 默认 `data.dataloader_num_workers=0`，避免 `StatefulDataLoader` 在训练结束时出现 worker 被系统回收的退出噪声
- RL 开启 `DEBUG_CHAIN=1` 时，默认调试文件路径是 `logs/debug/ts_chain_debug.jsonl`

关键脚本：

- 工具服务：[start_model_server.sh](recipe/time_series_forecast/start_model_server.sh)
- 高质量 SFT 构建：[build_etth1_high_quality_sft.py](recipe/time_series_forecast/build_etth1_high_quality_sft.py)
- SFT 训练：[run_qwen3-1.7B_sft.sh](examples/time_series_forecast/run_qwen3-1.7B_sft.sh)
- RL 训练：[run_qwen3-1.7B.sh](examples/time_series_forecast/run_qwen3-1.7B.sh)

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
SAVE_DIR=$PWD/artifacts/checkpoints/sft/time_series_forecast_sft_teacher200_new \
PROFILE_PATH=examples/time_series_forecast/configs/etth1_ot_qwen3_gpu012.sh \
RUN_MODE=train \
SFT_TRAIN_FILES=$PWD/dataset/ett_sft_etth1_runtime_ot_teacher200_gpu/train.parquet \
SFT_VAL_FILES=$PWD/dataset/ett_sft_etth1_runtime_ot_teacher200_gpu/val.parquet \
bash examples/time_series_forecast/run_qwen3-1.7B_sft.sh \
  trainer.resume_mode=disable
```


SFT 输出目录：

```bash
artifacts/checkpoints/sft/
```

后续 RL 需要用到的目录是：

```bash
artifacts/checkpoints/sft/<exp>/global_step_x/huggingface
```

### 4. 跑 RL

关键默认值（如 `MAX_PROMPT_LENGTH`、`MAX_RESPONSE_LENGTH`、`TEMPERATURE`、`VAL_TEMPERATURE`）只在 profile 文件里维护：

```bash
examples/time_series_forecast/configs/etth1_ot_qwen3_gpu012.sh
```

`run_qwen3-1.7B.sh` 不再内置这组默认值；优先级固定为：**命令行覆盖 > 环境变量 > profile 默认值**。

把 `MODEL_PATH` 换成**本轮最新 SFT** 导出的 HuggingFace 目录。不要继续复用 compact 协议更新前的旧 checkpoint：

如果你要**续跑当前已经到 100 step 的实验**，把总步数调大，并显式打开验证频率：

```bash
MODEL_PATH=/data/linyujie/Cast-R1-TS-main/Cast-R1-TS-main/artifacts/checkpoints/sft/time_series_forecast_sft_compact_state_v1/global_step_11/huggingface \
PROFILE_PATH=examples/time_series_forecast/configs/etth1_ot_qwen3_gpu012.sh \
RUN_MODE=train \
bash examples/time_series_forecast/run_qwen3-1.7B.sh \
  trainer.total_training_steps=200 \
  trainer.test_freq=5
```

如果你要**新开一轮 RL 实验**，不要和当前 `etth1_ot_qwen3_1_7b_rl_gpu012` 混用，用新的实验名并禁用自动续跑：

```bash
DEBUG_CHAIN=1 \
TS_CHAIN_DEBUG_FILE=$PWD/logs/debug/ts_chain_debug_eval5_test.jsonl \
MODEL_PATH=/data/linyujie/Cast-R1-TS-main/Cast-R1-TS-main/artifacts/checkpoints/sft/time_series_forecast_sft_teacher200_new/global_step_11/huggingface \
PROFILE_PATH=examples/time_series_forecast/configs/etth1_ot_qwen3_gpu012.sh \
RUN_MODE=train \
RL_EXP_NAME=etth1_ot_qwen3_1_7b_rl_gpu012_eval5_test \
bash examples/time_series_forecast/run_qwen3-1.7B.sh \
  trainer.total_training_steps=100 \
  trainer.test_freq=5 \
  trainer.resume_mode=disable
```

RL 输出目录：

```bash
artifacts/checkpoints/rl/
```

调试辅助（可选）：

```bash
# 从 chain debug 中抽取 format 失败样本 + reward 链路摘要
PYTHONPATH=$PWD conda run -n cast-r1-ts \
python recipe/time_series_forecast/analyze_chain_debug.py \
  --debug-file $PWD/logs/debug/ts_chain_debug_eval5_test.jsonl \
  --top-k-fail 10

# 校验 SFT/curated 数据里 Turn3 最终答案是否满足 96 行约束
PYTHONPATH=$PWD conda run -n cast-r1-ts \
python recipe/time_series_forecast/validate_turn3_format.py \
  --input-jsonl dataset/ett_sft_etth1_runtime_ot_teacher200_gpu/train_curated.jsonl \
  --expected-len 96 \
  --top-k-invalid 10
```

说明：

- RL 默认 `resume_mode=auto`。同一个 `RL_EXP_NAME` 下再次运行，不会从头覆盖，而是会自动续跑最近的 checkpoint。
- reward 默认开启严格长度约束（`TS_REWARD_STRICT_LENGTH=1`）：`pred_len != gt_len` 直接视为不合法答案并给 `-1.0`。如需回退旧行为可临时设置 `TS_REWARD_STRICT_LENGTH=0`。
- 如果当前实验已经跑到 `100 step`，再写 `trainer.total_training_steps=100` 通常不会继续训练；续跑时要把总步数设得更大，比如 `200`。
- 如果你想完全重新开始，请换一个新的 `RL_EXP_NAME`，或者加 `trainer.resume_mode=disable`。
- 当前这条 ETTh1/OT 训练链路里，`vLLM + load_format=dummy` 会导致 rollout 输出异常长的乱码文本，表现为 `response_length=3072`、`reward=0`。
- 现在脚本已经默认改成 `safetensors`，不需要再手工额外覆盖；只有在你明确要测试别的加载方式时，才需要自己改 `RL_ROLLOUT_LOAD_FORMAT`。
- 如果训练已经跑到目标 step 并写出 `global_step_x`，但最后只剩下 dataloader / vLLM 的退出日志，先以 checkpoint 是否成功落盘为准；当前默认配置已经把这类退出期噪声尽量收到了最小。

## 常见问题（FAQ）

### 1) `response_length` 设得不合理，导致长度失败或输出噪声多

- 现象：`pred_len!=96` 比例升高，或 reward 里 `length_mismatch` 增多。
- 建议：先用 profile 默认的 `RL_MAX_RESPONSE_LENGTH=4096`，不要一开始就调太小。
- 若你想强化“刚好 96 行”，优先从 prompt/SFT 数据约束入手，再小幅调温度，不要先硬砍长度上限。

### 2) `temperature` 过高或过低，格式与精度会互相拉扯

- 现象：
  - 过高：格式波动增大、长度不稳定。
  - 过低：输出过于保守，探索不足，reward 提升慢。
- 建议起点：`RL_TEMPERATURE=0.6`，`RL_VAL_TEMPERATURE=0.2`（profile 默认）。
- 调参策略：每次只改一个参数，小步（如 `0.1`）调整并观察 `5~10` 个 test 周期。

### 3) Batch / token 预算过激，容易 OOM 或吞吐异常

- 重点联动参数：`RL_TRAIN_BATCH_SIZE`、`RL_PPO_MINI_BATCH_SIZE`、`RL_PPO_MICRO_BATCH_SIZE`、`RL_ACTOR_MAX_TOKEN_LEN_PER_GPU`。
- 现象：OOM、step 时间极不稳定、Ray worker 频繁重启。
- 建议：保持 profile 默认（train 模式 `3/3/1`），遇到 OOM 优先降 batch，再考虑降 token 上限。

### 4) 训练步数和续跑模式设置不当，导致“看起来没训练”

- 现象：日志正常但几乎不前进，或很快结束。
- 常见原因：`trainer.total_training_steps` 小于（或等于）当前已训练步数。
- 建议：
  - 续跑：增大 `trainer.total_training_steps`，保持 `resume_mode=auto`。
  - 全新实验：新 `RL_EXP_NAME` + `trainer.resume_mode=disable`。

### 5) 参数分散在多处，导致配置混乱

- 现在关键默认值只在一个 profile 维护：

```bash
examples/time_series_forecast/configs/etth1_ot_qwen3_gpu012.sh
```

- 启动脚本要求传 `PROFILE_PATH`；优先级固定为：**命令行覆盖 > 环境变量 > profile 默认值**。

### 6) 服务器上 `git push` 认证失败（`Permission denied (publickey)`）

- 原因：浏览器登录 GitHub 不等于服务器里的 git 凭证；且当前环境需要走代理，SSH 更容易失败。
- 推荐：直接用 HTTPS + PAT，并固定 git 代理。

```bash
git remote set-url origin https://github.com/musihai/eeh1_test.git
git config --global http.proxy http://127.0.0.1:7897
git config --global https.proxy http://127.0.0.1:7897
git config --global credential.helper store
git push origin main
```

- 首次 push 输入 GitHub 用户名与 PAT，后续会复用缓存凭证。

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

## 推荐目录划分（整理建议）

当前工作区功能完整，但目录混合了“源码、实验产物、临时日志”。建议按“长期保留 vs 可再生产物”分层，后续更容易维护和清理。

建议结构（可渐进迁移，不需要一次性改完）：

```text
Cast-R1-TS-main/
  src/                      # 训练/数据/奖励核心代码（可由现有 recipe/, arft/ 渐进映射）
  scripts/                  # 入口脚本（可由 examples/time_series_forecast/*.sh 归拢）
  configs/                  # 统一配置（可由 examples/.../configs + recipe 配置归拢）
  data/
    raw/                    # 原始数据（如 ETT-small）
    processed/
      rl/                   # RL 训练集 jsonl
      sft/                  # SFT parquet/jsonl
  artifacts/
    checkpoints/
      sft/                  # SFT checkpoint
      rl/                   # RL checkpoint
    exports/                # hf export / 最终可发布模型
  logs/
    train/                  # 训练日志（原 /tmp/rl_*.log 可软链/复制过来）
    debug/                  # chain debug jsonl
    ray/                    # ray session 日志归档
  docs/                     # 说明文档（可放 修改方案.md、排障笔记）
  tests/                    # 单测
```

与当前目录的对应关系（建议）：

- `dataset/` → 逐步映射到 `data/raw` 与 `data/processed/{rl,sft}`。
- 现在默认统一落盘到 `artifacts/checkpoints/{rl,sft}`。
- 旧 `checkpoints/TimeSeriesForecast*`、`checkpoints/time_series_forecast_sft*` 已保留软链接兼容，历史命令仍可用。
- `outputs/`、`swanlog/`、`/tmp/ts_chain_debug*.jsonl` → 统一归档到 `logs/{train,debug}`。
- `examples/time_series_forecast/*.sh` → 逐步沉淀到 `scripts/`（保留兼容软链接即可）。
- `recipe/time_series_forecast/` + `arft/` → 保持现状可用，后续按模块逐步迁到 `src/`。

建议先做三件事（收益最高）：

1. 新建 `artifacts/` 与 `logs/` 两层，把 checkpoint / debug / train log 统一落盘。
2. 保持脚本入口不变，但把路径变量统一为这两层（避免到处写绝对路径）。
3. 每个实验固定一套命名：`<task>_<model>_<stage>_<date>`，并与 `RL_EXP_NAME` 对齐。

## 一句话注意事项

- 如果你重建过 `teacher200_gpu` 或更新过 compact 协议，就重新跑一次 SFT，不要直接接旧的 RL checkpoint。
- 工具服务命令是前台常驻，不会自动返回 shell。
- 正式 RL 之前，优先保证 SFT 已经成功导出 `huggingface` 目录。
- 如果你确实要启用 `swanlab`，手动覆盖 `SFT_LOGGER='["console","swanlab"]'` 或 `RL_LOGGER='["console","swanlab"]'`。
