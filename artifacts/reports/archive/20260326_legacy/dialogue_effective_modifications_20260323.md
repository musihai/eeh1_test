# 本轮对话有效修改总结

更新时间：`2026-03-23`

这份文档只总结两类内容：

1. 当前工作区里已经真实落到代码/脚本/文档的有效修改
2. 这些修改分别解决了什么问题

不写“讨论过但没有真正落地”的内容。

## 总体结论

这轮修改的主线目标有三条：

- 把 `SFT -> RL` 的三阶段流程重新收回到论文主线
- 把训练链、评测链、debug 观测链统一到同一套协议
- 清掉旧入口、旧脚本、旧路径，重新建立正式实验入口

从当前工作区 diff 来看，核心改动集中在：

- `recipe/time_series_forecast/`
- `examples/time_series_forecast/`
- `arft/ray_agent_trainer.py`
- `README.md`
- 对应测试文件

## 1. 三阶段工作流重新对齐

### 修改

涉及文件：

- `recipe/time_series_forecast/base.yaml`
- `recipe/time_series_forecast/diagnostic_policy.py`
- `recipe/time_series_forecast/prompts.py`
- `recipe/time_series_forecast/time_series_forecast_agent_flow.py`
- `recipe/time_series_forecast/build_etth1_rl_dataset.py`

关键改动：

- `max_steps` 改为 `4`，并增加 `max_prediction_attempts: 2`
- 诊断工具 batching 改为按顺序分批，不再走旧的隐式拆分逻辑
- prompt 明确区分 `diagnostic -> routing -> refinement`
- runtime 中强制：
  - 诊断阶段只允许 feature tools
  - routing 阶段只允许 `predict_time_series`
  - refinement 阶段不允许工具调用
- RL 数据 prompt 改成明确的 `<think>...</think><answer>...</answer>` 协议描述

### 解决的问题

- 修复了旧 `max_steps=3` 导致流程被提前截断的问题
- 修复了 SFT 轨迹和 RL runtime 的 turn budget 不一致问题
- 修复了 refinement 阶段继续调工具、routing 阶段调错工具的协议漂移问题
- 把 workflow 重新收敛到论文主线的三阶段流程

## 2. `predict_time_series` 调用与重试逻辑修正

### 修改

涉及文件：

- `recipe/time_series_forecast/time_series_forecast_agent_flow.py`

关键改动：

- 增加 `prediction_attempt_count` / `max_prediction_attempts`
- 只有 routing 阶段允许预测
- 预测失败时回滚 `prediction_call_count` / `prediction_step_index`
- episode 的有效 step budget 会结合 prediction retry 预算动态计算

### 解决的问题

- 修复了“第一次预测失败后状态被污染，后续 episode 基本不可恢复”的问题
- 修复了 `prediction_call_count` 和最终校验不一致的问题
- 避免了 routing 轮无限重试或错误重试

## 3. SFT 构造链重新收紧到论文主线

### 修改

涉及文件：

- `recipe/time_series_forecast/build_etth1_sft_dataset.py`

关键改动：

- 默认输出目录切到新的 step-wise 主线目录
- 默认输入不再指向旧 RL jsonl，而是指向 `teacher-curated` jsonl
- `build_feature_tool_results()` 改成固定按 `FEATURE_TOOL_BUILDERS` 产出完整 feature tools
- 不再使用旧的 `select_feature_tool_names()` 预选 teacher tools
- SFT record 从单条 transcript 改为 step-wise stage records：
  - diagnostic records
  - routing record
  - refinement record
- 引入 `turn3_target_mode`
  - 默认：`paper_strict`
  - 可选：`engineering_refine`
- `paper_strict` 下 Turn-3 默认保留 selected forecast，不再默认套局部规则修补器
- 对 `teacher_prediction_text` 增加缓存有效性检查；坏缓存会回退到 runtime 重算
- parquet 写出前增加 Turn-3 paper protocol 校验
- metadata 改成 `runtime_stepwise_sft`，并记录 protocol/target-mode 分布

### 解决的问题

- 修复了 builder 默认路径和当前数据协议不一致的问题
- 修复了 teacher 轨迹里“先用隐藏启发式替 agent 选工具”的 oracle 风险
- 修复了旧 SFT 监督和 runtime 三阶段分布不一致的问题
- 修复了坏缓存会直接把构建打死的问题
- 修复了 Turn-3 默认监督偏工程后处理、偏离论文文字主线的问题

## 4. Teacher-curation 与数据身份协议统一

### 修改

涉及文件：

- `recipe/time_series_forecast/build_etth1_high_quality_sft.py`
- `recipe/time_series_forecast/dataset_identity.py`

关键改动：

- teacher-eval 显式以 `allow_recovery=True` 比较原始预测串，而不是把它当 final Turn-3 协议
- teacher-curated 输出 metadata 改为：
  - `dataset_kind = runtime_sft_parquet`
  - `pipeline_stage = teacher200_runtime_sft`
  - 同时保留 `curated_jsonl_dataset_kind`
- `dataset_identity` 允许 `expected_kind` 接受多个合法 kind

### 解决的问题

- 修复了 teacher-curated 数据和 runtime-SFT 数据在 metadata kind 上对不齐的问题
- 修复了 builder/launcher 在 sibling metadata 校验时过于僵硬的问题
- 把 teacher-eval 的宽松解析限定在该使用的地方，不污染正式 final protocol 口径

## 5. Reward、离线校验和 debug 观测链统一到 strict protocol

### 修改

涉及文件：

- `recipe/time_series_forecast/reward.py`
- `recipe/time_series_forecast/validate_turn3_format.py`
- `recipe/time_series_forecast/time_series_forecast_agent_flow.py`
- `arft/ray_agent_trainer.py`

关键改动：

- `reward.py` 默认 `allow_recovery=False`
- `validate_turn3_format.py` 默认 strict；只有明确需要的路径才走宽松检查
- runtime 打分链改成按原始 final output 严格协议解析
- `reward_extra_info` / validation 导出链补齐了：
  - `required_feature_tool_count`
  - `missing_required_feature_tool_count`
  - `prediction_step_index`
  - `final_answer_step_index`
  - `sample_uid`
- trainer 的 `eval_step_samples.jsonl` / `eval_step_aggregate.jsonl` 聚合补齐了这些字段
- 统一 debug 输出到 `TS_CHAIN_DEBUG_FILE`，不再保留旧的独立 turn3 debug 路径协议

### 解决的问题

- 修复了主 runtime strict、离线分析宽松，导致离线结果偏乐观的问题
- 修复了 `ts_chain_debug` 和 `eval_step_*` 字段不一致的问题
- 修复了 validation 下 `sample_index` 不足以追样本的问题
- 修复了旧的 turn3 debug 路径协议与当前主链脱节的问题

## 6. 正式实验入口、命名和默认参数重建

### 修改

涉及文件：

- `examples/time_series_forecast/configs/etth1_ot_qwen3_gpu012.sh`
- `examples/time_series_forecast/run_qwen3-1.7B.sh`
- `examples/time_series_forecast/run_qwen3-1.7B_sft.sh`
- `artifacts/reports/final_launch_cmd.txt`
- `README.md`

关键改动：

- 正式实验统一命名为 `paper_strict_formal_20260323`
- RL / SFT 的 project 与 experiment name 全部切到 formal 主线
- launcher 默认路径切到新主线：
  - RL 数据：`ett_rl_etth1_paper_aligned_ot_curriculum_same2`
  - SFT checkpoint：`time_series_forecast_sft_paper_strict_formal_20260323`
- RL 默认参数重设为正式主线：
  - `rollout.n=8`
  - `temperature=1.0`
  - `val_temperature=0.0`
  - `max_response_length=4096`
  - `gpu_memory_utilization=0.15`
  - `max_num_seqs=4`
  - `enable_chunked_prefill=True`
- `run_qwen3-1.7B.sh` 强制优先使用 `RL_MODEL_PATH`
- 增加无效 rollout 参数组合的前置拦截
- `run_qwen3-1.7B_sft.sh` 增加 parquet 的 Turn-3 paper protocol 预校验
- README 重写成新的正式流程文档，并记录实际 SFT / RL val-only 结果

### 解决的问题

- 修复了旧 train/smoke 入口仍指向旧数据集、旧 checkpoint 的问题
- 修复了默认 RL 参数与论文 formal 设置偏差过大的问题
- 修复了 SFT/RL 启动时“路径对了但协议错了”的无声失败风险
- 让正式实验可以从单一 profile 和单一命名体系启动

## 7. 项目清理：删除旧脚本和非主线代码

### 修改

已删除文件：

- `recipe/time_series_forecast/analyze_chain_debug.py`
- `recipe/time_series_forecast/benchmark_models_on_rl_samples.py`
- `recipe/time_series_forecast/build_etth1_sft_subset.py`
- `recipe/time_series_forecast/retrain_expert_models_train_split.py`
- `tests/test_sft_subset_builder.py`

新增/保留清理记录：

- `artifacts/reports/archive_candidates_20260322.md`

### 解决的问题

- 去掉了不再服务于正式主线的离线辅助脚本
- 减少了旧路径、旧流程、旧调试习惯继续误导实验入口的风险
- 让 README 和主线代码对应关系更清晰

## 8. 测试补强

### 修改

新增或扩展的测试文件包括：

- `tests/test_diagnostic_policy.py`
- `tests/test_time_series_forecast_agent_flow.py`
- `tests/test_ray_agent_trainer_validation.py`
- `tests/test_sft_dataset_builder.py`
- `tests/test_validate_turn3_format.py`
- `tests/test_compact_protocol.py`
- `tests/test_high_quality_sft_builder.py`
- `tests/test_final_answer_parsing.py`
- `tests/test_etth1_feature_smoke.py`

### 解决的问题

- 给三阶段 workflow、strict protocol、validation 导出链、SFT builder 和 teacher builder 增加了回归保护
- 降低了后续再次出现“表面能跑、但 SFT/RL 协议已经漂了”的风险

## 9. 这轮修改实际解决的核心问题清单

可以归纳成下面 10 个已解决问题：

1. `max_steps=3` 导致 episode 在正式三阶段流程里被提前截断
2. `predict_time_series` 失败后状态污染，episode 不可恢复
3. 诊断、路由、refinement 的工具协议不一致
4. builder 默认输入路径仍指向旧 RL jsonl，不匹配当前 teacher-curated 主线
5. teacher 轨迹里存在隐式 oracle 式工具预选
6. Turn-3 默认目标过度依赖规则修补器，偏离论文主线
7. reward / validate / runtime 对 final protocol 的 strictness 不一致
8. `eval_step_*` 与 `ts_chain_debug` 的字段链不一致
9. 正式 train/smoke 入口仍可能落回旧 checkpoint / 旧数据 / 旧命名
10. 项目里残留若干不再属于正式主线的脚本和测试

## 10. 本轮有意识保留、没有主动改动的部分

这轮修改没有刻意去改下面这些东西：

- 没有把预测服务改成 CPU 主线；正式实验仍保持“训练卡和服务卡分离”
- 没有把 final answer 做硬解码或强制后处理美化
- 没有为了追求更高 reward 去改论文主线之外的目标设计

换句话说，这轮主要做的是“把流程和协议修正到能严肃复现实验”的工作，而不是为了出更好看的指标做额外工程增强。

## 11. 当前文档对应的证据范围

这份总结对应的是：

- 当前工作区 `git diff --stat`
- 当前仍在使用的正式 README
- 当前正式 SFT / RL 入口脚本
- 当前 formal run 的实际运行状态

如果后续继续修改代码，这份文档也需要同步更新。
