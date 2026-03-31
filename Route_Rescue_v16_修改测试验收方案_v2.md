# Route Rescue v16 修改 + 测试 + 验收方案

## 1. 目标与当前判断

当前状态下 **不能进入 RL**。  
原因已经很清楚：

- **refine 已基本达标**：  
  - `strict_ok_rate = 1.0`
  - `overall_changed_ratio = 13.9%`
  - `local_refine` 中有较高比例真正改到了 refined target  
- **route 未达标**：  
  - full SFT 在 val routing 样本上的 agreement 明显过低
  - 高置信 `mid/high` 样本上的 agreement 更低
  - route-only warm-up 自身也没学稳
- **route supervision 本身存在问题**：  
  - routing target 仍在注入 `RouteDecision(... reason_codes=...)`、`RouteSummary: ...` 等合成 heuristics 文本
  - 高置信 routing 样本太少，低置信样本占多数
- **protocol probe 当前口径不适用于 engineering_refine**：  
  - 当前 Turn 3 合法输出是 `decision=...`
  - 正确链路应先 `materialize_refinement_decision(...)`
  - 不能直接拿旧 `parse_final_answer_protocol(...)` 的 0% strict 作为否决依据

因此，当前应该进入：

## Stage 3.5：Route Rescue（路由抢救）

目标：

1. 彻底修正 route supervision 形式
2. 单独扩建一个高置信、平衡的 routing bootstrap 数据集
3. 做 routing repair SFT
4. 再做一次短的 full refresh SFT
5. route 过门后，才允许进入 RL

---

## 2. 根因判断

### 2.1 routing target 被 synthetic heuristics 污染
当前 routing target 中仍然存在：

- `RouteDecision(model=..., reason_codes=[...])`
- `RouteSummary: ...`

这会让模型学习成：

- “理由码模板 -> expert”

而不是：

- “根据状态证据 -> 选择 model_name”

### 2.2 高置信 routing 样本过少
当前统计显示：

- `routing_only/train.parquet` 中 `mid+high` 样本过少
- `full stepwise` 的 routing 行中低置信样本仍占绝对多数

这意味着 route-only warm-up 阶段真正值得强监督的样本太少，难以学稳。

### 2.3 ETTh1 上 route 标签本身有噪声
许多窗口中 best expert 与 second-best expert 的误差差距不大。  
因此 route 不是一个“天然非常清晰”的 4 分类任务，只适合对高 margin 窗口做强监督。

---

## 3. 修改方案

---

# Part A：修正 route supervision 形式

## A.1 核心原则
routing 阶段 target 必须改成：

- **极简动作监督**
- 不再注入 synthetic heuristics 文本

### routing target 只保留：
- 一个极短的 `<think>`
- 一个严格结构化的 `predict_time_series` tool call

### 推荐目标格式
```xml
<think>Use the diagnostic evidence to choose one forecasting model.</think>
<tool_call>
{"name":"predict_time_series","arguments":{"model_name":"patchtst"}}
</tool_call>
```

### 必须删除的内容
- `RouteDecision(... reason_codes=...)`
- `RouteSummary: ...`
- 任何人工合成的理由码文本
- 任何 expert-support 风格的中间描述

---

## A.2 需要修改的文件

### 文件
`recipe/time_series_forecast/build_etth1_sft_dataset.py`

### 修改点 1：routing stage assistant target 极简化
把 routing stage 的 assistant target 生成逻辑改成：

- `<think>`：一句中性话
- `<tool_call>`：只保留 `model_name`
- 不拼接任何 `reason_codes` / `RouteSummary`

### 修改点 2：不要把 synthetic route 文本写入后续 history
除了 target 本身要改，还要确保：

- routing synthetic 文本不进入后续 turn 的 memory/history

否则 full stepwise 时，Turn 3 仍会读到污染信息。

---

# Part B：重建 routing bootstrap 数据集

## B.1 为什么要单独重建
当前 route-only 数据中，高置信样本太少，不能有效训练 route policy。  
因此需要一个专门用于 route repair 的数据集：

## 新数据集命名建议
`routing_bootstrap_v16`

---

## B.2 数据集构建原则

### 数据源
不要只从最终 200 curated 样本里取。  
应从更大的 train pool 中重新挖 route 样本。

### 建议规模
最少从 **1000~3000 个 train 窗口** 中重新评估 4 个 expert。

### 每个窗口需要保存的字段
- `best_model`
- `second_best_model`
- `best_error`
- `second_best_error`
- `margin_abs = second_best_error - best_error`
- `margin_rel = (second_best_error - best_error) / second_best_error`

---

## B.3 route 样本筛选策略

### 不建议
不要继续只用固定的全局 `mid/high` 阈值。

### 建议做法
对每个 winning expert 单独分层：

1. 对该 expert 获胜的所有窗口，按 `margin_abs` 或 `margin_rel` 排序
2. 取前 **30%~40%** 作为“可学 route 样本”
3. 每个 expert 至少保留 **100~150 条**
4. 最终做平衡采样

### 目标规模
- train：`512 ~ 1024`
- val：`>= 128`
- 其中高置信 val：`>= 32`

---

## B.4 路由标签要求
routing bootstrap 中的 label 必须只包含：

- 目标 `model_name`

不要包含：

- `reason_codes`
- `RouteSummary`
- 任何辅助解释文本

---

# Part C：Routing Repair SFT

## C.1 训练起点
建议直接从当前已经训好的：

- `full v15 checkpoint`
- `global_step_142/huggingface`

继续做 route repair SFT。

### 原因
- refine 已基本修好
- 当前目标是把 route 拉回来
- 从 full v15 继续训成本最低

---

## C.2 训练设置建议

### 数据
使用新的 `routing_bootstrap_v16`

### 训练方式
只训 `routing_only`

### 推荐参数
- learning rate：`5e-6`
- epoch：`0.5 ~ 1.0`
- 每隔固定 step 做一次 route probe
- 按高置信 agreement / route regret 早停

---

## C.3 何时放弃“从 full v15 继续”
如果 route repair SFT 后仍然出现：

- 高置信 agreement 依然极低
- 仍然只在 `arima/chronos2` 间摆动
- 4 个 expert 分布完全打不开

则说明当前 full v15 已经对 route 形成强坏先验。  
这时再考虑：

- 从更早 checkpoint 继续
- 或从 base / routing warm-up 起点重训 route-only

---

# Part D：Full Refresh SFT

## D.1 为什么需要 full refresh
route repair 只训练 routing-only，可能会导致：

- refine 风格轻微漂移
- 三阶段衔接不够自然

因此 route repair 成功后，需要做一次短的 full refresh SFT。

---

## D.2 新 full 数据集命名建议
`full_stepwise_v16`

### 核心要求
1. routing rows 只保留“可学 route 样本”
2. per-model 平衡采样
3. 高置信 route 样本权重更高
4. diagnostic / refinement 继续保留主线
5. Turn 3 继续使用 `engineering_refine`

---

## D.3 full refresh 训练建议
### 起点
route repair 后的 checkpoint

### epoch
`0.2 ~ 0.5`

### 目的
把：

- route
- refine
- protocol
- 3-stage 衔接

重新揉回一个统一分布。

---

# Part E：修正 protocol probe 口径

## E.1 当前问题
你已经确认当前 probe 逻辑不适配 engineering_refine：

- 旧 probe 直接把生成文本送进 `parse_final_answer_protocol(...)`
- 但当前合法输出其实是 `decision=...`

### 正确链路
应当先：

1. `materialize_refinement_decision(...)`
2. 再检查最终 materialized answer 是否符合 protocol

---

## E.2 需要修改的文件
`probe_refinement_protocol.py`

### 修改目标
针对 `engineering_refine`：

- 先识别 `decision=...`
- 先走 `materialize_refinement_decision(...)`
- 再统计 strict / valid / accept

### 验收提醒
在 probe 没修口径前，不得再用旧 `protocol strict = 0%` 作为失败依据。

---

## 4. 测试方案

---

# Test A：routing bootstrap probe

## A.1 不能只看 exact agreement
这次 route probe 至少看 4 类指标：

### 1）exact agreement
- overall exact agreement
- high-confidence exact agreement

### 2）top-2 agreement
判断模型预测的 expert 是否落在窗口的 top-2 experts 中。

### 3）mean route regret
定义：

- `route_regret = chosen_expert_error - best_expert_error`

统计：

- overall mean regret
- high-confidence mean regret

### 4）predicted model distribution
检查是否再次塌缩。

---

## A.2 routing bootstrap probe 的最低通过门槛

- overall exact agreement ≥ **50%**
- high-confidence exact agreement ≥ **70%**
- top-2 agreement ≥ **85%**
- 4 个 expert 都能被预测到
- 无单 expert 占比 > **70%**
- high-confidence mean regret 明显下降
- 相比当前 full v15：
  - high-confidence exact agreement 至少提升 **30 个百分点**

---

# Test B：full refresh 后三组 probe

## B.1 refine probe
refine 不能退化：

- `strict_ok_rate = 1.0`
- `overall_changed_ratio` 继续在 **5% ~ 40%**
- `local_refine` 命中 refined target 比例 ≥ **60%**

## B.2 protocol probe
在修正 probe 口径后，再统计：

- strict / valid / accept

但前提必须是：

- 先走 `materialize_refinement_decision(...)`

## B.3 routing probe
full refresh 后 route 仍必须通过 route probe。

如果 full refresh 后 route 又明显掉回去，说明：

- full 数据集仍在污染 route

此时不能进入 RL。

---

## 5. 验收标准

---

# 第一关：Route Repair SFT 通过标准
在 `routing_bootstrap_v16` 验证集上：

- overall exact agreement ≥ **50%**
- high-confidence exact agreement ≥ **70%**
- top-2 agreement ≥ **85%**
- 4 个 expert 都有输出
- 无单 expert 占比 > **70%**
- mean regret 相比当前 full v15 明显下降

---

# 第二关：Full Refresh SFT 通过标准
## refine 必须继续通过
- `strict_ok_rate = 1.0`
- `overall_changed_ratio in [5%, 40%]`
- `local_refine_target_hit >= 60%`

## route 必须保持通过
- full refresh 后 route probe 不能明显退化
- 高置信 agreement 必须仍达到 route repair 阶段的主要门槛

## protocol 必须按新口径通过
- engineering_refine -> materialize -> final answer check

---

# 第三关：只有前两关都通过，才允许进入 RL
如果 route repair / full refresh 中任一环未通过，则：

- **禁止进入 RL**

---

## 6. 失败后的升级策略

如果完成 Route Rescue 后仍然出现：

- high-confidence exact agreement 上不去
- top-2 也不高
- regret 不下降
- 仍只在少数 expert 之间摆动

则说明：

- 在 ETTh1 + 当前 4 expert + 当前 state 表达下
- route task 的可学性本身不足

此时再进入下一步结构增强：

## 最小结构增强（不是 full v2）
保留 3-turn：

- Turn 1：diagnostic
- Turn 2：route
- Turn 3：final_select

但 Turn 3 允许从：

- routed expert baseline / refine
- best alternative expert baseline / refine

中选择最终方案。

这样仍尽量保留论文叙事，同时让 Turn 3 具备真正补救 route 错误的能力。

---

## 7. 推荐执行顺序

1. **先修 builder**
   - 去掉 `RouteDecision(... reason_codes=...)`
   - 去掉 `RouteSummary`
   - 不再把 synthetic heuristics 写入后续 history

2. **重建 `routing_bootstrap_v16`**
   - 从更大 train pool 挖样本
   - 每个 winner class 单独按 margin 分层
   - 做平衡采样

3. **从 full v15 checkpoint 做 route repair SFT**
   - 只训 routing-only
   - 每个 probe 周期看：
     - exact agreement
     - top-2 agreement
     - route regret
     - model distribution

4. **route repair 通过后，重建 `full_stepwise_v16`**

5. **做短的 full refresh SFT**
   - 确认 refine 不退化
   - route 不回落

6. **修 protocol probe 口径**
   - engineering_refine 先 materialize 再统计 strict/accept

7. **只有 Route Rescue 全部通过，才允许进 RL**

---

## 8. 一句话结论

前面的修改**没有白做**，因为 refine 已明显修好。  
现在的瓶颈已经非常清楚：

- **不是 Turn 3**
- **不是协议**
- **而是 route supervision 的形式和 route 数据质量**

因此当前最正确的动作不是进入 RL，也不是推翻重来，而是：

## **Stage 3.5：Route Rescue**
核心就是三件事：

1. **去掉 route synthetic heuristics target**
2. **扩建高置信、平衡的 routing bootstrap 数据**
3. **先做 route repair SFT，再做 full refresh SFT**

通过后，再进入 RL。
