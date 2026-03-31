# Route_Relaxation_v17 修改 + 测试 + 验收方案

## 1. 背景与目标

在完成 **Route Rescue v16** 后，当前结论已经比较明确：

- **refine 已基本修好**  
  - `strict_ok_rate = 1.0`
  - `overall_changed_ratio` 落在合理区间
  - `local_refine` 已经具备有效命中 refined target 的能力
- **route 仍未学稳**
  - exact single-expert routing 在当前设置下长期无法过门
  - 即使 route repair / from-base repair 也无法同时满足：
    - exact agreement
    - high-confidence agreement
    - top-2 agreement
    - 全 expert 覆盖
    - 不塌缩
- **更关键的是**：
  - exact agreement 低，并不总是对应更高 regret
  - 有些 collapse 到单 expert 的模型，虽然 exact 更差，但 regret 更低

这说明当前问题已经不是：

- 协议崩坏
- Turn 3 全 reject
- refine 不会动
- warm-start 污染

而是：

## 当前 route 任务定义过强
当前 v16 使用的 route 目标本质上是：

- 在 4 个 fixed experts 中
- 精确命中 oracle best expert

但在 **ETTh1 + 当前状态表示 + 当前 expert 集合** 下，这个标签本身是 **高噪声、低可分** 的。

---

## v17 的核心目标

因此，v17 不再继续追求：

- **exact single-expert route classification**

而改成：

## **默认专家（default expert） + 置信 override**

也就是：

- 先确定一个稳定的 **default expert**
- Turn 2 只学习：
  - `keep_default`
  - 或 `override_to_x`

这样 route 任务从：

- noisy 的 4 分类精确匹配

变成：

- 更可学的 **keep vs override**
- 再加一个较小的 override 方向选择

---

## v17 的设计原则

1. **尽量对齐论文**
   - 保留 Cast-R1 的 3-turn 语义：
     - Turn 1：diagnostic
     - Turn 2：route decision
     - Turn 3：reflection / refinement / final answer
   - 不直接跳成 full v2 两步 rerank

2. **承认 ETTh1 上 route task 的真实可学性边界**
   - 不再把 exact best expert 当唯一正确标签
   - route 的目标改为：  
     **相对于 default expert 是否值得偏离**

3. **让 Turn 3 真正承担 final selection / refinement 的职责**
   - Turn 2 不再过度承担“精确一锤定音选最优 expert”的责任
   - Turn 3 负责在 default / override 路径中做更稳的最终决策

---

# 2. 总体路线

v17 分为 5 个阶段：

1. **确定 default expert**
2. **重建 route 标签：从 exact-best 改为 keep_default / override**
3. **重建 routing override bootstrap 数据集**
4. **做 route-override warm-up SFT**
5. **做 full stepwise v17 refresh SFT，并重新 gate check**

---

# 3. Phase A：确定 default expert

## A.1 原则

default expert 不是按 “winner 次数最多” 选，  
而是按 **整体稳定性** 选。

### 优先考虑指标
- 全局 mean regret 最低
- 全局 mean MSE 稳定
- 在验证集上表现最稳
- collapse 成该 expert 的 policy，若 regret 仍较低，则说明它更适合作为 default

---

## A.2 当前建议

根据你当前 gate check：

- `route repair from base` collapse 到 `itransformer`
- 虽然 exact agreement 很差，但 **mean regret 最低**

因此：

## 建议将 `itransformer` 设为 v17 的 default expert

---

## A.3 记录 baseline

确定 default expert 后，需要记录两个 baseline：

- `default_expert_mean_mse`
- `default_expert_mean_regret`

它们将作为后续 v17 route / full stepwise 的最低比较线。

---

# 4. Phase B：重建 route 标签（从 exact best 改为 keep_default / override）

## B.1 新标签定义

对每个窗口，计算：

- `default_error`
- `best_error`
- `best_model`

定义：

Δ = default_error - best_error

然后：

### 情况 1：best_model == default_expert
标签定义为：

- `keep_default`

### 情况 2：best_model != default_expert
如果：

- `best_model != default_expert`
- 且 `Δ` 足够大（说明 override 确实有意义）

则标签定义为：

- `override_to_patchtst`
- `override_to_arima`
- `override_to_chronos2`

如果 default 未来改成别的 expert，则 override 集合相应调整。

---

## B.2 关键：必须设置 override 阈值

如果不设阈值，你只是把 noisy exact label 换了个形式重新引入。

### 推荐阈值策略（二选一）

---

### 方法 1：相对 improvement 阈值

定义：

(default_error - best_error) / default_error >= τ

推荐先试两档：

- `τ = 5%`
- `τ = 8%`

解释：

- 只有当 best expert 相比 default 真正带来足够收益时，才值得打 override 标签
- 否则全部并入 `keep_default`

---

### 方法 2：按 override model 分层取 top margin

对每个非 default expert：

1. 只考虑 `best_model == 该 expert` 的窗口
2. 按相对于 default 的 improvement 排序
3. 只保留前 30%~40% 作为 override 样本

这通常比固定阈值更稳。

---

## B.3 目标标签分布

新的 route 标签 **不再要求 4 类均衡**。

目标分布应该是：

- `keep_default`：主类
- `override_*`：少数但高质量样本

### 推荐比例
- `keep_default`：60% ~ 80%
- 所有 override 合计：20% ~ 40%

如果 override 太少：
- route 学不到偏离 default 的能力

如果 override 太多：
- 阈值太松，噪声会重新回流

---

# 5. Phase C：重建 routing override bootstrap 数据集

## C.1 数据集命名建议

建议新建目录：

- `routing_override_bootstrap_v17`

---

## C.2 数据源

不要只从最终 200 curated 样本里取。  
应从更大的 train pool 中重新挖 route 样本。

### 建议规模
- train pool：1000 ~ 3000 个窗口
- route bootstrap train：512 ~ 1024
- route bootstrap val：>= 128

---

## C.3 每个样本至少保存的字段

- `default_expert`
- `default_error`
- `best_model`
- `best_error`
- `improvement_vs_default`
- `route_label`  
  （取值为 `keep_default` 或 `override_to_x`）
- `route_label_confidence`
- `history / diagnostic evidence`

---

## C.4 数据清洗要求

### 必须保证：
- 不再注入 synthetic route heuristics：
  - `RouteDecision(... reason_codes=...)`
  - `RouteSummary`
  - 任意 route support codes
- 标签只表达：
  - keep_default
  - override_to_x

---

## C.5 验收要求

重建后的 `routing_override_bootstrap_v17` 应满足：

- `keep_default` 占比在 60%~80%
- 所有 override 标签合计在 20%~40%
- 各 override 子类都有样本
- 不允许某个 override 子类完全缺失
- train / val split 无 source overlap

---

# 6. Phase D：修改 Turn 2 的 action schema 与 target

## D.1 Turn 2 不再伪装成 predict_time_series(model_name=...)

当前 Turn 2 的真实任务已经不再是：

- “从 4 个 expert 里精确选谁来预测”

而是：

- “保留 default，还是 override 到另一个 expert”

因此建议 **单独定义 route action**：

## 新 action 名称建议
- `route_time_series`

---

## D.2 新 target 形式

### keep_default 的目标格式

```xml
<think>Use the evidence to decide whether the default forecaster should be kept or overridden.</think>
<tool_call>
{"name":"route_time_series","arguments":{"decision":"keep_default"}}
</tool_call>
```

### override 的目标格式

```xml
<think>The evidence suggests overriding the default forecaster.</think>
<tool_call>
{"name":"route_time_series","arguments":{"decision":"override","model_name":"patchtst"}}
</tool_call>
```

---

## D.3 修改原则

routing target 中：

- 不再输出 4 选 1 的 `predict_time_series`
- 不再出现 `reason_codes`
- 不再出现 `RouteSummary`
- 只保留：
  - `keep_default`
  - 或 `override + model_name`

---

## D.4 需要修改的文件

### 文件 1
`recipe/time_series_forecast/build_etth1_sft_dataset.py`
- 生成新的 route label
- 生成新的 route target

### 文件 2
`recipe/time_series_forecast/prompts.py`
- 修改 Turn 2 prompt
- 把任务描述从“选一个 expert”改成“是否保留 default，必要时 override”

### 文件 3
`time_series_forecast_agent_flow.py`
- 支持新 action：
  - `route_time_series(decision=..., model_name=...)`
- route state 中记录：
  - default choice
  - override choice

### 文件 4
相关 parser / probe 文件
- 支持新的 Turn 2 route schema

---

# 7. Phase E：增强 Turn 3 candidate pool（但仍保留 3-turn）

## E.1 为什么 Turn 3 要同步增强

如果 Turn 2 只学：

- keep_default
- 或 override_to_x

那么 Turn 3 必须能够在两条路径中做 final selection / refine。

否则 Turn 2 变柔了，Turn 3 还沿用旧的“只围绕 routed expert 做 3 个 edit”，最终效果仍然会受限。

---

## E.2 v17 的最小增强 candidate pool

建议 Turn 3 至少包含：

### 必选候选
- `default_expert__keep`
- `override_expert__keep`（如果 Turn 2 选择了 override）
- `default_expert` 的局部 refine 候选
- `override_expert` 的局部 refine 候选

### 第一轮不建议扩太大
暂时不要一口气引入所有 expert 的所有 refine 候选。  
先保留：

- default 路线
- override 路线

这样仍然符合 Cast-R1 的 3-turn 语义，且复杂度可控。

---

## E.3 Turn 3 仍保持 engineering_refine

不要退回 old paper_strict full forecast 文本生成。  
Turn 3 继续：

- `<think>...</think><answer>decision=...</answer>`
- 然后 materialize 成最终 forecast

---

# 8. Phase F：route-override warm-up SFT

## F.1 数据集

使用：

- `routing_override_bootstrap_v17`

## F.2 起点

推荐从当前最好的 full SFT checkpoint 继续：

- `global_step_142/huggingface`

原因：

- refine 已经比较稳
- 当前主要想修 route decision
- 从已有 full SFT 继续成本最低

---

## F.3 推荐训练参数

- learning rate：`5e-6`
- epoch：`0.5 ~ 1.0`
- 仅训练 routing / Turn 2
- 每个 probe 周期做 route override probe
- 按新 gate 指标早停

---

## F.4 如果继续失败怎么办

如果从 current full checkpoint 继续 route warm-up 后仍然：

- 完全不会 override
- 或 override 全部塌缩到单一 expert
- 或 `delta_vs_default_mean` 没有改善

再考虑：
- 从更早 checkpoint 继续
- 或从 base / route-only 起点重训

但第一轮先建议从当前 full checkpoint 继续。

---

# 9. Phase G：full_stepwise_v17 refresh SFT

## G.1 数据集命名建议

- `full_stepwise_v17`

---

## G.2 full_stepwise_v17 的核心要求

1. Turn 2 使用新的 keep_default / override 标签
2. Turn 3 使用增强后的 candidate pool
3. Turn 3 继续 engineering_refine
4. diagnostic 继续保留
5. synthetic route heuristics 彻底清除

---

## G.3 训练顺序

### Step 1
先做 route-override warm-up

### Step 2
route gate 通过后，再做 full refresh SFT

---

## G.4 full refresh 训练建议

- 起点：route-override warm-up 后的 checkpoint
- epoch：`0.2 ~ 0.5`

---

## G.5 stage 权重建议

相比 v16，建议进一步提升 Turn 3 重要性：

### 推荐权重
- `diagnostic : route : refinement = 1 : 2 : 3`

因为在 v17 中：

- Turn 2 不再承担精确选 oracle best expert 的职责
- Turn 3 才是最终结果质量的主要承担者

---

# 10. 测试方案（v17）

v17 的测试不再以“全体样本 exact winner agreement”作为第一核心指标。

---

## Test A：Route Override Probe

### A.1 核心指标 1：override 检测质量

把 route probe 改成：

- `keep_default` vs `override` 的二分类
- 再看 override 子集内部的 expert 选择质量

### 要统计的指标
- `override_precision`
- `override_recall`
- `override_f1`

---

### A.2 核心指标 2：delta_vs_default_mean

定义：

delta_vs_default = chosen_route_expert_error - default_expert_error

解释：

- 如果该值平均小于 0，说明 route policy 已经优于“永远固定选 default expert”

这是 v17 最重要的指标之一。

---

### A.3 核心指标 3：override 子集上的 exact / top-2 agreement

只在 override 子集上看：

- exact agreement
- top-2 agreement

解释：

- override 子集才是真正困难且有价值的 route 子任务
- 全体样本 exact 已不再是核心目标

---

### A.4 核心指标 4：distribution

查看：

- `keep_default` 占比
- 各 `override_to_x` 占比
- 是否存在 collapse

---

## Test B：Full Stepwise Probe

### B.1 refine probe
必须保证 refine 不退化：

- `strict_ok_rate = 1.0`
- `overall_changed_ratio` 继续在 5%~40%
- `local_refine` 命中 refined target 比例保持在合理区间

### B.2 protocol probe
engineering_refine 路径应：

1. 先 `materialize_refinement_decision(...)`
2. 再检查最终 answer protocol

### B.3 final performance probe
要统计：

- `orig_mse_mean`
- `final_vs_default_regret_mean`
- `final_vs_route_regret_mean`

---

# 11. 验收标准（v17）

---

## Gate 1：Route Override Gate

满足以下条件才算 Turn 2 过门：

### 必过条件
- `delta_vs_default_mean < 0`
- `override_precision >= 60%`
- `override_recall >= 50%`
- `override_f1 >= 55%`

### override 子集质量门槛
- override 子集 exact agreement ≥ **50%**
- override 子集 top-2 agreement ≥ **80%**

### 额外门槛
- 不允许所有样本都 collapse 成 `keep_default`
- 不允许所有 override 都 collapse 到单一 model，除非数据分布本身证明只有一个 override model 有效

---

## Gate 2：Full Stepwise Gate

full refresh 后，必须同时满足：

### refine 不退化
- `strict_ok_rate = 1.0`
- `overall_changed_ratio in [5%, 40%]`

### final policy 至少优于 default baseline
- `orig_mse_mean < default_expert_mean_mse`
- `final_vs_default_regret_mean < 0`

### route 行为仍合理
- Route Override Gate 的核心指标不显著退化

---

## Gate 3：进入 RL 的前置条件

只有满足以下三条，才允许进入 RL：

1. Route Override Gate 通过
2. Full Stepwise Gate 通过
3. protocol probe（按新口径）通过

否则：

- **禁止进入 RL**

---

# 12. 失败后的升级路线

如果完成 v17 后仍然出现：

- `delta_vs_default_mean >= 0`
- override F1 很低
- Turn 3 也救不回来
- final policy 仍不优于 default baseline

那么说明：

## 对 ETTh1 而言，learned route 本身不值得做

这时再考虑进入真正更强的结构调整：

- Turn 2 不再学习 route
- Turn 2 只做 proposal / shortlist
- Turn 3 做 final candidate selection

也就是接近 full v2 的时候。

但在这之前，v17 仍然是更稳、更贴近论文、也更低风险的一步。

---

# 13. 推荐执行顺序

1. **确定 default expert**
   - 先固定 `itransformer` 作为默认候选
   - 记录 `default_expert_mean_mse / regret`

2. **重建 route 标签**
   - 从 exact-best 改为：
     - `keep_default`
     - `override_to_x`
   - 使用 improvement threshold 或分层 top margin

3. **重建 `routing_override_bootstrap_v17`**
   - 从更大 train pool 抽取
   - 保证 keep_default / override 分布合理
   - 不再包含 synthetic route heuristics

4. **修改 Turn 2 prompt / target / action schema**
   - 新增 `route_time_series`
   - 支持 `decision=keep_default / override`

5. **做 route-override warm-up SFT**
   - 从当前 full checkpoint 继续
   - 用新的 gate 评估

6. **route gate 通过后，重建 `full_stepwise_v17`**
   - Turn 3 使用增强 candidate pool
   - 保持 engineering_refine

7. **做 full refresh SFT**
   - 检查 refine 不退化
   - 检查 final policy 是否优于 default baseline

8. **只有 v17 三重 gate 都通过，才允许进入 RL**

---

# 14. 一句话结论

v16 的失败并不意味着前面修改都白做了。  
恰恰相反，它已经把问题成功缩小到了：

## **exact single-expert routing 这个监督目标，在当前 ETTh1 设置下过强**

因此 v17 的正确动作不是继续硬修 v16，也不是立刻 full v2，  
而是：

## **把 route 从“精确选最优 expert”改成“默认专家 + 置信 override”**

这会是当前最稳、最符合数据实际、且仍尽量对齐论文叙事的下一步。
