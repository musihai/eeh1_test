# Route Proposal v18 修改 + 测试 + 验收方案

## 1. 当前结论

截至 v17，问题已经可以明确收敛为：

- **四个 expert 本身没有明显损坏**
  - expert 权重加载正常
  - 本地重跑 teacher-eval 与历史结果基本一致
  - full teacher-eval 上 4 个 expert 都会赢，不存在“只有一个 expert 真有效”
- **route 协议本身没有坏**
  - `route_time_series` 的解析是通的
  - v17 的 tool-call 有效率是 `100%`
- **真正坏的是 route supervision 和 route task formulation**
  - v16：exact 4-way oracle route 长期塌成单 expert
  - v17：default + override 改写后，塌成 always `keep_default`

因此，v18 不应继续在 v16/v17 的 route 定义上硬修，而应直接重写 Turn 2 的任务定义。

---

## 2. v16 / v17 为什么都失败

### 2.1 v16 的问题：exact 4-way route 过强

v16 让 Turn 2 学习：

- 从 4 个 fixed experts 中
- 精确命中 oracle best expert

但当前 ETTh1 设置下：

- top-1 / top-2 常常接近
- 仅靠 Turn 1 的诊断 state 很难稳定区分
- route label 本身带有高噪声

因此模型自然收敛到最安全的单 expert。

### 2.2 v17 的问题：标签进一步制造了“保守最优”

v17 引入了：

- `default_expert = itransformer`
- `keep_default`
- `override_to_x`

这一步方向是对的，但当前 builder 的实现仍有关键问题：

1. 对每个非 default expert，只保留前一部分样本做 override  
2. 其余哪怕 `best_model != default`，也重新打回 `keep_default`

结果是：

- 数据中存在大量“明明 default 明显更差，但标签仍是 keep_default”的样本
- prompt 还显式鼓励 `keep_default`
- 在 `70/30` 类别分布下，always-keep 成为最安全解

所以 v17 的失败不是“新问题”，而是 v16 同一个 route 偏置换了新的标签空间继续出现。

---

## 3. 论文为什么能做，我们这里为什么不行

### 3.1 论文里的 route 不是孤立分类器

根据 `paper.pdf` 第 3.4/3.5 节，论文里的 workflow 是：

1. Turn 1 获取诊断证据
2. Turn 2 选择 forecasting model 并调用
3. 该 model 的 forecast 会被写回 state
4. Turn 3 基于 updated state 做 reasoning / refinement

也就是说，论文里的 route：

- 是完整 stateful workflow 的一部分
- 不是单独抽出来做 route-only 分类
- 也不是最终不可撤销的承诺

### 3.2 我们当前 route-only warm-up 不是论文正式配方

仓库已有审计也明确写过：

- `routing_only / refinement_only` 是研究工具
- 不是论文声明的正式训练配方

因此，我们当前做法实际是在训练：

- 一个依赖隐藏 oracle loss 生成的 route label
- 但 prompt 里并没有暴露足够的 candidate-level证据
- 又要求模型在 Turn 2 提前做最终承诺

这与论文真正优化的学习问题并不相同。

### 3.3 别的成功方法通常怎么做

更稳的做法通常有三类：

1. **软门控 / 代价敏感 gating**
   - 不把接近 tie 的样本硬打成唯一正确标签
2. **proposal + rerank**
   - 先提 shortlist，再看 candidate forecast 做最终选择
3. **只在高置信 easy case 上训练 route**
   - 模糊样本直接丢弃或交给后续步骤

v18 将采用第 2 和第 3 类的组合。

---

## 4. v18 的核心设计

v18 的核心不是继续修“最终 route 分类器”，而是把 Turn 2 重新定义为：

## **候选提议（proposal），而不是最终承诺（final commitment）**

Turn 2 的职责改成：

- `keep_default`
- 或 `propose_alt(model_name=x)`

Turn 3 的职责改成：

- 在 `default path` 与 `alt path` 之间做最终选择
- 并在必要时对所选路径继续做 local refine

这样改后：

- Turn 2 不再承担“精确选中最终最优 expert”的全部责任
- Turn 3 才真正承担 final policy 的职责
- 更贴近论文里“route -> state update -> refine”的结构

---

## 5. v18 的两条硬原则

### 5.1 Route warm-up 只允许无矛盾标签

任何满足以下条件的样本：

- `best_model != default_expert`
- 且相对改进明显

都**禁止**再被标成 `keep_default`。

### 5.2 模糊样本不再参与 route warm-up

对于：

- `best_model != default_expert`
- 但改进不够明显
- 或 top-1 / top-2 margin 太小

这类样本不再强行打 hard route label，统一放入：

- `ambiguous`

它们不参加 Turn 2 route warm-up，只在 full-stepwise 阶段交给 Turn 3 处理。

---

## 6. Phase A：重建 route supervision（从 hard route 改为 triage route）

### A.1 固定 default expert

v18 第一轮仍固定：

- `default_expert = itransformer`

原因：

- 当前 collapse 到 `itransformer` 时 regret 最低
- 作为 default 更稳
- 但 v18 不再把它当“保守标签的大桶”

### A.2 计算基础字段

对每个窗口，保留：

- `default_error`
- `best_error`
- `best_model`
- `second_best_error`
- `second_best_model`
- `improvement_vs_default = default_error - best_error`
- `improvement_vs_default_rel = (default_error - best_error) / default_error`
- `route_margin_abs = second_best_error - best_error`
- `route_margin_rel = (second_best_error - best_error) / second_best_error`
- `default_in_top2`

### A.3 三段式标签

对每个窗口定义三类桶：

1. `must_keep`
2. `must_override_to_x`
3. `ambiguous`

### A.4 推荐阈值

第一轮建议使用：

- `tau_keep = 0.05`
- `tau_margin = 0.08`
- `tau_override_model(x) = max(q80_train_x, 0.35)`

其中：

- `q80_train_x` 表示在 train 中、`best_model == x` 的窗口里，
  `improvement_vs_default_rel` 的 80 分位数

### A.5 具体打标规则

#### must_keep

满足任一条件：

- `best_model == default_expert`
- `improvement_vs_default_rel <= tau_keep`

#### must_override_to_x

满足全部条件：

- `best_model = x != default_expert`
- `improvement_vs_default_rel >= tau_override_model(x)`
- `route_margin_rel >= tau_margin` 或 `default_expert` 不在 top-2

#### ambiguous

其余全部归入：

- `ambiguous`

### A.6 v18 的关键约束

必须保证：

- `must_keep` 中不再出现“明显该 override”的窗口
- `must_override_to_x` 只保留高置信 easy case
- `ambiguous` 明确从 route warm-up 中剔除

---

## 7. Phase B：重建 route bootstrap 数据集

### B.1 新数据集命名建议

- `routing_proposal_bootstrap_v18`

### B.2 数据集拆分方式

v18 不再只保留一个 val，而是拆成两套：

1. `val_natural`
   - 保持真实 keep / override 分布
   - 用于看 `delta_vs_default_mean`
   - 用于看自然场景下是否塌缩
2. `val_balanced`
   - 对 `must_keep` / `must_override` 做平衡采样
   - 用于看 route classification 能力

### B.3 推荐规模

- train pool：`2000 ~ 4000`
- route warm-up train：`768 ~ 1024`
- `val_natural`：`>= 192`
- `val_balanced`：`>= 192`

### B.4 训练集分布要求

route warm-up train 不再用 `70/30 keep/override`。

推荐：

- `keep : override = 50 : 50`

override 内部再按 model 平衡：

- `patchtst`
- `arima`
- `chronos2`

### B.5 数据集必须额外保留的字段

- `route_bucket = must_keep / must_override / ambiguous`
- `route_target_model`
- `default_expert`
- `route_default_path_id`
- `route_alt_path_id`
- `default_in_top2`
- `route_margin_rel`
- `improvement_vs_default_rel`

### B.6 数据集验收

必须满足：

- `must_keep` 与 `must_override` 无 source overlap
- train / val_natural / val_balanced / test 无 source overlap
- `must_keep` 中不存在明显高改进非 default 样本
- 每个 override model 在 train 至少 `>= 64`
- 每个 override model 在 `val_balanced` 至少 `>= 16`

---

## 8. Phase C：修改 Turn 2 的 action schema 与 prompt

### C.1 保留 `route_time_series`

action 名仍可保留：

- `route_time_series`

但语义改成：

- `keep_default`
- `propose_override`

### C.2 新 target 形式

#### keep_default

```xml
<think>Choose the routing action supported by the current state.</think>
<tool_call>
{"name":"route_time_series","arguments":{"decision":"keep_default"}}
</tool_call>
```

#### propose override

```xml
<think>Choose the routing action supported by the current state.</think>
<tool_call>
{"name":"route_time_series","arguments":{"decision":"override","model_name":"patchtst"}}
</tool_call>
```

### C.3 prompt 修改原则

必须做到：

- 不再使用保守偏置措辞
- 不再写 “only when clearly supports” 这种默认 discouraging override 的话
- `keep_default` 与 `override` 在语言上对称
- examples 一正一反，语气完全平衡

### C.4 route target 中禁止出现的内容

- `RouteDecision(... reason_codes=...)`
- `RouteSummary`
- synthetic route heuristics
- 带标签倾向的解释模板

---

## 9. Phase D：修改 Turn 2 的运行时语义

### D.1 Turn 2 不再是 final route

Turn 2 输出后：

- 系统**总是**保留 `default_expert__keep`
- 如果 Turn 2 提议了 `override_x`
  - 再额外 materialize 一条 `x__keep`

也就是说，Turn 2 的结果是：

- 决定是否把 `alt path` 引入 Turn 3

而不是：

- 决定最终只能沿哪条路径走

### D.2 这一步为什么重要

它直接修复当前最伤性能的结构问题：

- Turn 2 不必过早做不可逆承诺
- Turn 3 重新承担 final selection 的职责
- route 错了不等于 final 一定错

这比 v16/v17 都更贴近论文里：

- route 后 state 被更新
- 再进入 refinement / final decision

---

## 10. Phase E：增强 Turn 3 candidate pool

### E.1 v18 的最小 candidate pool

Turn 3 至少应包含：

- `default_expert__keep`
- `default_expert` 的 local refine 候选
- 如果 Turn 2 提议了 override：
  - `override_expert__keep`
  - `override_expert` 的 local refine 候选

### E.2 第一轮不要扩成全量 4-expert rerank

v18 仍保留 3-turn 语义，不直接跳 full v2。

因此第一轮只保留：

- default 路径
- proposal alt 路径

### E.3 Turn 3 仍保持 engineering_refine

不退回 paper-style 全 forecast 文本输出。

继续使用：

- `<think>...</think><answer>decision=...</answer>`

但 decision 空间要允许从：

- `default path`
- `alt path`

之间做最终选择。

---

## 11. Phase F：Route Proposal Warm-up SFT

### F.1 数据集

使用：

- `routing_proposal_bootstrap_v18`

其中：

- train 使用平衡集
- probe 同时看 `val_natural` 和 `val_balanced`

### F.2 起点

第一轮仍建议从当前最好的 full SFT checkpoint 继续：

- `global_step_142/huggingface`

### F.3 推荐参数

- learning rate：`5e-6`
- epoch：`0.5 ~ 1.0`
- 仅训练 routing / Turn 2
- 每 `8` step 做一次 probe
- 任意一次 probe 出现单动作塌缩就早停

### F.4 早停条件

以下任一成立就停：

- `keep_default_share > 0.85`
- 或某个 override model share `> 0.85`
- 或 `delta_vs_default_mean >= 0`
- 或 `override_f1` 连续 2 次 probe 不提升

---

## 12. Phase G：Full Stepwise v18 Refresh

### G.1 新数据集命名建议

- `full_stepwise_v18`

### G.2 full_stepwise_v18 必须满足

1. Turn 2 使用 proposal-style route
2. Turn 3 同时拥有 default / alt 路径候选
3. Turn 3 保持 engineering_refine
4. synthetic route heuristics 彻底清除
5. ambiguous 样本只进入 full-stepwise，不进入 route warm-up

### G.3 训练顺序

1. 先 route proposal warm-up
2. Gate 1 通过后，再做 full refresh SFT
3. 只允许短 refresh：
   - epoch `0.2 ~ 0.5`

---

## 13. 测试方案（v18）

### Test A：Data Audit

必须统计：

- `must_keep / must_override / ambiguous` 分布
- 每个 override model 的数量
- `must_keep` 中高改进非 default 样本数量
- split overlap

#### 新增必须指标

- `contradictory_keep_count`
  - 定义：`route_label=keep_default` 且 `best_model != default_expert`
  - 且 `improvement_vs_default_rel >= tau_override_model(best_model)`

v18 要求：

- `contradictory_keep_count = 0`

### Test B：Route Proposal Probe

route probe 分两套看：

#### B.1 `val_balanced`

看 route classification 能力：

- `keep_vs_override_f1`
- `override_precision`
- `override_recall`
- `override_f1`
- `override_subset_exact_agreement`
- `override_subset_top2_agreement`

#### B.2 `val_natural`

看自然分布上的真实收益：

- `delta_vs_default_mean`
- `mean_route_regret`
- `keep_default_share`
- `override_share`
- `requested_model_distribution`

### Test C：Full Stepwise Probe

必须新增 3 个指标：

- `final_vs_default_regret_mean`
- `final_vs_route_regret_mean`
- `route_error_turn3_rescue_rate`

其中：

- `route_error_turn3_rescue_rate`
  - 表示 Turn 2 proposal 不理想时，Turn 3 仍成功把 final 拉回到优于 default 的比例

### Test D：Protocol / Refine Probe

必须保证：

- `strict_ok_rate = 1.0`
- `overall_changed_ratio` 不退化
- `local_refine` 质量不显著下降

---

## 14. 验收标准（v18）

## Gate 0：Data Gate

满足以下条件才允许进入 route warm-up：

- `contradictory_keep_count = 0`
- train keep/override 比例在 `[45%, 55%]`
- 每个 override model 在 train `>= 64`
- 每个 override model 在 `val_balanced >= 16`
- 无 split overlap

## Gate 1：Route Proposal Gate

### 在 `val_balanced` 上必须满足

- `keep_vs_override_f1 >= 70%`
- `override_precision >= 70%`
- `override_recall >= 60%`
- `override_f1 >= 65%`
- `override_subset_exact_agreement >= 60%`
- `override_subset_top2_agreement >= 85%`

### 在 `val_natural` 上必须满足

- `delta_vs_default_mean < 0`
- `keep_default_share <= 85%`
- 不允许 collapse 到单一 override model

## Gate 2：Full Stepwise Gate

必须同时满足：

- `strict_ok_rate = 1.0`
- `overall_changed_ratio in [5%, 40%]`
- `orig_mse_mean < default_expert_mean_mse`
- `final_vs_default_regret_mean < 0`
- `final_vs_route_regret_mean <= 0`
- `route_error_turn3_rescue_rate > 0`

## Gate 3：进入 RL 的前置条件

只有同时满足：

1. Gate 0 通过
2. Gate 1 通过
3. Gate 2 通过
4. protocol probe 通过

才允许进入 RL。

否则：

- **禁止进入 RL**

---

## 15. 执行顺序

1. 固定 `default_expert = itransformer`
2. 重写 route builder，生成 `must_keep / must_override / ambiguous`
3. 构建 `routing_proposal_bootstrap_v18`
4. 先过 Gate 0
5. 做 route proposal warm-up SFT
6. 先过 Gate 1
7. 再构建 `full_stepwise_v18`
8. 做 short full refresh SFT
9. 过 Gate 2
10. 只有三重 gate 都通过，才允许进 RL

---

## 16. 如果 v18 仍失败

如果 v18 之后仍出现：

- `delta_vs_default_mean >= 0`
- balanced route probe 仍塌缩
- `route_error_turn3_rescue_rate` 很低
- final 仍不能稳定优于 default baseline

则说明：

## 当前 ETTh1 设置下，Turn 2 仍不适合学习明确 route decision

那时应直接进入更大的结构重构：

- Turn 2 不再学习 route
- Turn 2 只做 proposal / shortlist generation
- Turn 3 直接做 candidate rerank + final selection

也就是转向真正的 full v2。

---

## 17. 一句话结论

v18 的核心不是继续修 v16/v17 的 route 标签，而是：

## **把 Turn 2 从“最终路由分类器”改成“高置信候选提议器”，并把模糊样本交还给 Turn 3 做最终选择。**

这一步既保留了论文的 3-turn 叙事，也正面修复了当前最致命的问题：

- route 标签矛盾
- route 标签不可观测
- safe majority action 导致的单动作塌缩
