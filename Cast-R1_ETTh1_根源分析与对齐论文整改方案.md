# Cast-R1 ETTh1 代码根源分析与对齐论文整改方案

日期：2026-03-25  
适用对象：当前 `eeh1_test-main` 代码仓库 + `ETTh1 代码对齐与坍塌排查报告`

---

## 1. 结论先行

当前 ETTh1 复现出现“模型选择坍塌到单一 expert”的**主根因**，不是某个 expert 天然最优，也不是 RL 单独学坏了，而是：

> **SFT 阶段形成了一个耦合的错误行为先验（behavioral prior）**：
> 模型被共同教成了“按 heuristic 倾向选模 + 后续尽量 KEEP，而不是基于状态自适应路由并在 Turn 3 做可靠修正”的策略。

更具体地说，当前问题是下面四层叠加造成的：

1. **routing prior 偏 heuristic**  
   SFT 的 routing teacher 主要来自 heuristic，而不是 offline best / reference teacher。
2. **runtime 继续泄露 routing 候选模型**  
   Turn 1 在进入 Turn 2 之前就把候选 expert 暗示给模型，进一步放大错误先验。
3. **Turn 3 学成了 KEEP-style 保守策略**  
   一旦 Turn 2 选错，Turn 3 几乎没有真实补救能力。
4. **RL 只是在 exploitation 这个错误 warm start**  
   delayed reward 本身没有错，但在当前 warm start 下，RL 更像在放大已有偏置，而不是重新学会 state-aware routing。

因此，当前现象的本质不是：

- “论文方法不行”；
- “ETTh1 上就应该只选一个模型”；
- “某个 prompt 写得不好”。

而是：

> **论文里的 sequential decision policy，被当前实现落成了 heuristic scaffold + imitation-heavy prior + RL exploitation 的系统。**

---

## 2. 论文主线与当前代码实现的关键差别

### 2.1 论文主线

Cast-R1 的主线是：

1. Turn 1：先做 feature extraction / diagnostic planning；
2. Turn 2：基于 **updated state** 做 adaptive model selection；
3. Turn 3：对 selected forecast 做 reflection / refinement / final answer；
4. 训练上采用 **SFT + multi-turn RL + curriculum RL**；
5. reward 是 **episode-level delayed reward**，但它优化的是整条 sequential workflow，而不是只奖励某一步。

换句话说，论文强调的是：

> **让 agent 在 memory/state 上学会“如何逐步决策”。**

### 2.2 当前代码的实际学习机制

当前代码虽然保留了外形：

- 三阶段；
- feature tools；
- predict_time_series；
- Turn 3 finalization；
- SFT + RL；

但实际形成的学习机制更接近：

> **heuristic 先预路由 -> SFT 教模型复现这套预路由 -> Turn 3 学 KEEP -> RL 继续强化这个先验**

所以问题不在“有没有三阶段”，而在：

> **策略形成机制没有真正对齐论文里的 state-aware learned policy。**

---

## 3. 结合代码和实验结果，问题为什么会出现

---

### 3.1 根因一：diagnostic policy 把“先看证据再选模型”提前变成了“先猜模型”

#### 代码位置

- `recipe/time_series_forecast/diagnostic_policy.py`
- `recipe/time_series_forecast/time_series_forecast_agent_flow.py`
- `recipe/time_series_forecast/prompts.py`

#### 关键代码链

当前链路里：

1. `build_diagnostic_plan(history_values)` 会直接给出：
   - `tool_names`
   - `primary_model`
   - `runner_up_model`
   - `rationale`
2. `time_series_forecast_agent_flow.py` 在 episode 一开始就把它们写进：
   - `self.required_feature_tools`
   - `self.diagnostic_plan_reason`
   - `self.diagnostic_primary_model`
   - `self.diagnostic_runner_up_model`
3. `prompts.py -> build_runtime_user_prompt()` 在 diagnostic stage 中又把这些内容拼进 prompt。

也就是说，当前 agent 不是：

> 先做 feature extraction -> 更新 state -> 再自主 routing

而是更接近：

> heuristic 先给出 primary/runner-up -> LLM 顺着这个候选框架去执行

#### 代码证据

- `diagnostic_policy.py` 中 `_select_primary_model(...)` 和 `_heuristic_model_scores(...)` 直接把历史窗口映射到 `arima / patchtst / itransformer / chronos2`。
- `time_series_forecast_agent_flow.py` 中初始化时：
  - `self.diagnostic_primary_model = diagnostic_plan.primary_model`
  - `self.diagnostic_runner_up_model = diagnostic_plan.runner_up_model`
- `prompts.py` 中 diagnostic prompt 会出现：
  - `Current strongest routing hypothesis: ...`
  - `Planned comparison focus: ... versus ...`

#### 实验印证

实验 A（去掉 runtime 候选模型提示）后：

- `validation_reward_mean` 上升
- `final_answer_accept_ratio` 上升
- `strict_length_match_ratio` 上升
- `orig_mse_mean` 更优

说明：

> **runtime heuristic leakage 是真实存在的，而且会继续放大坍塌。**

#### 本质问题

这一步让论文中的：

- `state -> routing`

退化成：

- `heuristic pre-routing -> prompt hint -> routing imitation`

所以模型更像在“遵从提示”，不是在“根据 state 决策”。

---

### 3.2 根因二：SFT 的 routing teacher 不是在教“最优路由”，而是在教“heuristic 路由”

#### 代码位置

- `recipe/time_series_forecast/build_etth1_sft_dataset.py`

#### 关键代码链

当前 SFT builder 里，核心逻辑是：

- `_select_prediction_model_by_heuristic(history_values)`
- `selected_prediction_model = heuristic result`
- `routing_policy_source = "heuristic_rule_based"`

并且 routing assistant 的内容会围绕这个 selected model 构造 reasoning。

#### 实验印证

你当前结果已经证明：

- `selected_prediction_model` 与 `reference_teacher_model` 的源样本级一致率只有 `0.27`
- step-wise parquet 中：
  - `train_routing_policy_source_distribution = {'heuristic_rule_based': 200}`

这说明 warm start 学到的是：

> **heuristic teacher 的行为先验**

而不是：

> **offline best expert / reference teacher 路由**

#### 为什么这会直接导致坍塌

SFT 在 Cast-R1 中本来就是 strong behavioral prior。  
如果这个 prior 本身已经把“什么模式 -> 选什么模型”教偏了，那么 RL 很难凭 delayed reward 把整个路由逻辑重新纠正回来。

#### 更关键的一点

实验 B-2 已经证明：

> **只把 routing label 换成 reference teacher，不足以修复问题。**

反而会在当前配方下把更多样本推向 `arima`，并拉坏 fixed-set `selected_forecast_orig_mse_mean`。

这说明根因已经不是单独的 routing label，而是：

> **更广义的 SFT behavioral prior。**

它还包括：

- routing reasoning 文本模式；
- non-routing stage supervision；
- Turn 3 KEEP-style 目标；
- 不同 stage 的重复采样和配比方式。

---

### 3.3 根因三：Turn 3 被学成了“尽量别动”

#### 代码位置

- `recipe/time_series_forecast/prompts.py`
- `recipe/time_series_forecast/build_etth1_sft_dataset.py`
- `recipe/time_series_forecast/time_series_forecast_agent_flow.py`

#### 当前协议特征

在 refinement stage，prompt 明确要求：

- 只能 `KEEP` 或 `LOCAL_REFINE`
- `If unsure, choose KEEP.`
- `LOCAL_REFINE` 只允许很有限的局部修改
- 不允许再调用工具

#### 实验印证

实验 C 表明：

- 只放宽 Turn 3 wording，几乎没行为变化；
- `refinement_changed_ratio` 仍然为 0；
- `final_vs_selected_mse_mean` 仍然为 0；
- selected expert 完全不变。

这说明：

> Turn 3 的 KEEP 偏置主要不是 runtime wording，而是已经固化在 policy prior 里。

#### 进一步证据

实验 D 中，严格不改 paper protocol、只增强 Turn 3 stronger-SFT 后：

- 第一次出现真实 refinement；
- 但 routing 同时坍到 `patchtst=32/32`；
- 格式稳定性明显变差。

这说明：

> Turn 3 并不是独立模块，它和 routing policy 在同一套 SFT 行为模板里耦合学习。

所以当前问题不是“Turn 3 正例不够”这么简单，而是：

> **routing prior 和 refinement prior 在整条 SFT 轨迹里一起学歪了。**

---

### 3.4 根因四：stage-level SFT 是耦合复制，而不是解耦训练

#### 当前实现的问题

当前 SFT 数据构造和增强方式，主要还是以“样本整条轨迹”为单位。

这会导致一个严重副作用：

- 当你想增强 Turn 3 时，如果复制的是整条轨迹；
- routing / diagnostic 行也会被一起放大；
- 最后不是只增强 refine，而是把 route prior 也一起带偏。

实验 D 已经直接证明了这一点：

> 增强 Turn 3 后，首次出现非零 refinement，但同时 routing 坍到了 `patchtst=32/32`。

#### 本质问题

你现在训练的不是：

- routing skill
- refinement skill

而是训练了一种整体行为模板：

> “这类样本 -> 选某个 expert -> 最后尽量 KEEP”

这就是为什么现在最合理的判断是：

> **SFT 阶段学歪的不是某一个 stage，而是 stage-coupled policy prior。**

---

### 3.5 根因五：RL 不是主犯，但会把错误 warm start 固化成稳定坍塌

#### 代码位置

- `recipe/time_series_forecast/time_series_forecast_agent_flow.py`
- `recipe/time_series_forecast/reward.py`

#### 代码事实

- `_compute_intermediate_reward(...)` 直接返回 `0.0`
- reward 主要在 final step 通过 `compute_score(...)` 统一计算
- Turn 2 只允许预测一次
- Turn 3 不允许再调用工具

#### 为什么这本身并不违背论文

论文就是 delayed reward。  
所以“中间奖励为 0”本身不是偏离论文。

#### 真正的问题是什么

当前 warm start 已经是错的：

- routing 倾向 heuristic；
- Turn 3 倾向 KEEP；
- runtime 还继续暗示候选模型；

在这种情况下，RL 更容易学到：

> **固定选择某个整体更稳的 expert，然后尽量别改。**

因此：

- RL 不是根因；
- 但 RL 会把错误的 prior exploitation 成稳定的坍塌行为。

---

## 4. 为什么当前结果不能解释成“环境天然单峰最优”

这点非常关键。

从你的 teacher-eval 统计看：

- train best-model 分布不是单峰；
- val best-model 分布不是单峰；
- test best-model 分布也不是单峰；
- 四个 expert 都没有失败；
- debug 聚合中：
  - `prediction_model_defaulted_ratio = 0.0`
  - `prediction_tool_error_count = 0`

这说明：

1. 不是某个 expert 根本不可用；
2. 不是某个 expert 在所有 split 上天然绝对统治；
3. 当前塌到 `arima`、`patchtst`、甚至 `itransformer=32/32` 的不稳定现象，更像：
   - checkpoint 偏置
   - prompt 偏置
   - warm-start 偏置
   - runtime leak 偏置
   被后续训练放大。

所以，当前坍塌是**训练动力学问题**，不是**环境唯一最优**。

---

## 5. 对齐论文后，问题的真正根源应该怎么表述

推荐采用下面这句作为最终归因：

> **当前 ETTh1 复现中的坍塌，根源在于 SFT 阶段形成了耦合的错误行为先验：模型被共同塑造成“按 heuristic 倾向进行单一路由选择，并在后续阶段偏向保守接受而非有效修正”的策略；runtime heuristic leakage 与 RL exploitation 主要是在放大这一先验。**

这句话比“routing label 错了”更完整，也比“Turn 3 不会改”更准确。

---

## 6. 如何根治：必须按论文主线“拆耦合 + 重建 warm start + 再进 RL”

根治这个问题，不能再用“单点 patch”思路。  
需要按下面顺序执行。

---

### 6.1 第一层修复：彻底去掉 runtime 的模型候选泄露

#### 目标

让 Turn 1 只负责：

- 决定要看哪些 feature tools；
- 不再提前把 expert 候选注入 prompt。

#### 修改建议

**保留：**

- `required_feature_tools`
- `diagnostic_plan_reason` 中与 feature evidence 相关的内容

**删除 / 默认禁用：**

- `diagnostic_primary_model`
- `diagnostic_runner_up_model`
- `looks strongest`
- `distinguish X from Y`
- 任何显式 expert 候选暗示

#### 代码文件

- `recipe/time_series_forecast/prompts.py`
- `recipe/time_series_forecast/time_series_forecast_agent_flow.py`
- `recipe/time_series_forecast/diagnostic_policy.py`

#### 说明

你已经做了 Patch A，这一步应当直接保留为默认正式链路，不再回滚。

#### 作用

把当前：

- heuristic pre-routing -> LLM 跟随

改回更接近论文的：

- evidence gathering -> state update -> routing

---

### 6.2 第二层修复：SFT 改成 stage-decoupled，而不是 trajectory-coupled

这是根治的核心。

#### 关键原则

不要再“整条轨迹一起复制、一起强化”。  
必须把：

- routing skill
- refinement skill

拆开训练。

---

#### 6.2.1 Routing-only SFT

##### 目标

让 Turn 2 真正学会：

> 根据 analysis state 选模型

而不是根据 heuristic 模板复读。

##### 修改建议

在 `build_etth1_sft_dataset.py` 中新增 / 保留一条独立 builder 分支：

- 只构造 `turn_stage == routing` 的训练样本；
- routing label 用：
  - `reference_teacher_model`
  - 或 `offline_best_model`
- routing reasoning 文本去 expert 模板化：
  - 不要写“patchtst for motifs / arima for autocorrelation”这类句子；
  - 只写证据特征，如：
    - change points 多
    - local peaks 稳定
    - residual excursions 高
- 可以提高 routing rows 的 repeat factor 或 loss weight；
- 但**不要复制该样本的 diagnostic/refinement rows**。

##### 为什么这样做

实验 B-2 已经说明：

- 只改 routing label source 不够；
- reasoning 模板、其它 stage 监督也会继续把模型拉回旧 prior。

所以 routing-only SFT 必须做到：

> **只强化 routing skill，不让其它 stage 的错误 prior 一起被重复。**

---

#### 6.2.2 Refinement-only SFT

##### 目标

让 Turn 3 学会：

> 在 selected forecast 基础上做少量但真实有效的修正

而不是默认 KEEP。

##### 修改建议

在 `build_etth1_sft_dataset.py` 中单独处理 `turn_stage == refinement`：

- 保留 `validated_keep` 与 `local_refine` 两类 target；
- 但降低 `validated_keep` 权重；
- 提高 `local_refine` rows 的 repeat factor / loss weight；
- **只在 refinement rows 内重平衡**；
- 不要通过 source-level oversampling 复制整条轨迹；
- 不要让 routing / diagnostic rows 因此被一起增采样。

##### 为什么这样做

实验 D 已经清楚说明：

- 如果增强 Turn 3 的方式是整条轨迹一起放大；
- routing prior 也会被一起带偏；
- 结果就是 refine 只改善了一点点，但 routing 直接塌到 `patchtst=32/32`。

所以正确修法是：

> **只增强 refinement rows，不连带复制其它 stage。**

---

### 6.3 第三层修复：Turn 3 的目标从“谨慎不动”改成“证据充分时敢改”

#### 当前问题

Turn 3 现在的核心目标太偏向：

- 合法复制；
- 格式稳定；
- 尽量不改。

这会让它退化成纯 post-copy stage。

#### 正确方向

不是放飞 Turn 3，而是让它：

- 保持严格格式；
- 但在确有必要时，执行真实且有限的修正。

#### 建议做法

1. 训练数据中增加高质量 `LOCAL_REFINE` 正例；
2. `LOCAL_REFINE` 提高训练权重；
3. `validated_keep` 不删除，但降权；
4. 保持 strict protocol，不要改成自由生成；
5. 后续在 RL 或 reward 里可加一个很弱的 refine-improvement bonus：
   - 当 `final_vs_selected_mse < 0` 时给轻微奖励；
   - 当 refinement 使误差恶化时给轻微惩罚。

#### 注意

不要再把主要精力放在“改 Turn 3 prompt wording”。  
实验 C 已经证明，这条路几乎没用。

---

### 6.4 第四层修复：重建 warm start，不要直接承接旧链路

#### 当前风险

如果继续沿用旧的：

- 旧 prompt
- 旧 routing teacher
- 旧 reward 口径
- 旧 step-wise parquet
- 旧 warm start checkpoint

那么旧 prior 会继续被继承。

#### 正确做法

重新走一遍 clean warm start：

1. 保留 hint-drop runtime；
2. 使用新的 stage-decoupled SFT builder；
3. 重建新的 step-wise parquet；
4. 训练新的 SFT warm start；
5. 在固定验证集上先检查 warm start 是否仍然坍塌；
6. 只有确认 warm start 已明显改善后，再接 RL。

#### 为什么这一步是必须的

RL 不该被当成“修复旧 prior 的工具”。  
在论文里，RL 的角色是：

> **放大正确的 behavioral prior，而不是修复错误初始化。**

---

### 6.5 第五层修复：RL 只做“放大正确先验”，不要一上来大改 reward

#### 当前判断

此阶段不建议立刻大改 reward 主体。

原因：

- delayed reward 本身符合论文；
- 现在主要根因仍在 SFT；
- 在错误 warm start 上调 reward，容易越调越乱。

#### 正确顺序

1. 先修正 SFT prior；
2. 再用固定验证子集观察：
   - 路由是否还塌；
   - Turn 3 是否仍然几乎不改；
3. 只有 warm start 改善后，才考虑加入很轻量的 RL shaping：
   - route entropy / diversity 的弱约束；
   - refine improvement bonus 的弱约束。

#### 注意

不要把 RL 改成“强制平均使用 expert”。  
那会偏离论文，也会破坏真正的 adaptive routing。

---

## 7. 推荐的代码级修改方案

---

### 7.1 `recipe/time_series_forecast/prompts.py`

#### 保留

- Patch A：默认去掉候选模型提示

#### 建议进一步修改

1. diagnostic stage：
   - 不再显示 `diagnostic_primary_model / diagnostic_runner_up_model`
   - 只显示：
     - 该窗口需要哪些 feature tools
     - 分析目标是“建立足够 routing evidence”

2. routing stage：
   - 删除或弱化强 expert 模板句子：
     - `patchtst -> motifs`
     - `arima -> autocorrelation`
     - `chronos2 -> irregular`
     - `itransformer -> drift`
   - 改成更中性的 routing instruction：
     - “Use the analysis history to choose the expert whose inductive bias best matches the observed evidence.”

3. refinement stage：
   - 保留 strict format 和 KEEP/LOCAL_REFINE 二选一；
   - 但不要再把 prompt 改动作为主修复路径；
   - 这里只做最小必要改动，避免继续引入格式噪声。

---

### 7.2 `recipe/time_series_forecast/diagnostic_policy.py`

#### 建议修改

把 `DiagnosticPlan` 的职责缩小：

- 保留 `tool_names`
- 保留 feature-oriented rationale
- 弱化或删除 `primary_model / runner_up_model`

更理想的做法：

- 可以内部继续算 heuristic scores，用于决定建议调用哪些 feature tools；
- 但这些 score 不应直接暴露给 runtime prompt。

#### 目标

让 diagnostic policy 只做：

> **证据采集规划**

而不做：

> **显式 expert 候选预路由**

---

### 7.3 `recipe/time_series_forecast/time_series_forecast_agent_flow.py`

#### 建议修改

1. 初始化阶段：
   - 保留 `required_feature_tools`
   - 不再把 `diagnostic_primary_model` 和 `diagnostic_runner_up_model` 用作 prompt 输入

2. `_build_user_prompt()`：
   - 调用 `build_runtime_user_prompt()` 时，不再传 expert 候选字段；
   - 或默认传空字符串。

3. Turn 3 阶段：
   - 保留当前 canonical forecast 逻辑；
   - 不在这里做大改；
   - 真正的修复重心放在训练数据和目标上。

---

### 7.4 `recipe/time_series_forecast/build_etth1_sft_dataset.py`

这是主战场。

#### 必改方向

##### A. 新增 routing-only builder 模式

建议新增参数，例如：

- `--sft-stage-mode {full,routing_only,refinement_only}`

在 `routing_only` 模式下：

- 只生成 `turn_stage == routing` 行；
- label = `reference_teacher_model` / `offline_best_model`；
- routing reasoning 文本去 expert 模板化；
- 允许 routing row repeat / weight 提升；
- 不复制 diagnostic/refinement 行。

##### B. 新增 refinement-only weighting 模式

在 `refinement_only` 模式下：

- 只对 `turn_stage == refinement` 行做重平衡；
- `validated_keep` 降权；
- `local_refine` 升权；
- 不通过 source-level oversampling 复制整条轨迹；
- 不改变 routing/diagnostic 行分布。

##### C. 拆开 metadata 统计

额外输出：

- `routing_only_selected_model_distribution`
- `refinement_target_distribution`
- `turn_stage_loss_weight_summary`
- `source_sample_coverage_by_stage`

方便判断 stage-local 调整是否干净。

##### D. routing reasoning 改写

不要再写：

- “I choose patchtst because ...”
- “arima is strongest ...”

改成：

- “The evidence suggests strong local repetition with stable spacing.”
- “The evidence suggests abrupt structural changes and persistent drift.”

由 label 决定监督目标，但 reasoning 文本尽量保持 feature-first。

---

## 8. 建议的实验顺序（最小闭环）

下面这个顺序最稳。

### 实验 1：hint-drop 基线固定

目的：

- 固定一个相对干净的 runtime 版本。

要求：

- 后续所有实验都复用该版本；
- 不再回滚到旧 prompt。

---

### 实验 2：routing-only SFT

目的：

- 只测 routing prior 是否能被拉回 state-aware 路线。

配置：

- reference teacher label
- 去 expert 模板化 reasoning
- 只训练 routing rows
- 固定验证子集比较

重点指标：

- `selected_model_distribution`
- `selected_forecast_orig_mse_mean`
- route switch 是否仍大面积流向单一 expert

---

### 实验 3：refinement-only SFT

目的：

- 只测 Turn 3 能否学会少量真实修正，而不拖偏 routing。

配置：

- 只训练 refinement rows
- `local_refine` 升权
- `validated_keep` 降权
- 不复制 routing/diagnostic rows

重点指标：

- `refinement_changed_ratio`
- `refinement_improved_ratio`
- `final_vs_selected_mse_mean`
- `selected_model_distribution`
- `final_answer_accept_ratio`

---

### 实验 4：merge / joint warm start

目的：

- 将修正后的 routing skill 与 refinement skill 合并
- 检查新 warm start 是否仍单峰坍塌

可选方式：

- 先训 routing-only，再训 refinement-only
- 或多任务 joint，但 stage-local weighting 明确区分

只有这一步稳定后，才进入 RL。

---

### 实验 5：新 warm start + RL

目的：

- 让 RL 在正确 prior 上继续优化

要求：

- 继续使用固定验证子集
- 暂不大改 reward 主体
- 只在必要时添加很弱的 route/refine shaping

---

## 9. 判断“问题是否被根治”的标准

不要只看 reward。  
至少同时看下面四类指标。

### 9.1 routing 是否仍然极端单峰

看：

- `selected_model_distribution`

目标：

- 不要求四个 expert 完全均匀；
- 但不应长期出现：
  - `patchtst=32/32`
  - `arima=23, patchtst=9` 这种极端单峰分布。

### 9.2 selected forecast 质量是否回升

看：

- `selected_forecast_orig_mse_mean`
- `orig_mse_mean`

目标：

- 不要出现“分布看起来多样了，但 forecast 更差”的伪改善。

### 9.3 Turn 3 是否开始产生真实修正

看：

- `refinement_changed_ratio`
- `refinement_improved_ratio`
- `final_vs_selected_mse_mean`

目标：

- 从近似 0 提升到“小而真实的有效修正”；
- 同时不要严重破坏格式稳定性。

### 9.4 格式与协议是否仍稳定

看：

- `final_answer_accept_ratio`
- `strict_length_match_ratio`
- `missing_answer_close_tag`
- `invalid_answer_shape`

目标：

- 修复 routing/refinement 的同时，不能把 protocol 打崩。

---

## 10. 最终整改路线图

建议把后续整改压成一句执行路线：

> **去掉 runtime 候选模型泄露 -> 将 SFT 改成 routing/refinement 解耦训练 -> 重建 clean warm start -> 再在其上进行 RL。**

如果再压成一句最短版：

> **当前问题根子在 SFT：模型被共同教成了“按 heuristic 选模 + 尽量 KEEP”的策略，而不是按 state 做 adaptive routing 再做 reliable refinement；根治必须从 SFT 解耦开始，而不是继续靠 prompt patch 或 RL 救火。**

---

## 11. 可直接交给 Codex 的执行清单

1. 固化 Patch A，默认关闭所有 diagnostic model hints。  
2. 在 `build_etth1_sft_dataset.py` 中新增 `routing_only` / `refinement_only` builder 模式。  
3. 将 routing reasoning 文本改成 feature-first，不出现显式 expert 模板句。  
4. routing-only builder 使用 `reference_teacher_model/offline_best_model` 作为监督标签。  
5. refinement-only builder 只对 `local_refine` 行加权，不复制整条源轨迹。  
6. 训练两个独立 warm start：  
   - `sft_route_only_refteacher_nomodelhint`  
   - `sft_turn3_refine_stage_local`  
7. 在固定验证集 `val_fixed32_refcmp_20260325.jsonl` 上比较：  
   - `selected_model_distribution`  
   - `selected_forecast_orig_mse_mean`  
   - `refinement_changed_ratio`  
   - `refinement_improved_ratio`  
   - `final_answer_accept_ratio`  
8. 只有当 warm start 不再显著坍塌后，才继续 RL。  
9. RL 第一轮不大改 reward；只在必要时增加轻量 route/refine shaping。  
10. 所有实验结果统一汇总到新的审计报告，避免与旧链路混用。

---

## 12. 一句话总结

**你的代码现在的问题，不是“没做出论文的三阶段”，而是“把论文里应当由 state 学出来的 sequential decision，提前写成了 heuristic scaffold，并在 SFT 里把 routing 与 KEEP 风格一起学死了”。真正的修复方式不是继续补丁式调 prompt，而是把 SFT 解耦、重建 warm start，再让 RL 去优化正确的先验。**
