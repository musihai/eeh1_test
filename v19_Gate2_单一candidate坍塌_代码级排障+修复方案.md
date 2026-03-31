# v19_Gate2_单一candidate坍塌_代码级排障+修复方案

## 1. 适用范围

本文档只针对当前 **v19 Gate 2 失败** 这一类问题，聚焦解决：

> Turn 3 final selector 总是输出同一个 `candidate_id`（当前表现为始终选择 `patchtst__keep`）

当前已知现象：

- `protocol_ok_rate = 1.0`
- `materialize_ok_rate = 1.0`
- Turn 3 训练链路已经修通
- SFT 不再是 `loss_mask = 0` / `NaN`
- 但行为上仍然坍塌为单一候选：
  - `selected_candidate_distribution = {'patchtst__keep': 64}`
  - `single_candidate_max_share = 1.0`
- risky 子集上有改善：
  - `risky_final_vs_default_mean = -0.3576`
- 但 `default_ok` 子集被过度 override：
  - `default_ok final_vs_default_mean = +0.4502`

因此这次问题已经不是：

- expert 权重坏了
- parser 坏了
- materialization 坏了
- Turn 3 完全没训练到

而是：

## 一个“训练上可行、行为上坍塌”的 final selection 问题

---

## 2. 当前问题的最可能根因

本轮 Gate 2 失败，我建议优先按以下 4 个根因排查：

### 根因 1：有效训练步数太少
当前 warm-up checkpoint：

- `global_step_16`

这通常只够模型学会：

- 输出合法 action schema
- 记住一个高频/最安全候选

但不足以学会：

- 细粒度地区分多个 candidate

### 根因 2：Turn 3 监督仍然太薄
当前 target 的真正区分信息可能仍然主要集中在最后几个 token：

- `candidate_id=patchtst__keep`
- `candidate_id=itransformer__keep`
- ...

如果 `<think>` 区分度不够，模型容易学成：

- “固定输出一个默认答案字符串”

### 根因 3：candidate 顺序/展示方式存在强偏置
需要怀疑：

- `patchtst__keep` 是否总排第一
- 或其 summary/head/tail preview 是否最短、最稳定、最像默认答案
- 或其槽位位置最容易被模型学成 shortcut

### 根因 4：final_select_only 训练集本身不平衡
即使 val gold 分布平衡，也不代表 train 的 `final_candidate_label` 平衡。  
如果 train 中 `patchtst__keep` 明显偏多，模型塌到它是合理结果。

---

## 3. 立即执行的排障顺序

本轮不要先改大结构，不要先进 RL，先按以下顺序做代码级排障。

---

# Step 1：核查 final_select_only 训练集分布

## 目标
确认真正进入 trainer 的 **train.parquet** 是否平衡。

## 需要统计
对：

- `dataset/ett_sft_etth1_v19_final_select_only/train.parquet`

统计：

1. `final_candidate_label` 全局分布
2. 按 `risk_label` / `turn2_decision` 分组后的分布
   - `default_ok` 子集上的 `final_candidate_label`
   - `default_risky` 子集上的 `final_candidate_label`
3. 每类样本数
4. 是否存在极端稀有类

## 通过条件
- 全局上不要求完全平衡，但不能出现：
  - 某一类 > 50% 且其余类明显过少
- 在 `default_ok` / `default_risky` 子集内：
  - gold label 分布应有可学性
  - 至少不能全部偏向同一类

## 如果失败
若 train gold 分布严重偏 `patchtst__keep`：

### 立即修复
- 对 `final_candidate_label` 做 **balanced sampling**
- 或 per-class repeat factor
- 或 loss weighting

---

# Step 2：核查真实训练步数是否足够

## 目标
确认此次 Gate 2 失败是否本质上因为只训练了极少 steps。

## 需要记录
- `len(train.parquet)`
- `per_device_batch_size`
- `world_size`
- `gradient_accumulation_steps`
- `steps_per_epoch`
- `optimizer_step_count`
- 实际 `global_step`

## 核心判断
如果最终 warm-up 只有：

- `global_step = 16`

那么这轮结果**不应被当作方法已失败的充分证据**。

## 通过标准
对于这种 final selection warm-up：

### 建议最低门槛
- 先保证 **200 个 optimizer steps**

在达到这个量之前，不建议对行为坍塌下最终结论。

## 如果失败
若当前是 “1 epoch 只跑 16 步”：

### 立即修复
- 不再按 epoch 控制
- 改成 **按 steps 控制训练**
- 例如：
  - `max_steps = 200`
  - `save/eval every 25 or 50 steps`

---

# Step 3：核查 candidate 顺序偏置

## 目标
确认模型是不是在学“候选槽位偏置”而不是学真正比较。

## 需要检查
在最终构造 Turn 3 prompt 的逻辑里，确认：

- `patchtst__keep` 是否总在第一位
- `itransformer__keep` 是否总在某固定位置
- `default_ok` 与 `default_risky` 的 candidate 排列是否固定
- summary / head / tail preview 的模板长度是否有显著差异

## 最小 A/B 测试
保持所有数据不变，只做一个改动：

### A 版
- 保持当前 candidate 顺序

### B 版
- 对 candidate 顺序做随机打乱
- 但 gold `candidate_id` 保持正确映射

然后重新做短 warm-up，对比：

- `selected_candidate_distribution`
- `single_candidate_max_share`
- `exact_match_rate`
- `final_vs_default_mean`

## 结果解释
### 如果 B 版不再总选 `patchtst__keep`
或变成总选“第一个槽位”：

说明：

## **存在强顺序偏置**
这时必须修：

- candidate 顺序随机化
- 或显式位置去偏置

---

# Step 4：核查 Turn 3 supervision 是否过薄

## 目标
判断当前 `<think> + candidate_id` 是否真的提供了足够区分监督。

## 需要检查
随机抽样训练集中的 `final_select_only` rows：

- 查看 assistant target
- 统计 `<think>` 的文本多样性
- 统计 `<think>` 是否真正引用了样本特异的信息
- 统计 `<think>` 是否只是模板常量句

## 当前怀疑
如果 `<think>` 基本都是模板化常量句：

- 真正的监督信号几乎全压在最后 `candidate_id=...` 几个 token 上
- 模型很容易塌成一个高频答案

## 修复方向
把 Turn 3 target 改成：

### “比较理由 + 最终 candidate_id”
不是简单一句空泛解释，而是要求：

- 对 default path 与 alt path 做显式比较
- 说明为什么当前 candidate 更合适
- 再输出 `candidate_id`

### 目标样式建议
```xml
<think>
The default-path candidate is stable but misses the recent slope change.
The selected alternative better matches the latest local dynamics without introducing implausible spikes.
</think>
<answer>candidate_id=patchtst__keep</answer>
```

### 要求
- `<think>` 必须和样本有关
- builder 不能用常量模板敷衍
- reasoning 文本要真正携带判别信息

---

# Step 5：核查 Turn 3 输入的 candidate 摘要是否足够可分

## 目标
确认压缩后的 summary/head/tail preview 是否还能支持 final selection。

## 背景
你已经为了修 prompt 过长，把：

- full forecast dump

压缩成了：

- compact summary
- head preview
- tail preview

这解决了训练链路，但可能又削弱了候选可分性。

## 建议做一个可学性审计
### 数据
使用最终给 Turn 3 的 candidate-level features：

- candidate summaries
- head/tail stats
- default/alt path 标记
- 其他你实际注入 prompt 的可见信息

### 任务
预测 gold `final_candidate_label`

### 模型
- Logistic Regression
- RandomForest
- LightGBM / XGBoost

### 观察指标
- macro F1
- top2 accuracy
- confusion matrix

## 结果解释
### 如果简单模型也学不好
说明：

## **Turn 3 当前可见的 candidate 摘要仍然不足以支持 final selection**
这时不能只靠加步数，必须增强输入表示。

### 如果简单模型能学好
说明问题更偏训练方式/顺序偏置，而不是信息不足。

---

## 4. 修复方案（按优先级）

---

# 修复 1：先把 warm-up 拉到足够 steps

## 目标
不要再用 “1 epoch = 16 steps” 的 warm-up 来判定行为是否真的学会。

## 建议
把 final-select warm-up 改成：

- `max_steps = 200`
- 或至少 `100 steps`

### 训练策略
- 每 `25 steps` eval 一次
- 每 `25 steps` probe 一次
- 记录 collapse 是否缓解

## 验收指标
- `single_candidate_max_share` 不再是 `1.0`
- `selected_candidate_distribution` 至少覆盖 2~3 类
- `final_vs_default_mean` 逐步下降

---

# 修复 2：对 final_select_only 做 class balancing

## 目标
防止 train gold 分布把模型推向单一候选。

## 建议做法
### 方法 A：balanced sampler
对 `final_candidate_label` 做采样平衡。

### 方法 B：repeat factor
对少数类做 oversampling。

### 方法 C：class-weighted loss
如果 trainer 支持，可对 label 做类别加权。

## 推荐顺序
先做 **A 或 B**，因为实现更直接。

## 验收指标
重新训练后：

- `selected_candidate_distribution` 不应继续塌成单一类
- `single_candidate_max_share < 0.7`

---

# 修复 3：candidate 顺序随机化

## 目标
打掉“候选槽位偏置”。

## 改法
在构建 Turn 3 prompt 时：

- 对 candidate blocks 做随机顺序
- gold `candidate_id` 动态映射
- 可选保留一个 `candidate_rank` 字段供调试

## 注意
### 训练时
- 随机打乱

### 验证 / probe 时
有两种方式：

#### 方式 1
固定一个随机种子后打乱（便于复现）

#### 方式 2
测试两版：
- 固定顺序
- 随机顺序

看模型是否存在明显顺序依赖

## 验收指标
- 随机化后不再稳定塌到 `patchtst__keep`
- 若塌缩发生变化，说明之前存在顺序偏置

---

# 修复 4：增强 Turn 3 reasoning supervision

## 目标
不要让监督信号只压在最后几个 token 上。

## 改法
builder 生成 Turn 3 target 时，增加一段有判别力的 reasoning 文本。

### reasoning 必须包含
- default path 与 alt path 的比较
- 局部动态是否匹配
- 是否存在不合理波动 / spike
- 为什么当前 candidate 更好

### 禁止
- 常量模板句
- 所有样本几乎一样的空洞推理
- 只写一句 “select the best candidate”

## 实现建议
可以先用规则化模板生成，但模板输入必须来自样本真实统计：

例如：
- recent slope mismatch
- local volatility fit
- spike risk
- head/tail deviation
- candidate summary contrast

## 验收指标
- `<think>` 文本多样性显著提高
- candidate collapse 缓解
- `exact_match_rate` 上升
- `top2_hit_rate` 上升

---

# 修复 5：必要时增强 Turn 3 输入表示

## 目标
若 summary/head/tail preview 仍不足以支撑 final selection，则增强 candidate 可见信息。

## 建议逐步增强，不要一次全恢复 full forecast dump

### 第一层增强
在 candidate block 中加入：
- 与 default path 的相对差异摘要
- 局部 slope 指标
- local volatility 指标
- recent-window mismatch 指标

### 第二层增强
对 head/tail preview 做更结构化表示：
- 不只是 raw 数字片段
- 而是附上局部统计对比

### 第三层增强（必要时）
仅在 risky 子集上，给更丰富的 candidate preview

## 验收指标
简单模型在 candidate features 上的：
- macro F1
- top2 accuracy
应明显改善

---

## 5. 建议的实验矩阵

不要一次改太多，建议按以下顺序做最小实验矩阵。

### Exp-1：仅加步数
- 其他不动
- train 到 100~200 steps

### Exp-2：加步数 + train class balancing
- 看是否还塌到 `patchtst__keep`

### Exp-3：加步数 + class balancing + candidate 随机顺序
- 检查顺序偏置

### Exp-4：加步数 + class balancing + reasoning supervision 增强
- 检查“薄 supervision”假设

### Exp-5：若前面都不够，再增强 candidate features
- 检查 Turn 3 可见信息是否仍不足

---

## 6. Gate 2 重验收标准（修复后）

修复后，重新做 Gate 2 时建议使用以下门槛：

### 必过条件
- `protocol_ok_rate = 1.0`
- `materialize_ok_rate = 1.0`
- `single_candidate_max_share < 0.7`
- `selected_candidate_distribution` 至少覆盖 2 类以上
- `final_vs_default_mean < 0`

### 更强目标
- risky 子集：
  - `final_vs_default_mean < 0`
- default_ok 子集：
  - 不应被显著过度 override
  - 即 `default_ok final_vs_default_mean` 接近 0 或小于 0

### 行为目标
- 不再固定输出同一个 `candidate_id`
- collapse 被明显缓解

---

## 7. 进入下一阶段的条件

只有当修复后 Gate 2 通过，才允许：

- 构建 `full_stepwise_v19`
- 进入 Gate 3
- 再考虑 RL

否则：

- 禁止进入 RL
- 先把 final selection 行为学稳

---

## 8. 一句话结论

当前 v19 并不是“大方向又错了”，而是终于走到了真正的 **final selection** 阶段，结果暴露出一个新的、更具体的问题：

> **Turn 3 的训练步数太少、监督过薄、候选展示可能存在顺序偏置，因此模型学成了一个单一默认答案 `patchtst__keep`。**

所以现在最正确的动作不是继续大改结构，而是：

1. **先把 final-select warm-up 拉到足够 steps**
2. **检查并修复 train label 不平衡**
3. **随机化 candidate 顺序**
4. **增强 Turn 3 的判别性 reasoning supervision**
5. 必要时再增强 Turn 3 可见的 candidate 信息

这才是当前 Gate 2 失败下，最稳、最有针对性的修复路径。
