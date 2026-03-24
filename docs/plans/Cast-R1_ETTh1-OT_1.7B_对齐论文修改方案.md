# Cast-R1 ETTh1-OT（单变量，Qwen3-1.7B）对齐论文修改方案

## 0. 文档目的

这份文档的目标不是“机械复刻论文全部实验条件”，而是把当前工程整理成三层清晰结构：

1. **论文硬约束**：必须对齐，否则不能称为 Cast-R1 风格实现。
2. **资源受限下的工程增强**：论文没有明确规定，但在 ETTh1-OT 单变量 + Qwen3-1.7B + 4x5090 条件下合理。
3. **当前代码现状与清理项**：哪些已经对齐，哪些还没有，哪些旧开关/旧分支应直接删除。

当前现实约束：

- 数据任务：ETTh1-OT 单变量
- 骨干模型：Qwen3-1.7B
- 计算资源：4 张 5090

这意味着：**我们追求的是“机制对齐论文”，不是“数值复现论文 8B / 多变量结果”。**

---

## 1. 论文硬约束

下面这些内容，应该被视为本次修改的硬边界。

### 1.1 三阶段 agent workflow

论文明确要求：

- **Turn 1**：只做 feature extraction / diagnostics
- **Turn 2**：基于 Turn 1 的分析结果做 model routing，并调用 forecasting model
- **Turn 3**：基于 analysis history + prediction results 做 reasoning、refinement、final output generation

这意味着：

- Turn 1 不能调预测工具
- Turn 2 不该提前输出最终答案
- Turn 3 不该再调工具

### 1.2 memory-based state management + stage-aware prompt

论文明确把 memory 分成两部分：

- `Analysis History`
- `Prediction Results`

系统必须根据 memory 状态决定当前处于哪个阶段，并只把当前阶段需要的信息拼进 prompt。

这意味着：

- prompt 不能是静态大杂烩
- 运行时必须以 memory 驱动 turn-stage
- workflow 违规应在 runtime 被拒绝

### 1.3 SFT warm start

论文中的 SFT 不是普通 instruction tuning，而是：

- curated decision trajectories
- 来自 heuristics / rule-based strategies
- 对中间状态提供 step-level supervision

因此 SFT 必须教会：

- 正确工具调用格式
- 正确三阶段顺序
- 正确 routing 语义
- 正确 Turn 3 反思/修正语义

### 1.4 multi-turn RL（GRPO）

论文里 RL 负责优化长程决策，而不是凭空创造三阶段能力。

所以 RL 的角色应该是：

- 优化 planning / tool usage / routing / refine
- 在 episode-level delayed reward 下学到更稳的 sequential policy

而不是：

- 用 RL 去弥补 SFT 数据本身错误的 target

### 1.5 curriculum learning

论文 3.5.2 的 curriculum 难度来自两维：

- `reference teacher prediction difficulty`
- `normalized permutation entropy`

训练顺序是：

1. 低 teacher error + 低复杂度
2. 更高 teacher error，但结构仍相对清晰
3. 高 teacher error + 高 entropy / 高噪声

所以 curriculum 不能只按时间或随机切分。

### 1.6 multi-view reward

论文 reward 的核心原则是：

- 主项：normalized/log-transformed MSE
- 辅项：trend / season consistency
- 辅项：local turning-point alignment
- 约束项：format validity
- 约束项：output length consistency

关键点：

- 预测误差是主信号
- trend/season 和 turning point 不是可有可无
- length 是辅助约束，不应压过主误差项

### 1.7 Refine 机制必须存在，但不要求每条样本都改数值

论文 Table 13 说明：

- 去掉 `Refine` 会明显变差

但论文并没有说：

- 每条样本都必须改数值
- Turn 3 必须总是显著偏离 selected forecast

因此正确理解应是：

- Turn 3 必须具有“检查 selected forecast 是否合理，并在必要时修正”的能力
- 对证据一致的样本，**keep unchanged** 也是合法行为
- 问题不在于“有没有改”，而在于“是否进行了真实的反思判断”

### 1.8 输出协议

论文 prompt design 明确要求：

- `<think>...</think>`
- `<answer>...</answer>`

因此 runtime 也应以此为协议边界。  
不能只要求 `<answer>`，否则论文里的 reflection 会在实现上退化成可选项。

---

## 2. 资源受限下允许保留的简化

下面这些不是论文原始设置，但可以接受，只要明确写成“工程版简化”。

### 2.1 数据仍使用 ETTh1-OT 单变量

论文 ETTh1 是 7 变量；当前任务是 ETTh1-OT 单变量。  
这会带来两个直接差异：

- iTransformer 的优势可能弱于论文多变量场景
- prompt 和特征工具会更偏向单序列结构判断，而不是 cross-channel reasoning

### 2.2 backbone 仍使用 Qwen3-1.7B

论文表明 1.7B < 4B < 8B。  
因此当前目标应是：

- 行为对齐论文
- 指标明显好于当前实现
- ablation 方向与论文一致

而不是要求达到 8B 的 ETTh1 指标。

### 2.3 teacher 池先保留 PatchTST / Chronos2 / iTransformer / ARIMA

论文支持多 forecasting tools 协同。  
在单变量版本里，teacher 的相对强弱可能变化，但不应在重构前就武断删除某个 teacher。

因此默认做法是：

- 先重评估
- 再决定是否缩池

### 2.4 训练超参允许资源自适应

论文 appendix 给了 1.7B 的一套 RL 设定，但你当前是 4x5090、多卡工程版。  
因此：

- `rollout_n`
- `temperature`
- `KL`
- `response_length`
- `epochs`

可以做资源自适应调整，但必须单独标成“工程超参”，不能写成“论文硬约束”。

---

## 3. 当前代码现状：哪些已经部分对齐，哪些还没有

这一节只描述**当前 repo 的真实状态**，不再沿用已经过时的旧诊断。

### 3.1 已经部分对齐论文的部分

#### A. reward 主线已经不是 strict ablation

当前 [reward.py](/data/linyujie/Cast-R1-TS-main/Cast-R1-TS-main/recipe/time_series_forecast/reward.py) 顶部已经是 composite reward 默认设置：

- change-point / season-trend 主项默认开启
- 主误差项固定为 normalized + log-transformed MSE
- 旧的 strict-style reward 开关已经从主线移除

也就是说：

- 文档里不能再把当前主线写成“还是 strict ablation”
- 现在的问题不是 reward 仍是旧版，而是 reward、SFT、runtime 之间还不一致

#### B. RL dataset 已经有 curriculum 相关字段

当前 [build_etth1_rl_dataset.py](/data/linyujie/Cast-R1-TS-main/Cast-R1-TS-main/recipe/time_series_forecast/build_etth1_rl_dataset.py) 已经有：

- `reference_teacher_error`
- `normalized_permutation_entropy`
- `reference_teacher_error_band`
- `normalized_permutation_entropy_band`
- `difficulty_stage`
- `curriculum_stage`

所以：

- “RL 数据完全没有 curriculum 元信息”这句已经过时
- 当前更需要关注的是：这些字段是否被训练链路真正使用，以及生成质量是否健康

### 3.2 仍明显不对齐论文的部分

#### A. SFT target 主线已经清理，但 refine 仍是工程实现

当前 [build_etth1_sft_dataset.py](/data/linyujie/Cast-R1-TS-main/Cast-R1-TS-main/recipe/time_series_forecast/build_etth1_sft_dataset.py) 主线已经完成了这几项清理：

- GT fallback 已删除
- last-value padding 已删除
- `prediction_mode` 旧分支已删除
- `keep_selected_forecast` / `route_then_refine` 旧语义标签已收敛为：
  - `validated_keep`
  - `local_refine`

当前仍未完全对齐论文的地方在于：

- `local_refine` 仍然是**确定性规则器**生成的工程 target
- 还不是论文原文定义的学习式 revision / refinement 机制
- keep / refine 的样本比例与阶段覆盖还没有单独做重平衡

#### B. Turn 3 的 runtime 协议仍偏弱

这一项的历史主问题已经修掉：

- parser 现在要求 `<think>...</think><answer>...</answer>`
- 缺 `<think>`、缺 `<answer>`、tag 外有额外文本都会被拒绝

当前剩下的问题不是“协议没检查”，而是：

- Turn 3 仍然是自由数值生成
- 因此即使协议边界正确，仍可能出现 `94/95`、缺 `</answer>`、尾部重复

#### C. Turn 3 仍是自由数值生成

虽然 prompt 已经写成“selected forecast 作为初始预测”，但模型仍然需要自己重新生成：

- `<think>`
- 96 行数值
- `</answer>`

所以依然会出现：

- 94/95 行
- 缺 `</answer>`
- 尾部数值循环

#### D. refinement supervision 还不够强

这一项相比旧版本已经前进了一步：

- 当前已有显式 `validated_keep / local_refine`
- 当前已有 `refined_prediction_text`
- 当前已有 `refine_ops_signature / refine_gain_* / refine_changed_*`

但仍不完全够强，因为：

- refine target 还是规则器离线合成
- 还没有 sample balancing 去保证模型不会重新塌回“几乎总是 keep”

#### E. 观测链路仍有部分断点

这一项也比旧版本更好：

- chain debug / aggregate / sample 的大部分主线字段已经打通
- 旧的 reward/debug 字段残留也已经基本清理

剩余工作主要是：

- 在长跑实验里继续验证新字段的稳定性
- 确保 README / debug 分析脚本与当前字段保持一致

---

## 4. 本次大对齐的总原则

### 4.1 论文硬约束

必须做到：

- 三阶段 workflow
- memory 驱动 stage-aware prompting
- SFT warm start
- GRPO 多轮 RL
- curriculum 两维
- multi-view reward
- Turn 3 真正执行 reflection + refine + final output

### 4.2 工程实现原则

本次应当优先追求：

- 新主线单一明确
- 删除旧分支，不保留多套互相冲突的路径
- 把 ablation 变成“单独实验配置”，而不是散落在主线代码里的环境变量开关

一句话说：

> 这次不是再叠一层兼容逻辑，而是把主线收敛成一条 paper-aligned pipeline。

---

## 5. 逐文件修改建议

## 5.1 `recipe/time_series_forecast/build_etth1_sft_dataset.py`

### 目标

把 SFT 从“teacher 包装器”改成“paper-aligned decision trajectory builder”。

### 当前状态与剩余修改

#### A. 删除 legacy `prediction_mode`

这一项已完成。已删除：

- `ground_truth`
- `preferred`
- `reference_teacher` 之外的分支选择逻辑

主线只保留：

- `reference_teacher` 作为 base forecast 来源

如果后续需要做 ablation，应单独做脚本，不要放在主 builder 主线里。

#### B. 删除 GT fallback

这一项已完成。已删除：

- `_ground_truth_prediction_text(...)`
- `fallback_ground_truth`
- `prediction_source = "ground_truth"` 主线兜底

任何 teacher prediction 缺失都应：

- 回退到第二 teacher 或
- 直接丢样本

但不能把 GT 塞进 Turn 3 target。

#### C. 删除 last-value padding

这一项已完成。已删除：

- `trimmed = trimmed + [trimmed[-1]] * (...)`

长度不合法的 prediction：

- 标 invalid
- 不进入 SFT target

#### D. 保留 `keep unchanged`，但必须是“反思后的 keep”

论文允许 Turn 3 在证据一致时保持 selected forecast 不变。  
因此不要把目标写成“每条样本都必须改数值”。

这一项当前主线已完成到：

- `validated_keep`
- `local_refine`

并且：

- `validated_keep` 必须带简短但真实的 reflection
- `local_refine` 必须带明确数值 target

#### E. 增加显式 refined target

这一项当前主线已完成到：

- `base_teacher_prediction_text`
- `refined_prediction_text`
- `refine_ops`
- `refine_gain_mse`
- `refine_gain_mae`
- `turn3_target_type` (`validated_keep` / `local_refine`)

这里的规则化 refine 不是论文原文细节，而是**工程实现论文 Refine 机制**的手段。  
论文要求的是：

- Turn 2 给出 `initial forecast`
- Turn 3 做 reasoning-based forecasting / revision / refinement

但论文没有给出显式的离线 refine 标签体系。  
因此 `validated_keep / local_refine` 应明确标注为**工程实现标签**，而不是论文原始术语。

当前主线已实现的局部修正包括：

- isolated spike smoothing
- flat-tail repair
- local level adjustment
- local slope adjustment
- amplitude clip

#### F. 反思文本必须和 target 一致

当前最需要避免的是：

- 文本说“做了修正”
- 数值上其实没动

或反过来：

- 数值动了
- 文本却说“我保持不变”

Turn 3 的 `<think>` 必须严格由 target 类型驱动生成。

#### G. “需要反思”的样本必须离线筛选，而不是全量样本一刀切

Turn 3 的 refine 样本不应覆盖所有 teacher 样本。  
必须先区分：

- **routing 问题**：selected model 整体就错了，这类问题属于 Turn 2，不应强塞给 Turn 3
- **local refinement 问题**：selected model 整体可用，但存在局部缺陷，这类问题才属于 Turn 3

正确的离线筛选标准应是：

1. 先把 Turn 2 的 `selected_prediction_model` 输出作为 `base forecast`
2. 只允许在 `base forecast` 的基础上做**局部、小幅、可解释**的修正
3. 只有当修正后的候选满足以下条件时，样本才标为 `local_refine`：
   - 相比 `base forecast` 有真实误差改善
   - 改动范围局部，不是重写整条 96-step forecast
   - 改动幅度受限，不允许全局大改
4. 如果不满足上述条件，则回退为 `validated_keep`

也就是说：

- `validated_keep` 教模型“什么时候不要乱改”
- `local_refine` 教模型“什么情况下该做局部修正”

#### H. GT 只用于离线选择 refine candidate，不能直接塞进 Turn 3 target

GT 的合理用途是：

- 离线评估不同 refine candidate 的效果
- 判断某个 candidate 是否真的优于 `base forecast`

GT 不能直接用于：

- 把 Turn 3 target 直接替换成 ground truth
- 让模型学成“看到 selected forecast 后直接复述 GT”

因此正确做法是：

- 先构造若干基于 `base forecast` 的局部 refine candidate
- 再用 GT 只做离线 selection
- 最终保留一个 `validated_keep` 或 `local_refine` target

#### I. 不把 `teacher_blend` 当成 Turn 3 主线

论文主线是：

- Turn 2 选一个 forecasting model
- Turn 3 基于该 selected forecast 做 reasoning / refinement

这一项当前主线已完成：

- 第二 teacher 可以继续用于离线分析、fallback 或 teacher 评估
- 但不应把 `teacher_blend` 当成 Turn 3 主线 target

否则会把“单模型 routing + refine”重新拉回“多模型混合决策”，偏离论文主线。

#### J. 建议显式记录 refine 的位置和幅度

这一项当前主线已完成。SFT 样本已补充：

- `refine_changed_value_count`
- `refine_first_changed_index`
- `refine_last_changed_index`
- `refine_changed_span`
- `refine_mean_abs_delta`
- `refine_max_abs_delta`

这样后续能直接检查：

- 模型是不是只会复制 selected forecast
- 模型是不是改得过大
- 模型的 refine 是否真的是局部修正

---

## 5.2 `recipe/time_series_forecast/build_etth1_rl_dataset.py`

### 目标

RL dataset 主线不需要大推翻，但需要收敛命名与输出。

### 剩余主要工作

#### A. 以论文字段为主命名

主线保留并统一：

- `reference_teacher_error`
- `normalized_permutation_entropy`
- `curriculum_stage`

类似：

- `difficulty_stage`
- `curriculum_band`

这类工程中间字段可以保留作 debug，但不要让主线语义混乱。

#### B. 产出显式 staged train files

继续产出：

- `train_stage1.jsonl`
- `train_stage12.jsonl`
- `train_stage123.jsonl`

让 curriculum 调度在数据侧明确，而不是训练时隐式拼接。

#### C. metadata 必须能复核样本健康度

至少输出：

- teacher error thresholds
- entropy thresholds
- stage distribution
- offline best model distribution

---

## 5.3 `recipe/time_series_forecast/build_etth1_high_quality_sft.py`

### 目标

这个文件的角色应定义为：

- teacher benchmark
- teacher metadata collector
- curated SFT candidate selector

### 必改

#### A. 把“论文硬约束”和“工程采样策略”分开

论文没有规定：

- candidate 必须 2400
- curated 必须 600
- easy/medium/hard 必须 30/40/30

所以这些都应写成：

- 默认工程配置
- 可调超参

而不是“论文要求”。

#### B. selection score 应包含三件事

- teacher quality
- diversity coverage
- optional refine gain

其中 refine gain 是工程增强，不是论文硬约束。

#### C. 必须显式输出 teacher 健康统计

至少包括：

- best teacher distribution
- margin distribution
- invalid teacher ratio
- stage distribution

---

## 5.4 `recipe/time_series_forecast/reward.py`

### 目标

把 reward 主线收敛成**单一的 paper-aligned reward implementation**。

### 必改

#### A. 删除主线里的 legacy reward 开关

这一项已完成。以下旧开关已从主线删除：

- `TS_STRICT_LENGTH_GATE`
- `TS_STRICT_RAW_MODE`
- `TS_REWARD_USE_ORIG_SCALE`

原因：

- 当前主线已经明确是 paper-aligned composite reward
- 旧开关继续存在，只会让训练在不同 run 里悄悄切回旧逻辑
- ablation 应通过单独配置或单独 reward variant 完成，不应让主线代码到处 if/else

#### B. 保留论文主项与辅项

主线 reward 只保留：

- normalized/log-transformed MSE
- trend/season consistency
- turning-point alignment
- format validity
- length consistency

#### C. `smooth penalty` 只能作为可选工程项

如果要保留 `smooth`：

- 必须在文档里标明是工程增强
- 不要写成论文硬约束

默认建议：

- 第一版先不把它做成主项
- 只有当前 5 项不够稳定时，再作为工程增强打开

#### D. 协议检查必须包含 `<think>`

这一项当前也已完成。reward / parser / protocol 现在统一要求：

- 缺 `<think>` 视为协议不完整
- 缺 `<answer>` 视为协议失败

否则论文中的 reflection 永远无法在实现上成立。

---

## 5.5 `recipe/time_series_forecast/prompts.py`

### 目标

prompt 需要更严格贴近论文 Strategy Box，但也不能把 Turn 3 写成“必须改数值”。

### 当前状态与剩余工作

#### A. Turn 1

继续保持：

- only feature tools
- no prediction function

#### B. Turn 2

继续保持：

- 根据 analysis history routing
- exactly one forecasting model invocation

#### C. Turn 3

Turn 3 应明确为：

- use selected model prediction as the base forecast
- reflect against analysis history and recent history constraints
- keep unchanged if base forecast is already consistent
- otherwise make small, local, evidence-based corrections

不要再出现会把策略推向纯复制的表述。  
也不要写成“必须总是修正”。

#### D. `<think>` 不是装饰

prompt 里必须明确：

- `<think>` 是 reflection on consistency / adjustment
- `<answer>` 是最终 96 点序列

并且 runtime 必须和 prompt 约束一致。

---

## 5.6 `recipe/time_series_forecast/time_series_forecast_agent_flow.py`

### 目标

让 runtime 真正执行论文协议，而不是“prompt 这么写，但 parser 不管”。

### 必改

#### A. workflow 违规要 fail fast

必须显式拒绝：

- Turn 1 调预测工具
- Turn 2 未调预测工具就想输出最终答案
- Turn 3 再调工具

这一项当前已基本完成。

#### B. 最终答案协议必须检查 `<think> + <answer>`

不能再只接受 `<answer>`。

#### C. refinement 监控继续保留

继续记录：

- `selected_forecast_exact_copy`
- `refinement_changed_value_count`
- `final_vs_selected_mse`

但这些是诊断指标，不应替代协议约束。

---

## 5.7 `examples/time_series_forecast/configs/etth1_ot_qwen3_gpu012.sh`

### 目标

配置文件里应明确区分：

- smoke / debug 配置
- formal training 配置

### 原则

- 论文 appendix 的 `G=8, temp=1.0` 是参考，不是 4x5090 下的硬约束
- 当前 1.7B 版本应先优先保证稳定和协议收敛
- 所有 legacy debug env 不应散落成长期默认值

---

## 6. 可直接删除的旧开关、旧接口、旧代码

这一节是这次大整理必须执行的内容。

### 6.1 SFT builder 中可删除的旧分支

这一项已完成清理，已删除：

- `prediction_mode = "ground_truth"`
- `prediction_mode = "preferred"`
- `_ground_truth_prediction_text`
- `fallback_ground_truth`
- last-value padding

### 6.2 reward 中可删除的旧主线开关

这一项已完成清理，已删除：

- `TS_STRICT_LENGTH_GATE`
- `TS_STRICT_RAW_MODE`
- `TS_REWARD_USE_ORIG_SCALE`

如果确实要做 ablation：

- 建单独 reward variant
- 或单独实验配置文件

不要把旧逻辑留在主线里随时可被环境变量切回。

### 6.3 已弃用的语义标签要收敛

这一项当前已完成主线收敛。旧标签：

- `keep_selected_forecast`
- 当前基于弱启发式的 `route_then_refine`

已经收敛成：

- `validated_keep`
- `local_refine`

### 6.4 文档和脚本里的旧路径、旧实验名要一起删

包括：

- 旧 `v1/v2` 的实验命名
- 已废弃的数据目录默认值
- 已废弃 README 命令

这次要做的是**主线收敛**，不是继续维持多条历史分支并存。

---

## 7. 推荐训练流程

这次不建议再“边跑 RL 边补洞”，而是按下面顺序重建：

### Phase 0：teacher benchmark

输出：

- teacher ranking
- per-stage / per-band 胜率
- margin 分布

### Phase 1：重建 RL dataset

输出：

- `train_stage1.jsonl`
- `train_stage12.jsonl`
- `train_stage123.jsonl`

### Phase 2：重建 HQ-SFT 候选集

目标：

- teacher 质量可控
- stage 覆盖合理
- candidate 池和 curated 集规模作为工程超参管理

### Phase 3：重建 SFT runtime dataset

要求：

- 无 GT fallback
- 无 last-value padding
- 有显式 Turn 3 target type
- 有 refine stats

### Phase 4：SFT-only sanity

至少检查：

- Turn 1 合规率
- Turn 2 routing 合规率
- Turn 3 `<think> + <answer>` 协议合规率
- validated_keep / local_refine 分布

### Phase 5：curriculum RL

顺序：

1. stage1
2. stage12
3. stage123

---

## 8. 成功标准

### 8.1 行为层

成功不应只看 MSE，还要看：

- Turn 1 / 2 / 3 workflow 合规
- `<think>` 不再缺失
- Turn 3 不再主要靠协议漏洞通过
- selected model distribution 不完全塌缩

### 8.2 数值层

在 ETTh1-OT + 1.7B 设定下，更合理的成功标准是：

- full pipeline 优于当前实现
- `w/o Refine` 变差
- `w/o Curriculum` 变差
- composite reward 优于旧 strict-style 路径

### 8.3 工程层

主线代码应做到：

- 单一路径
- 少开关
- 少兼容分支
- 文档、README、配置、代码一致

---

## 9. 最终判断

这次真正需要做的，不是再补一个局部 patch，而是：

1. **把论文硬约束与工程增强分开**
2. **把已经过时的旧诊断从文档里清掉**
3. **把主线里所有历史兼容开关和弃用分支删除**
4. **用单一、可解释、paper-aligned 的 pipeline 重新组织 SFT -> RL**

一句话概括：

> 先做“主线收敛与旧逻辑删除”，再做“论文对齐实现”，最后再跑新的 SFT 和 curriculum RL。
