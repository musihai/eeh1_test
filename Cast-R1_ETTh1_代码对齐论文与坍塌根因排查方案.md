# Cast-R1（ETTh1专项）代码对齐论文与“单模型坍塌”根因排查方案

> 目标：这份文档用于直接交给 Codex 执行排查与最小修改。当前范围只考虑 **ETTh1**，不要求一次性扩展到多数据集。重点是两件事：
>
> 1. 判断当前代码哪些地方已经偏离论文主线，并识别哪些偏离会显著影响效果；
> 2. 系统排查“策略坍塌为只选择一个预测模型”的根因，明确该看哪些指标、在哪些文件里看、如何判因。

---

## 0. 给 Codex 的执行要求

请按下面顺序执行，不要一上来大改训练框架：

1. **先做静态审计**：确认代码与论文主线是否一致，列出“高风险偏离点 / 中风险偏离点 / 暂不处理项”。
2. **再做日志排查**：基于现有 debug 文件、聚合指标、训练脚本和数据构造脚本，判断“单模型坍塌”的根因更偏向：
   - SFT 先验过强；
   - RL exploitation 过强；
   - expert 环境不对等；
   - runtime heuristic 泄露；
   - Turn 3 反思修正太弱；
   - feature analysis 覆盖不足。
3. **最后再做最小修改方案**：每次只改一个因素，便于做 ablation 和归因，不要多处同时重构。

输出物要求：

- 一份审计报告：
  - `high_risk_deviations`
  - `medium_risk_deviations`
  - `non_blocking_deviations`
- 一份根因判断报告：
  - `likely_root_cause_ranked`
  - `evidence`
  - `next_experiments`
- 最多提交 **3 组最小补丁**，每组补丁只服务一个明确假设。

---

## 1. 论文主线（ETTh1 范围内应尽量保留的核心）

当前只做 ETTh1，因此“单数据集”本身不构成严重偏离。真正需要对齐的是下面这些主线机制：

1. **三阶段 agent 流程**
   - Turn 1：feature extraction / planning
   - Turn 2：adaptive model selection + forecasting tool invocation
   - Turn 3：reflection / refinement / final answer

2. **memory-based state management**
   - 不能每一轮都从零开始；
   - 要让已提取的分析结果、中间预测结果、工具执行记录进入状态。

3. **SFT + RL 两阶段训练**
   - SFT 负责教会流程和动作模式；
   - RL 负责学会真正的自适应路由与长程策略。

4. **curriculum RL**
   - 难度推进可以是简化版，但不能完全丢掉“先易后难”的思路。

5. **最终策略应该来自状态驱动的选择，而不是过强的外部启发式提示**
   - 可以有 planning；
   - 但不应在 runtime 中把“更推荐选哪个预测模型”暗示得过强。

---

## 2. 当前代码中需要优先排查的“高风险偏离点”

以下偏离点即使只做 ETTh1，也可能明显伤害论文主线，或直接导致效果变差、策略坍塌。

---

### 高风险偏离点 A：runtime 里提前注入了候选预测模型提示，削弱真正的 adaptive model selection

#### 现象

在 agent runtime 中，诊断规划不是只决定“该用哪些 feature tools”，还提前产生了候选预测模型信息：

- `build_diagnostic_plan(...)`
- `diagnostic_primary_model`
- `diagnostic_runner_up_model`

并且这些信息会进入 Turn prompt。

#### 重点文件

- `recipe/time_series_forecast/time_series_forecast_agent_flow.py`
- `recipe/time_series_forecast/prompts.py`
- `recipe/time_series_forecast/diagnostic_policy.py`

#### 为什么这是高风险偏离

论文主线更接近：

- 先做 feature extraction；
- 再基于更新后的 state 自适应做模型选择。

而当前实现更接近：

- heuristic router 先给出候选模型；
- LLM 在这个强提示下做执行。

这会导致：

1. RL 更容易学成“沿着 heuristic 提示走”；
2. 路由更容易坍塌到一个模型；
3. 最终学到的是“复读提示”，不是“根据状态做选择”。

#### 建议的最小修改

第一步不要删掉 `required_feature_tools`，只先削弱模型提示：

- 保留 `required_feature_tools`；
- 从 runtime prompt 中去掉：
  - `diagnostic_primary_model`
  - `diagnostic_runner_up_model`
- 保证 Turn 2 只能看到：
  - 历史序列
  - 已执行 feature tool 的结果
  - 当前允许动作空间

#### 修改后重点观察指标

- `selected_model_distribution`
- `prediction_requested_model_distribution`
- `analysis_coverage_ratio_mean`
- `prediction_tool_error_count`
- `final_answer_accept_ratio`
- `strict_length_match_ratio`

#### 验证结论

如果去掉 runtime 候选模型提示后，模型选择分布变得更分散，且总体 reward / MSE 没显著恶化，说明当前坍塌有较大概率是 **runtime heuristic 泄露** 造成的。

---

### 高风险偏离点 B：SFT routing 标签高度依赖 heuristic，可能把 RL warm start 带偏

#### 现象

SFT 构造中，路由标签并非来自“离线真实最优 expert”，而是高度依赖 heuristic：

- `selected_prediction_model`
- `routing_policy_source = "heuristic_rule_based"`

#### 重点文件

- `recipe/time_series_forecast/build_etth1_sft_dataset.py`
- `recipe/time_series_forecast/build_etth1_high_quality_sft.py`
- `recipe/time_series_forecast/build_etth1_heuristic_curated_sft.py`

#### 为什么这是高风险偏离

SFT 会形成很强的行为先验。如果这个先验本身已经偏向某一个 expert，或者与“样本上的真实最优 expert”不一致，那么 RL 通常不会自动纠偏，反而会进一步放大这种偏差。

#### 建议的最小修改

先不要大改 reasoning 文本，只先校正 routing teacher：

1. 统计当前 SFT 数据中 `selected_prediction_model` 的分布；
2. 统计 heuristic label 与 offline best expert 的一致率；
3. 如果一致率不高，改成：
   - routing 标签优先采用 **离线评测最优 expert**；
   - heuristic 仅用于生成文字分析；
4. 对 routing rows 做 **per-model balance**，避免单模型占绝对多数。

#### 修改后重点观察指标

- `train_selected_prediction_model_distribution`
- `train_routing_row_selected_prediction_model_distribution`
- `routing_policy_source_distribution`
- SFT 后 `selected_model_distribution`
- RL 初期 `selected_model_distribution`

#### 验证结论

如果仅仅重新平衡 SFT routing 标签后，RL 不再快速塌成单模型，说明根因主要在 **SFT behavioral prior**。

---

### 高风险偏离点 C：expert 环境不对等，RL 会被迫锁死到唯一稳定模型

#### 现象

即使 agent 逻辑没问题，只要 expert 之间质量或稳定性不对等，RL 也会自然坍塌到单一模型。

需要重点怀疑：

- 某个 expert checkpoint 明显更强；
- 某个 expert 调用错误率高；
- 某个 expert 输出长度更容易不合法；
- 某个 expert 服务更稳定，其他 expert 经常 fallback / default。

#### 重点文件

- `recipe/time_series_forecast/model_server.py`
- `recipe/time_series_forecast/model_path_utils.py`
- `tests/test_model_server_batch.py`
- `artifacts/reports/rl_model_benchmark_*.json`
- 相关 checkpoint / config / provenance 文件

#### 为什么这是高风险偏离

如果环境本身是单峰最优，策略学成“永远选它”并不一定是 bug，而是合理 exploitation。此时修改 agent 逻辑意义不大，先要确认 expert 本身是否公平可选。

#### 建议的最小修改

先做离线强制指定评测，不要先改 agent：

对同一批 ETTh1 样本，分别强制指定：

- `arima`
- `patchtst`
- `itransformer`
- `chronos2`

输出每个 expert 的：

- 有效预测率
- 平均 `orig_mse`
- 平均 `orig_mae`
- 长度匹配率
- 接口错误率
- default / fallback 比例

#### 修改后重点观察指标

- `prediction_tool_error_count`
- `prediction_model_defaulted_ratio`
- 每个 expert 的平均 `orig_mse`
- 每个 expert 的平均 `orig_mae`
- 每个 expert 的长度合法率

#### 验证结论

如果某个 expert 在质量和稳定性上都显著更强，那么“总是选它”首先是环境问题，不应先归咎于 RL 实现。

---

### 高风险偏离点 D：Turn 3 被收缩得过死，反思修正能力过弱，无法纠正 Turn 2 选错模型

#### 现象

Turn 3 当前被严格限制为：

- `KEEP`
- `LOCAL_REFINE`
- 并且 prompt 中明确写了：`If unsure, choose KEEP.`

#### 重点文件

- `recipe/time_series_forecast/prompts.py`
- `recipe/time_series_forecast/validate_turn3_format.py`
- `recipe/time_series_forecast/time_series_forecast_agent_flow.py`

#### 为什么这是高风险偏离

论文主线里的 Turn 3 是 reflection / refinement / final answer。当前实现把它压成了“保守复制或局部微调”，这会让训练更稳定，但也会导致：

- 一旦 Turn 2 选错 expert，Turn 3 几乎没能力救回来；
- 训练会进一步偏向“选一个最稳 expert 就行”；
- agent 的 reasoning / revision 上限被压低。

#### 建议的最小修改

先做轻微放宽，不要完全开放生成：

- 仍保留 strict output protocol；
- 但允许 Turn 3：
  - 修正多个不连续位置；
  - 或做“受限重写”，而不是只能一个连续局部段；
- 去掉或弱化 `If unsure, choose KEEP.` 的强保守指令。

#### 修改后重点观察指标

- `refinement_changed_ratio`
- `refinement_improved_ratio`
- `refinement_degraded_ratio`
- `selected_forecast_exact_copy_ratio`
- `final_vs_selected_mse_mean`

#### 验证结论

如果放宽 Turn 3 后，`refinement_improved_ratio` 明显上升，且最终误差下降，说明之前策略过于依赖 Turn 2 的单次路由，Turn 3 太弱确实在放大坍塌。

---

## 3. 当前代码中“中风险偏离点”

这些点会影响效果或可解释性，但优先级低于上面 4 项。

---

### 中风险偏离点 E：curriculum 被压成 easy / medium / hard，保留主线但粒度较粗

#### 重点文件

- `recipe/time_series_forecast/build_etth1_rl_dataset.py`
- `recipe/time_series_forecast/curriculum_utils.py`

#### 判断

这属于论文思路的简化版，不算严重偏离。只要仍基于 teacher error / entropy 做难度推进，主线还在。

#### 建议

先不用优先大改。除非排查表明 curriculum 组织本身严重失衡，否则不要先动这里。

---

### 中风险偏离点 F：memory 更像工程化状态槽，而不是更抽象的 state abstraction

#### 重点文件

- `recipe/time_series_forecast/time_series_forecast_agent_flow.py`
- `recipe/time_series_forecast/agent_flow_support.py`

#### 判断

只要状态中确实保留了：

- feature analysis
- prediction result
- executed tools
- workflow progress

那么它仍然符合论文“memory-based state management”的主要方向。不是当前最可能导致坍塌的根因。

#### 建议

只需确认状态是否真的在起作用，不需要优先重写整个 memory abstraction。

---

## 4. 当前实现中“暂不视为主线偏离”的部分

这些内容先不要作为主要问题处理：

1. **只做 ETTh1**
   - 当前目标是专项复现，不要求立刻扩展到多数据集。
2. **三阶段 prompt / workflow**
   - 这与论文主线一致。
3. **reward 以最终 delayed reward 为主**
   - 这是论文核心思想之一。
4. **1.7B 实现配置**
   - ETTh1 + 1.7B 可以作为论文附录层级的专项复现版本。

---

## 5. “只选择一个预测模型”时的可能问题清单

下面这些原因要按优先级排查。

---

### 根因 1：某个 expert 在 ETTh1 上确实整体最好

#### 典型表现

- `selected_model_distribution` 明显单峰；
- 强制指定每个 expert 后，单个 expert 的平均误差显著最好；
- 其他 expert 并无明显异常，只是性能差。

#### 结论

这是“环境真实单峰最优”，不一定是 bug。

---

### 根因 2：SFT 先验已经塌了，RL 只是继续放大

#### 典型表现

- `train_selected_prediction_model_distribution` 本身就高度偏向某个模型；
- `routing_policy_source` 主要是 heuristic；
- SFT 后的 validation 选择分布已经单峰；
- RL 只是进一步强化。

#### 结论

这是 **SFT behavioral prior** 问题。

---

### 根因 3：runtime heuristic 泄露，把路由提前导向单模型

#### 典型表现

- `diagnostic_primary_model` 与最终 `prediction_requested_model` 高度一致；
- 模型选择行为与 feature state 的关系不强；
- 去掉 runtime 候选模型提示后，分布变得更分散。

#### 结论

这是 **runtime policy leakage** 问题。

---

### 根因 4：expert 质量或稳定性不对等

#### 典型表现

- 某些 expert 的 `prediction_tool_error_count` 更高；
- 某些 expert 更容易 fallback / default；
- 某些 expert 输出格式更不稳定；
- 强制评测时性能和稳定性明显失衡。

#### 结论

这是 **environment unfairness** 问题。

---

### 根因 5：探索不足，RL 过早进入 exploitation

#### 典型表现

- `actor_rollout_ref.actor.entropy_coeff=0`；
- 动作空间本来就已被 hard-gate 和 heuristic 提示收窄；
- RL 早期就快速固定到一个 expert。

#### 重点文件

- `examples/time_series_forecast/run_qwen3-1.7B.sh`
- 相关 RL 启动脚本与配置

#### 结论

这是 **exploration 不足** 问题。

---

### 根因 6：Turn 3 几乎只复制 Turn 2 结果，导致“选错即无救”

#### 典型表现

- `selected_forecast_exact_copy_ratio` 很高；
- `refinement_changed_ratio` 很低；
- `refinement_improved_ratio` 接近 0；
- `final_vs_selected_mse_mean` 非常接近 0。

#### 结论

这是 **refinement ineffective** 问题。

---

### 根因 7：feature analysis 覆盖率太低，导致大多数样本进入几乎相同的状态

#### 典型表现

- `analysis_coverage_ratio_mean` 偏低；
- `missing_required_feature_tool_count_mean` 偏高；
- `analysis_state_signature_distribution` 过于集中；
- 实际执行的 feature tools 数很少。

#### 结论

这是 **state information too weak** 问题。

---

## 6. 必查指标清单：要看什么、在哪看、用来判断什么

下面是 Codex 排查时必须导出的指标表。

---

### A. 用于判断是不是 SFT 先验塌缩

#### 重点指标

- `train_selected_prediction_model_distribution`
- `train_routing_row_selected_prediction_model_distribution`
- `routing_policy_source_distribution`

#### 重点文件

- SFT dataset builder 输出的 metadata / parquet 统计
- `recipe/time_series_forecast/build_etth1_sft_dataset.py`
- `recipe/time_series_forecast/build_etth1_high_quality_sft.py`

#### 判断逻辑

- 若 SFT 路由标签本身已经单峰，先怀疑 SFT；
- 若 SFT 标签较平衡，而 RL 后单峰，优先怀疑 RL / environment / leakage。

---

### B. 用于判断是不是 RL 进一步放大单峰

#### 重点指标

- `selected_model_distribution`
- `prediction_requested_model_distribution`
- `workflow_status_distribution`
- `turn_stage_distribution`

#### 重点文件

- `logs/debug/<RUN_TAG>/eval_step_aggregate.jsonl`
- `logs/debug/<RUN_TAG>/eval_step_samples.jsonl`
- `arft/trainer_validation_support.py`

#### 判断逻辑

- SFT 后就单峰：prior 问题更大；
- RL 后才单峰：reward / expert quality / exploration 更值得怀疑。

---

### C. 用于判断 expert 环境是否不对等

#### 重点指标

- `prediction_tool_error_count`
- `prediction_model_defaulted_ratio`
- 每个 expert 的平均 `orig_mse`
- 每个 expert 的平均 `orig_mae`
- 每个 expert 的长度合法率
- 每个 expert 的有效预测率

#### 重点文件

- `recipe/time_series_forecast/model_server.py`
- `artifacts/reports/rl_model_benchmark_*.json`
- 强制 expert 评测脚本 / 输出

#### 判断逻辑

- 若一个 expert 在质量和稳定性上都明显更优，策略单峰可能是合理现象。

---

### D. 用于判断 runtime heuristic 是否在泄露路由

#### 重点指标

- `diagnostic_primary_model`
- `diagnostic_runner_up_model`
- 最终 `prediction_requested_model`
- 二者一致率

#### 重点文件

- `logs/debug/<RUN_TAG>/ts_chain_debug.jsonl`
- `recipe/time_series_forecast/time_series_forecast_agent_flow.py`
- `recipe/time_series_forecast/prompts.py`

#### 判断逻辑

- 若 `diagnostic_primary_model -> prediction_requested_model` 一致率很高，说明 agent 很可能在“复读预规划”。

---

### E. 用于判断 Turn 3 是否几乎没起作用

#### 重点指标

- `selected_forecast_orig_mse_mean`
- `final_vs_selected_mse_mean`
- `refinement_delta_orig_mse_mean`
- `refinement_changed_ratio`
- `refinement_improved_ratio`
- `refinement_degraded_ratio`
- `selected_forecast_exact_copy_ratio`

#### 重点文件

- `logs/debug/<RUN_TAG>/eval_step_aggregate.jsonl`
- `arft/trainer_validation_support.py`

#### 判断逻辑

- 如果几乎总是 exact copy，且 improvement 接近 0，说明 Turn 3 基本失效。

---

### F. 用于判断 feature analysis 是否太弱

#### 重点指标

- `analysis_coverage_ratio_mean`
- `feature_tool_count_mean`
- `required_feature_tool_count_mean`
- `missing_required_feature_tool_count_mean`
- `history_analysis_count_mean`
- `analysis_state_signature_distribution`
- `required_feature_tool_signature_distribution`

#### 重点文件

- `logs/debug/<RUN_TAG>/eval_step_aggregate.jsonl`
- `arft/trainer_validation_support.py`
- `recipe/time_series_forecast/agent_flow_support.py`

#### 判断逻辑

- 如果状态几乎没有分化，路由自然容易塌到一个模型。

---

## 7. 必查日志文件

Codex 排查时优先读取这三个文件：

- `logs/debug/<RUN_TAG>/ts_chain_debug.jsonl`
- `logs/debug/<RUN_TAG>/eval_step_aggregate.jsonl`
- `logs/debug/<RUN_TAG>/eval_step_samples.jsonl`

### 最少需要输出的排查结果

1. `Turn 1` 是否只调用 feature tools；
2. `Turn 2` 是否只调用一次 `predict_time_series`；
3. `Turn 3` 是否没有非法工具调用；
4. `selected_model_distribution` 是否单峰；
5. `prediction_requested_model_distribution` 是否单峰；
6. `prediction_tool_error_count` 是否为 0；
7. `prediction_model_defaulted_ratio` 是否接近 0；
8. `selected_forecast_exact_copy_ratio` 是否过高；
9. `analysis_coverage_ratio_mean` 是否过低。

---

## 8. 建议的排查顺序（必须按顺序执行）

---

### 阶段 1：先做静态代码审计，不改代码

#### 目标

确认是否存在上文列出的高风险偏离点。

#### 必看文件

- `recipe/time_series_forecast/time_series_forecast_agent_flow.py`
- `recipe/time_series_forecast/prompts.py`
- `recipe/time_series_forecast/diagnostic_policy.py`
- `recipe/time_series_forecast/build_etth1_sft_dataset.py`
- `recipe/time_series_forecast/build_etth1_high_quality_sft.py`
- `recipe/time_series_forecast/build_etth1_rl_dataset.py`
- `recipe/time_series_forecast/model_server.py`
- `examples/time_series_forecast/run_qwen3-1.7B.sh`
- `examples/time_series_forecast/configs/etth1_ot_qwen3_gpu012.sh`

#### 输出

- 列表化给出：
  - 高风险偏离点
  - 中风险偏离点
  - 不属于关键偏离的项

---

### 阶段 2：先判定是不是 SFT 先验问题

#### 目标

从数据构造层面确认 routing label 是否已塌缩。

#### 必做

- 统计 SFT 数据集中：
  - `selected_prediction_model` 分布
  - routing rows 的 `selected_prediction_model` 分布
  - `routing_policy_source` 分布
- 若可能，计算 heuristic routing 与 offline best expert 的一致率。

#### 输出

- 一个表格：
  - model -> count / ratio
  - heuristic-best-expert agreement ratio

---

### 阶段 3：再判定是不是 RL / runtime 放大了单峰

#### 目标

比较 SFT 后与 RL 后的模型选择分布。

#### 必做

- 从 validation 聚合中提取：
  - `selected_model_distribution`
  - `prediction_requested_model_distribution`
- 从 debug 文件提取：
  - `diagnostic_primary_model`
  - `prediction_requested_model`
  - 两者一致率

#### 输出

- 判断：
  - 是 SFT 就塌；
  - 还是 RL 后才塌；
  - runtime heuristic 是否明显在控制最终选择。

---

### 阶段 4：评估 expert 环境是否公平

#### 目标

判断是不是某个 expert 本来就遥遥领先。

#### 必做

- 对同一批 ETTh1 样本做强制 expert 评测；
- 输出每个 expert 的：
  - `orig_mse`
  - `orig_mae`
  - 成功率
  - 长度合法率
  - tool error rate

#### 输出

- 一个比较表；
- 判断“单模型选择”是否具有环境合理性。

---

### 阶段 5：最后才看 Turn 3 和 exploration 问题

#### 目标

判断是不是 refinement 太弱，或 RL 探索过少。

#### 必做

- 统计：
  - `selected_forecast_exact_copy_ratio`
  - `refinement_changed_ratio`
  - `refinement_improved_ratio`
  - `refinement_degraded_ratio`
- 检查 RL 配置：
  - `actor_rollout_ref.actor.entropy_coeff`

#### 输出

- 若 `entropy_coeff=0` 且模型选择极早单峰，需要单独做 exploration ablation；
- 若 Turn 3 几乎只复制，需要单独做 refinement 放宽 ablation。

---

## 9. 建议给 Codex 的最小实验矩阵

不要一次改很多东西。先做下面 3 个最小实验。

---

### 实验 A：去掉 runtime 候选模型提示

#### 只改

- 不再把 `diagnostic_primary_model`
- `diagnostic_runner_up_model`

注入到 runtime prompt。

#### 不改

- `required_feature_tools`
- 三阶段流程
- reward
- expert 环境

#### 目的

验证是否存在 runtime policy leakage。

---

### 实验 B：SFT routing 标签改成“离线最优 expert + per-model balance”

#### 只改

- SFT 路由标签构造逻辑
- 保持 reasoning 文本和其他流程尽量不变

#### 目的

验证是否是 SFT behavioral prior 过强。

---

### 实验 C：轻微放宽 Turn 3

#### 只改

- 允许多个不连续点修正，或受限重写
- 弱化 `If unsure, choose KEEP.`

#### 不改

- Turn 3 的 strict format 协议

#### 目的

验证 Turn 3 太弱是否在放大单模型坍塌。

---

## 10. 根因判断决策树

请 Codex 在最终报告里按下面逻辑给结论。

### 情况 1

- SFT 路由分布已单峰；
- RL 只是继续单峰；

#### 结论

优先根因：**SFT behavioral prior**。

---

### 情况 2

- SFT 分布还行；
- RL 之后迅速单峰；
- 且某 expert 强制评测最好；

#### 结论

优先根因：**environment true single winner**。

---

### 情况 3

- SFT 分布还行；
- RL 之后单峰；
- `diagnostic_primary_model -> prediction_requested_model` 一致率很高；

#### 结论

优先根因：**runtime heuristic leakage**。

---

### 情况 4

- 专家性能接近；
- 但 RL 很快塌缩；
- `entropy_coeff=0`；

#### 结论

优先根因：**exploration insufficient**。

---

### 情况 5

- Turn 2 选错后，Turn 3 基本不修正；
- `selected_forecast_exact_copy_ratio` 高；
- `refinement_improved_ratio` 很低；

#### 结论

优先根因：**refinement ineffective**。

---

## 11. 建议 Codex 最终提交内容格式

请严格按下面结构给结果：

### A. 代码对齐论文审计

- `high_risk_deviations`
- `medium_risk_deviations`
- `non_blocking_items`

### B. 单模型坍塌根因判断

- `rank_1_root_cause`
- `rank_2_root_cause`
- `rank_3_root_cause`
- `evidence`

### C. 指标表

至少包含：

- SFT routing 分布
- RL routing 分布
- runtime primary-vs-final 一致率
- expert 强制评测表
- Turn 3 effectiveness 表
- analysis coverage 表

### D. 最小修改补丁

最多 3 组补丁，每组补丁都必须包含：

- 改动目标
- 改动文件
- 改动内容
- 预期影响
- 风险
- 需要重新跑哪些实验

---

## 12. 一句话执行摘要

当前 ETTh1 专项复现最该优先排查的，不是“单数据集”本身，而是：

1. **runtime 是否提前泄露了模型选择方向**；
2. **SFT routing teacher 是否已经把策略带偏**；
3. **expert 环境是否本来就不公平**；
4. **Turn 3 是否过弱，导致选错后无法纠偏**。

只有先把这四点判清，后续改 RL 或 prompt 才不会盲改。
