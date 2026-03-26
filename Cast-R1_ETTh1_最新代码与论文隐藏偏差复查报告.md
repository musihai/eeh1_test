# Cast-R1 ETTh1 最新代码与论文隐藏偏差复查报告

日期：2026-03-25  
对象：`eeh1_test-main (3).zip` 与 `2602.13802v1.pdf`

---

## 1. 本次复查要回答的问题

这次不是再查“表面上有没有三阶段 / memory / SFT+RL / GRPO”，而是专门查下面这类问题：

> **表面上看起来和论文一致，但内部真实实现方式已经和论文不一致，进而影响训练动力学或结论解释。**

我重点对照了：

- 论文 §3.1–§3.5、§4.1–§4.4、Appendix C/D/E
- 代码中的以下关键链路：
  - `recipe/time_series_forecast/time_series_forecast_agent_flow.py`
  - `recipe/time_series_forecast/prompts.py`
  - `recipe/time_series_forecast/diagnostic_policy.py`
  - `recipe/time_series_forecast/build_etth1_sft_dataset.py`
  - `recipe/time_series_forecast/build_etth1_rl_dataset.py`
  - `recipe/time_series_forecast/reward.py`
  - `arft/core_algos.py`
  - `arft/agent_flow/agent_flow.py`
  - `examples/time_series_forecast/run_qwen3-1.7B.sh`
  - `examples/time_series_forecast/configs/etth1_ot_qwen3_gpu012.sh`

---

## 2. 总体结论

### 2.1 先说结论

**最新代码已经修掉了上一轮复查里最明显的一批“表面一致、实则强 heuristic 泄露”的问题，但仍然存在几处更深层、且比 prompt 细节更严重的隐藏偏差。**

其中，当前最重要的问题不是 routing prompt 文案，而是：

### P0 级问题

1. **GRPO 的组内比较很可能仍然没有真正使用“每条 trajectory 的终局 delayed reward”**，而是错误地拿到了该 trajectory 的**首个 step 分数**。这会导致论文里“对 8 条完整 episode 做 group-relative optimization”的关键训练信号在实现里失真。

### P1 级问题

2. **Turn 1 的 planning 仍然是外置 heuristic 计划器，不是 agent 自主规划。** 这会把论文里的“agent 先做 planning 再决定信息采集策略”，实现成“外部规则先决定好工具集合，agent 只负责执行”。
3. **RL 配置与论文 Appendix C.2 仍有明显差异**，包括：
   - `norm_adv_by_std_in_grpo=False`
   - `KL=0.01`（论文写 `0.04`）
   - refinement turn 强制 `temperature=0.0`
4. **reward 虽然仍是 delayed final reward，但内部已经变成“MSE 主导 + 结构项 gated tie-break”**，不再等价于论文中“统一聚合的 multi-view reward”。
5. **curriculum 仍然是简化版三档切片**，不是论文文字描述的双轴渐进式课程。

### 2.2 同时也要说明：哪些之前的问题已经基本修掉了

最新代码里，以下问题**默认情况下已经不再是主问题**：

- runtime prompt 里直接泄露 `diagnostic_primary_model / runner_up_model`
- Turn 1 里显式出现 `looks strongest` / `distinguish X from Y` 这类候选模型暗示
- routing prompt 里显式要求按“固定 expert 模板表”选模型

也就是说：

> **上一轮最明显的“prompt 级预路由泄露”已基本收住；现在剩下的问题更偏训练链路与策略实现层。**

---

## 3. 本次复查确认“已经对齐论文”的部分

下面这些点，当前代码已经基本保留了论文主线，不能再误判成偏差：

### 3.1 三阶段 workflow 仍然在

当前 agent flow 仍然是：

- Turn 1：diagnostic / feature extraction
- Turn 2：routing / `predict_time_series`
- Turn 3：refinement / final output

对应代码：
- `recipe/time_series_forecast/time_series_forecast_agent_flow.py`
  - `_current_turn_stage()` + `_tool_schemas_for_turn()` + `_build_user_prompt()`
- `recipe/time_series_forecast/prompts.py`
  - `get_runtime_turn_info(...)`
  - `build_runtime_user_prompt(...)`

这和论文 Appendix D.2 的 stage-aware prompt construction 仍然是一致的。

### 3.2 memory-based state 仍然在

当前代码仍然有：

- `history_analysis`
- `prediction_results`
- stage-aware prompt assembly
- refinement 阶段的 recent-window truncation

对应代码：
- `recipe/time_series_forecast/time_series_forecast_agent_flow.py`
- `recipe/time_series_forecast/prompts.py:201-248`

这仍然符合论文 §3.2、Appendix D.1 / D.2 的 memory-based state 管理思路。

### 3.3 heuristic-curated SFT 本身并不违背论文

当前 builder 仍然允许 heuristic / rule-based SFT 轨迹构造。这个点**不是偏差**，因为论文 §3.5.1 本来就明确写了：

- curated decision trajectories can be constructed using forecasting heuristics and rule-based strategies.

所以：

> “SFT 里用了 heuristic trajectory” 这件事本身，不是代码偏离论文的证据。

### 3.4 训练态 `G=8` 仍然在

当前训练脚本里仍然有：

- `actor_rollout_ref.rollout.n=$ROLLOUT_N`
- train profile 默认 `RL_ROLLOUT_N=8`
- train profile 默认 `RL_TEMPERATURE=1.0`

对应代码：
- `examples/time_series_forecast/run_qwen3-1.7B.sh:274-277`
- `examples/time_series_forecast/configs/etth1_ot_qwen3_gpu012.sh:198-204`

所以从“训练意图”上看，代码仍然是在做论文意义上的 group rollout，不是退化成单条 PPO。

---

## 4. 当前仍然存在的“隐藏偏差”

---

## 4.1 P0：GRPO 的 trajectory score 取值链路仍然不对

这是本次复查里**最严重**的问题。

### 表面上看起来一致的地方

表面上看，代码有：

- `adv_estimator=grpo`
- `rollout.n=8`
- delayed final reward
- multi-turn agent rollout

所以很容易误以为：

> “论文里的 GRPO 已经按原样落地了。”

### 但内部实现并不一致

当前 `GRPO` 很可能**不是**在比较“每条完整 trajectory 的终局 reward”，而是在比较“该 trajectory 第一次出现时的 score”。

#### 证据链 1：每条 trajectory 的中间 step reward 明确是 0

代码：
- `recipe/time_series_forecast/time_series_forecast_agent_flow.py:919-926`

```python
    def _compute_intermediate_reward(...):
        return 0.0
```

而在主循环里：
- 非 final step 会把 `reward_score` 设成 `_compute_intermediate_reward(...)`
- 只有 final answer 被 accepted 时，才会调用 `_compute_final_reward(...)`

代码：
- `recipe/time_series_forecast/time_series_forecast_agent_flow.py:308-367`

这意味着一条正常三阶段轨迹往往是：

- step1：reward = 0.0
- step2：reward = 0.0
- step3：reward = final_score

#### 证据链 2：训练 batch 里每个 step 都会被展开成单独样本，并共享同一个 trajectory uid

代码：
- `arft/agent_flow/agent_flow.py:672-696`

这里每条 trajectory 的多个 step 都会进入 batch，且使用同一个 `trajectory_uid`。

#### 证据链 3：GRPO outcome advantage 只取每个 `trajectory_uid` 第一次出现的 score

代码：
- `arft/core_algos.py:144-147`

```python
for i in range(bsz):
    if trajectory_uids[i] not in request2score:
        request2score[trajectory_uids[i]] = scores[i]
        id2score[index[i]].append(scores[i])
```

这里的逻辑是：

- 某个 `trajectory_uid` 第一次出现时，才把 `scores[i]` 记进组比较分数。

但如果 batch 内 step 顺序就是 trajectory 的自然顺序（先 Turn 1，再 Turn 2，再 Turn 3），那“第一次出现的分数”大概率就是：

- Turn 1 的 `0.0`

而不是：

- Turn 3 的终局 final reward

### 为什么这是严重偏差

论文 Appendix C.2 里的 GRPO 语义是：

> 对同一个 input query 采样 8 条完整 trajectory，再按这些 trajectory 的终局 reward 做 group-relative comparison。

而当前实现如果拿的是首步分数，那它实际上更像：

> 用一组首步零分去做 group-relative comparison

这会直接导致：

- 优势信号接近无效
- route / refine 很难得到真正的 episode-level credit assignment
- RL 表面上在跑，实际上学不到论文里的 long-horizon policy

### 结论

这不是“实现口味不同”，而是 **correctness 级 bug**。  
如果只修一个地方，**先修这个。**

---

## 4.2 P1：Turn 1 planning 仍然不是 agent 自主规划，而是外置 heuristic

### 表面上看起来一致的地方

表面上代码有：

- Turn 1 diagnostic
- diagnostic plan
- feature tools
- diagnostic batches

看起来很像论文 §3.4 的 planning → feature extraction。

### 但内部实现不一致

当前并不是 agent 自己先规划“该调用哪些 tools”，而是：

- `build_diagnostic_plan(...)` 先由规则代码算出 `tool_names`
- episode 一开始就把 `required_feature_tools` 固定下来
- prompt 再要求模型“按这个 plan 执行”

#### 证据

代码：
- `recipe/time_series_forecast/diagnostic_policy.py:229-291`
  - `build_diagnostic_plan(...)`
- `recipe/time_series_forecast/time_series_forecast_agent_flow.py:188-200`
  - episode 开头直接调用 `build_diagnostic_plan(...)`
- `recipe/time_series_forecast/prompts.py:284-288`
  - `Follow the diagnostic plan and call only the feature tools exposed in this turn.`

### 为什么这是偏差

论文 §3.4 的语义是：

- agent 先做 planning
- 再决定信息采集策略
- 再调用工具

当前代码则是：

- 外部 heuristic 先做 planning
- agent 只负责在给定批次里执行工具

也就是说，当前代码里的 planning 更像：

> external heuristic planner

而不是：

> learned planning action

### 结论

这仍然是一个结构性偏差。  
它不会像 GRPO bug 那样直接让训练失真，但会让你**不能把当前结果解释成“planning 也是 agent 学出来的”。**

---

## 4.3 P1：当前 RL 配置并不是 paper-exact 的 GRPO 配置

### 4.3.1 `norm_adv_by_std_in_grpo=False`

代码：
- `examples/time_series_forecast/run_qwen3-1.7B.sh:284-285`

```bash
"algorithm.use_kl_in_reward=False"
"algorithm.norm_adv_by_std_in_grpo=False"
```

代码注释里自己也写了：
- `True` 更接近 original GRPO
- `False` 更接近 Dr.GRPO / centered outcome advantage

所以当前实现不是 paper-exact 的“标准 GRPO 优势标准化”。

### 4.3.2 KL 系数不是论文值

代码：
- `examples/time_series_forecast/configs/etth1_ot_qwen3_gpu012.sh:89-90`

```bash
export RL_KL_LOSS_COEF="${RL_KL_LOSS_COEF:-0.01}"
```

而论文 Appendix C.2 给的是：
- `KL coefficient = 0.04`

### 4.3.3 refinement turn 被强制 `temperature=0.0`

代码：
- `recipe/time_series_forecast/time_series_forecast_agent_flow.py:743-770`（`_prepare_sampling_params`）

其中 refinement 分支显式设置：

- `temperature = 0.0`
- `top_p = 1.0`

### 为什么这是偏差

论文 Appendix C.2 对 RL rollout 的描述是：

- `generation temperature = 1.0`
- 用于鼓励 diverse exploration during sampling

当前实现里，虽然 Turn 1 / Turn 2 还可能保留随机性，但 Turn 3 被强制 deterministic，会明显削弱 trajectory diversity，尤其削弱不同 refinement 策略之间的 GRPO 比较空间。

### 结论

如果你的目标是“严格论文复现”，这些都算偏差。  
如果你的目标是“先稳定训练”，它们可以视为工程折中，但必须明确写出来，不能再说成完全 paper-exact。

---

## 4.4 P1：reward 不再是 paper-exact 的统一 multi-view aggregation

### 表面上看起来一致的地方

表面上代码仍然有：

- MSE
- trend / season
- change point / structure
- format / length
- delayed final reward

所以很容易让人觉得：

> “reward 基本和论文一致。”

### 但内部实现已经变了

代码：
- `recipe/time_series_forecast/reward.py:442-458`

当前逻辑是：

1. 先算 `mse_score`
2. 再算 `structural_tie_break_gate`
3. 只有当 `gate > 0` 时，才给：
   - `change_point_score`
   - `season_trend_score`

也就是说，结构项不是始终参与，而是：

> **只有当 MSE 已经足够好时，才作为 tie-break 出场。**

### 为什么这是偏差

论文 §3.5.2 写的是：

- 多视角 reward components with predefined weights
- aggregate into a single scalar signal

更像“统一聚合”。

而当前实现更像：

- `MSE-first`
- `structure as gated tie-break`

### 结论

这不是完全背离论文，但**已经不是 paper-exact reward recipe**。

---

## 4.5 P2：curriculum 仍然是简化版三档切片，不是论文文字描述的双轴渐进过程

### 当前实现

代码：
- `recipe/time_series_forecast/build_etth1_rl_dataset.py:175-180`
- `recipe/time_series_forecast/build_etth1_rl_dataset.py:318-325`

现在的 curriculum 是：

- 先把 `teacher error` 和 `entropy` 各自分 low/medium/high
- 再取较大 rank 映射成：
  - `easy`
  - `medium`
  - `hard`
- 然后训练集切成：
  - `train_stage1`
  - `train_stage12`
  - `train_stage123`

### 为什么这和论文不完全一样

论文 §3.5.2 描述的是双轴渐进：

1. 低 intrinsic complexity + 低 prediction difficulty
2. 更高 prediction difficulty，但结构仍规整
3. 更高 stochasticity / noise

当前实现仍然用了同样的两类信号，但训练调度已经被压平为三档。

### 结论

这是**方向一致、实现简化**，不是致命 bug；  
但如果你要写“严格按论文 curriculum 复现”，这仍然不够严谨。

---

## 4.6 P2：当前 runtime prompt 已经不是 Appendix E.5 的原始提示词了

### 这次和上次不一样的地方

最新代码里，默认已经把上一轮最危险的 leakage 去掉了：

代码：
- `recipe/time_series_forecast/prompts.py:36-54`
  - `_sanitize_diagnostic_plan_text(...)`
- `recipe/time_series_forecast/prompts.py:250-268`
  - `TS_INCLUDE_DIAGNOSTIC_MODEL_HINTS` 默认 `False`

这意味着：

- `looks strongest`
- `distinguish X from Y`
- `primary_model / runner_up_model` 注入

默认都不会再进入 runtime prompt。

### 但新的问题是

当前 prompt 已经变成了一个“anti-collapse 工程版 prompt”，不再和论文 Appendix E.5 一致。

#### 例如 Turn 2 当前 prompt

代码：
- `recipe/time_series_forecast/prompts.py:291-310`

现在强调：

- `Base the decision on the maintained analysis state, not on a fixed model-to-pattern template.`
- `Compare the selected model against at least one plausible alternative...`

而论文 Appendix E.5 给的是更直接的 expert mapping：

- PatchTST 对 local temporal patterns
- iTransformer 对 cross-channel dependency
- ARIMA 对 linear trends / stable seasonality
- Chronos2 对 irregular / noisy / zero-shot

### 结论

这是一个**有意的工程修正**，不是 bug。  
但如果你的目标是“prompt 也要 appendix-faithful”，那它依然是偏差。

---

## 4.7 P2：默认 profile / checkpoint / dataset 路径仍然指向工程版 `mv1/tsfix` 链路

代码：
- `examples/time_series_forecast/configs/etth1_ot_qwen3_gpu012.sh:45-47`
- `examples/time_series_forecast/configs/etth1_ot_qwen3_gpu012.sh:106-108`

默认值是：

- `RL_CURRICULUM_DATASET_DIR=.../dataset/ett_rl_etth1_mv1`
- `RL_MODEL_PATH=.../time_series_forecast_sft_mv1_tsfix_20260324/...`
- `SFT_DATASET_DIR=.../ett_sft_etth1_runtime_ot_teacher200_mv1_stepwise_r25_tsfix`

### 为什么这也是“隐藏偏差”

因为表面上你运行的是“正式 profile”，但内部实际上默认连的是：

- 工程修复版数据
- 工程修复版 warm start
- 不是 paper-exact 的 clean reproduction pipeline

### 结论

如果你对外表述是“严格按论文复现”，那当前默认 profile 还不够干净。  
如果你对外表述是“在论文主线上的工程改造版”，那就没问题。

---

## 4.8 P2：硬件 / backbone /长度配置 与论文仍然不能说“完全一致”

### 当前代码

- `MODEL_PATH` 默认指向 `Qwen3-1.7B`
- 训练默认 3 张 GPU
- 服务默认单独占 1 张 GPU
- prompt / response 长度默认：
  - `RL_MAX_PROMPT_LENGTH=9216`
  - `RL_MAX_RESPONSE_LENGTH=3072`

### 论文

论文内部其实自己也不完全一致：

- 正文 4.1：Qwen3-8B, 4×A800 80GB
- Appendix C.1：Qwen3-1.7B, 单张 RTX 4090D
- Appendix C.2：RL max prompt / response = 8192 / 4096

### 结论

这里要分两种说法：

- **如果你说“对齐 Appendix 1.7B setting”**：当前代码更接近，但仍不是单卡 4090D，也不是完全一致长度配置。
- **如果你说“对齐正文主实验 setting”**：当前代码显然不是 8B / 4×A800。

所以这一块不是代码 bug，但**不能再模糊表述成“完全与论文一致”。**

---

## 5. 当前代码里“容易让人误判成偏差，但其实不算偏差”的点

这一节专门防止误报。

### 5.1 token 级生成 + tool parser，不等于论文违规

当前 agent 仍然是：

- 先 LLM 生成文本
- 再由 parser 解析 tool call
- 再执行工具

这意味着它不是“显式离散动作头”。

但论文并没有要求必须用独立 action head；论文强调的是：

- structured action space
- stage-aware admissible actions
- tool invocation as actions

所以：

> **当前不是严格“离散动作头采样”，但这本身不构成论文偏差。**

真正的问题不在这里，而在：

- planning 外置
- GRPO reward 链路错误
- reward / curriculum / prompt / hyperparam 偏离

### 5.2 heuristic-curated SFT 不是偏差

这个前面已经说过，不再重复。

---

## 6. 最终优先级排序

### P0：必须先修

#### 1）修 `GRPO` 的 trajectory score 取值逻辑

要保证：

- 每条 trajectory 参与 group comparison 的分数 = **该 trajectory 的终局 delayed reward**
- 而不是首步 reward

如果这个不修，后面的 prompt / reward / curriculum 再精细，RL 也仍然不是论文意义上的 GRPO。

---

### P1：紧接着修

#### 2）决定你到底要走哪条线

要先明确：

- **论文忠实复现线**
- **工程稳定增强线**

因为现在这两个目标已经开始分叉。

如果走论文忠实线，应尽量：

- 恢复更接近 Appendix E.5 的 runtime prompt
- 恢复更接近 paper 的 RL 超参
- 弱化工程化 `mv1/tsfix` 路径依赖

如果走工程稳定线，应明确承认：

- 当前 prompt / profile / reward / curriculum 已经带有工程性 anti-collapse 修正

#### 3）把 planning 从外置 heuristic 逐步收回 agent

当前至少要做到：

- heuristic 只作为 tool availability prior
- agent 自己决定是否真的调用、调用几个、以什么顺序调用

而不是像现在这样：

- heuristic 直接把 Turn 1 的工具集合定死

#### 4）恢复更接近 paper 的 GRPO 配置

至少包括：

- `norm_adv_by_std_in_grpo=True`
- `KL≈0.04`
- 慎重评估 refinement stage 强制 `temperature=0.0` 是否保留

---

### P2：如果要继续追求 paper fidelity 再做

#### 5）reward 恢复到更接近“统一 multi-view aggregation”

而不是当前：

- `MSE-first`
- `structure as gated tie-break`

#### 6）curriculum 恢复到更接近双轴渐进过程

而不是简单的：

- `easy / medium / hard`
- `stage1 / stage12 / stage123`

#### 7）清理默认 profile 的工程链路命名与依赖

让“paper reproduction profile”与“engineering profile”分开。

---

## 7. 一句话总判断

**最新代码已经修掉了上一轮最明显的 runtime heuristic 泄露问题，但仍然存在一个比 prompt 更严重的隐藏不一致：当前 GRPO 很可能没有真正按论文要求使用“每条完整 trajectory 的终局 delayed reward”做组内比较。除此之外，外置 heuristic planning、工程化 reward/curriculum、非 paper-exact 的 RL 配置，以及默认 `mv1/tsfix` profile，仍然让“表面上像论文、内部实现却并不完全一致”这个问题继续存在。**

---

## 8. 最后给你的判断建议

如果你现在要对外汇报，我建议你用下面这套说法：

### 可以说“已经基本对齐论文主线”的部分

- 三阶段 workflow
- memory-based state
- tool-augmented routing
- SFT + RL 两阶段框架
- delayed final reward
- 训练态 `G=8` rollout 设计

### 暂时不要说成“完全严格复现”的部分

- GRPO 终局 reward 取值链路
- planning 是不是 agent 自主学出的
- prompt 是否 Appendix-faithful
- reward 是否 paper-exact multi-view aggregation
- curriculum 是否 paper-exact 双轴渐进
- RL 超参数 / 硬件 / backbone 是否与论文 setting 完全一致

