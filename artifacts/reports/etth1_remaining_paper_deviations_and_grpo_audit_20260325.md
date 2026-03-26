# ETTh1 剩余论文偏差与 GRPO 审计报告

日期：2026-03-25

## 1. 结论先行

当前代码在三阶段工作流、memory-based state、delayed reward、`G=8` 组采样这些主框架上，已经基本保留了论文主线；但如果目标是“严格按论文复现”，现在仍有几处重要偏差。

其中最严重的不是 prompt 细节，而是 **GRPO 的 trajectory outcome 取值链路存在实现级错误**：

> 当前 `compute_grpo_outcome_advantage(...)` 很可能拿到的是每条 trajectory 的首个 step 分数，而不是终局 delayed reward。

由于当前中间 step 的 reward 被显式设为 `0.0`，这意味着在正常三阶段轨迹里，GRPO 组内比较极可能在比较一组 `0.0`，而不是比较最终 forecast 的 reward。  
论文里比较的是 **完整轨迹的最终 reward**，所以这些中间 `0.0` 占位分本来就不应该进入 GRPO outcome。  
这不是“论文里的 GRPO 本来就像随机采样”，而是 **当前实现没有把论文里的“按终局 trajectory reward 做组相对优势”真正落下来**。

另外，剩余的论文偏差主要有五类：

1. Turn 1 planning 仍然是外置 heuristic，不是 agent 自主规划。
2. 当前 prompt / routing wording 已被去模板化，和论文 appendix 的 Turn 2 prompt 不再一致。
3. curriculum 实现被简化成 `easy / medium / hard` 三档切片，不是论文描述的双轴渐进课程。
4. reward 的结构项被降成了 `MSE-gated tie-break`，不是论文里统一聚合的多视角 reward。
5. RL 超参数和采样细节与论文不一致，包括 `KL=0.01`、`norm_adv_by_std_in_grpo=False`、refinement turn 强制 `temperature=0.0` 等。

---

## 2. 这次对照所用证据

### 论文

- `paper.pdf`
- 重点参考：
  - §3.4 workflow / state update / adaptive routing
  - §3.5 SFT + multi-turn RL + curriculum RL
  - Appendix C.2 training protocol
  - Appendix D.1 / D.2 state update and stage-aware prompt construction
  - Appendix E.5 prompt design

### 当前代码

- `recipe/time_series_forecast/diagnostic_policy.py`
- `recipe/time_series_forecast/time_series_forecast_agent_flow.py`
- `recipe/time_series_forecast/prompts.py`
- `recipe/time_series_forecast/build_etth1_rl_dataset.py`
- `recipe/time_series_forecast/reward.py`
- `recipe/time_series_forecast/reward_metrics.py`
- `examples/time_series_forecast/run_qwen3-1.7B.sh`
- `examples/time_series_forecast/configs/etth1_ot_qwen3_gpu012.sh`
- `arft/core_algos.py`
- `arft/ray_agent_trainer.py`
- `arft/agent_flow/agent_flow.py`

### 现有日志

- `logs/debug/*/ts_chain_debug.jsonl`
- 重点观察 `trainer_reward_input`

---

## 3. 先纠正一件事：哪些点其实并不偏离论文

下面这些点，严格说 **不是** 论文偏差：

- `SFT` 使用 `200` 条 curated 样本。
  论文 Appendix C.2 明确写的是 `200 curated samples`。

- `SFT` 基于 heuristic / rule-based trajectories 构造监督。
  论文 §3.5.1 明确写了：curated decision trajectories are constructed using existing forecasting heuristics and rule-based strategies。
  所以“heuristic-curated SFT”本身并不违背论文。

- delayed reward 主要在 final step 结算。
  论文 §3.5.2 明确是 `episode-level delayed reward`。

- RL 训练时使用 `G=8` 和 `temperature=1.0` 的意图。
  当前训练脚本确实这样配置了，见：
  - `examples/time_series_forecast/run_qwen3-1.7B.sh:232-277`
  - `examples/time_series_forecast/configs/etth1_ot_qwen3_gpu012.sh:198-204`

换句话说，**真正的偏差不在“有没有 GRPO / 有没有 heuristic curated SFT / 有没有 delayed reward”**，而在更细的实现层。

---

## 4. 当前仍偏离论文的点

## 4.1 高风险偏差一：Turn 1 planning 仍然是外置 heuristic，不是 agent 学出来的

### 论文怎么写

论文 §3.4 写的是：

- 系统先基于初始 state 做 planning
- 决定要调用哪些 feature extraction tools
- 然后工具输出再写回 state
- 再基于 updated state 做 adaptive routing

这意味着 “信息采集策略” 本身属于 sequential decision policy 的一部分。

### 当前代码怎么做

当前是先由规则代码算出诊断计划，再让模型去执行：

- `recipe/time_series_forecast/diagnostic_policy.py:229-290`
  - `build_diagnostic_plan(...)` 先计算 feature snapshot
  - 再用 `_heuristic_model_scores(...)` 和 `MODEL_TOOL_REQUIREMENTS`
  - 直接决定 `tool_names`

- `recipe/time_series_forecast/time_series_forecast_agent_flow.py:188-200`
  - episode 一开始就调用 `build_diagnostic_plan(...)`
  - 把 `required_feature_tools` 固定下来

- `recipe/time_series_forecast/prompts.py:278-288`
  - diagnostic prompt 直接要求：
  - `Follow the diagnostic plan and call only the feature tools exposed in this turn.`

### 结论

这会把论文里的：

- `agent plans tool usage`

退化成：

- `external heuristic plans tool usage`
- `agent only executes the provided plan`

这不是 prompt 小问题，而是 **Planning module 仍未真正 agent 化**。

---

## 4.2 高风险偏差二：当前 GRPO 实现没有按论文使用终局 trajectory reward 做组内比较

这是本轮发现的最重要问题。

### 论文怎么写

论文 §3.1 和 §3.5.2 / Appendix C.2 写得很清楚：

- 每个 input query 采样 `G=8` 条 trajectory
- 用这些 trajectory 的相对表现来算 GRPO advantage
- reward 是 `episode-level delayed reward`
- 作用在完整 forecasting episode 结束后

也就是说，论文的 GRPO 比较对象是：

> 同一个输入下的多条完整决策轨迹，它们的终局 reward 谁更好。

### 当前代码的关键链路

#### A. 中间 step reward 被显式设为 `0.0`

- `recipe/time_series_forecast/time_series_forecast_agent_flow.py:919-926`
  - `_compute_intermediate_reward(...)` 直接 `return 0.0`

- `recipe/time_series_forecast/time_series_forecast_agent_flow.py:357-367`
  - 非 final step 的 `reward_score` 来自 `_compute_intermediate_reward(...)`

#### B. 当前 agent flow 框架还要求每个 step 都必须有 `reward_score`

- `arft/agent_flow/agent_flow.py:446`
  - `assert step.reward_score is not None`

这意味着当前实现并不是“只在 terminal step 才有 outcome reward”，而是：

- 每一步都必须产出一个 `reward_score`
- 非 terminal step 只能用 `0.0` 占位

这和论文的 trajectory-level delayed reward 语义本身就不一致。

#### C. agent flow 在 postprocess 时按 step 顺序写入 trajectory

- `arft/agent_flow/agent_flow.py:672-696`
  - 每条 trajectory 的 step 按顺序写入 batch
  - 每条 trajectory 的多个 step 共用同一个 `trajectory_uid`

#### D. 但 GRPO 只取每个 `trajectory_uid` 第一次出现的 score

- `arft/core_algos.py:144-147`
  - `if trajectory_uids[i] not in request2score:`
  - 只有第一次出现时才把 `scores[i]` 放进 `id2score[index[i]]`

### 这意味着什么

如果一条正常 episode 是：

1. diagnostic step -> reward `0.0`
2. routing step -> reward `0.0`
3. final refinement step -> reward `0.43`

那么按当前实现，`compute_grpo_outcome_advantage(...)` 极可能记录的是：

- 该 trajectory 的第一个 step score = `0.0`

而不是：

- 终局 delayed reward = `0.43`

### 日志证据

现有 debug 日志里的 `trainer_reward_input` 也非常吻合这个推断：

- `logs/debug/mv1hintdrop_turn3relax_fixedref32_20260325_162554/ts_chain_debug.jsonl`
- `logs/debug/mv1ref_hintdrop_val32_20260325_160021/ts_chain_debug.jsonl`
- `logs/debug/mv1r50_fixedref32_20260325_170337/ts_chain_debug.jsonl`

多处都出现固定模式：

- `reward_scores_head = [0.0, 0.0, final_score, 0.0, 0.0, final_score, ...]`

这说明每条 trajectory 的三个 step 确实是：

- 前两个 step 为 `0.0`
- 最后一步才是非零终局 reward

结合 `arft/core_algos.py:144-147` 的“只取第一次出现”逻辑，可以合理推出：

> 当前 GRPO 在正常三阶段 episode 上，大概率没有用到最终 reward 做组比较。

### 结论

这不是“更像随机采样”这么轻的问题。  
更准确地说：

> 当前实现很可能让 GRPO 组优势基线建立在首步零分上，而不是论文要求的终局 delayed reward 上。

这会直接导致：

- 组内相对优势接近全零
- actor 几乎得不到有效的 route / refine credit assignment
- RL 看起来像在“采样很多轨迹，但学不到有意义的偏好”

这是 **P0 级别 correctness bug**。

进一步说，**中间 reward 不该只是“设成 0”**，而应该从 GRPO outcome 里移除。  
如果要和论文对齐，至少要满足下面之一：

1. 非 terminal step 不再写入 GRPO 可见的 `reward_score`；
2. 或 advantage 计算时，显式只取每条 `trajectory_uid` 的最后一个 step reward；
3. 或先按 trajectory 聚合出唯一终局 outcome，再做组内比较。

---

## 4.3 高风险偏差三：当前不是原始 GRPO，而是更接近 Dr.GRPO 式 centered outcome

### 论文怎么写

论文 Appendix C.2 的文字是标准 GRPO 表述：

- 8 trajectories are sampled for each input query
- compute the relative advantage

### 当前代码怎么配

- `examples/time_series_forecast/run_qwen3-1.7B.sh:285`
  - `algorithm.norm_adv_by_std_in_grpo=False`

- `arft/core_algos.py:124-126`
  - 注释明确写了：
  - `True` 是 original GRPO
  - `False` 更像 `Dr.GRPO`

### 结论

当前训练链路并不是论文原文意义上的“原始 GRPO 标准化优势”，而是：

- group-centered outcome advantage
- 但不做 std normalization

这本身未必一定更差，但 **它不是论文里的那版 GRPO**。

---

## 4.4 高风险偏差四：refinement turn 被强制 `temperature=0.0`，和论文整条 episode 的探索设定不一致

### 论文怎么写

论文 Appendix C.2：

- generation temperature = `1.0`
- 用于鼓励 diverse exploration during sampling

### 当前代码怎么做

- `recipe/time_series_forecast/time_series_forecast_agent_flow.py:757-769`
  - refinement stage 强制：
  - `params["temperature"] = 0.0`
  - `params["top_p"] = 1.0`

### 结论

当前 rollout 并不是“整条 trajectory 都按 `temperature=1.0` 探索”。
实际更接近：

- Turn 1 / Turn 2 可能保留训练温度
- Turn 3 final answer 强制 deterministic decoding

这会削弱 episode-level trajectory diversity，尤其是让 GRPO 很难比较不同 refinement 策略。

---

## 4.5 中风险偏差一：reward 的结构项被降成了 MSE-gated tie-break，而不是始终参与统一聚合

### 论文怎么写

论文 §3.5.2 写的是多视角 reward 统一聚合：

- normalized + log-transformed MSE
- trend consistency
- seasonal consistency
- turning-point alignment
- format validity
- output length consistency

### 当前代码怎么做

- `recipe/time_series_forecast/reward.py:443-453`
  - 先算 `mse_score`
  - 再算 `structural_tie_break_gate = compute_structural_tie_break_gate(norm_mse)`
  - 只有 gate > 0 时，才加：
    - `change_point_score`
    - `season_trend_score`

- `recipe/time_series_forecast/reward_metrics.py:19-20`
  - `STRUCTURAL_TIE_BREAK_SCALE = 0.2`
  - `STRUCTURAL_TIE_BREAK_MAX_NORM_MSE = 0.5`

### 结论

当前实现把论文里的结构 reward 项变成了：

- “只有 MSE 已经够好时才出场的弱 tie-break”

这比论文原文的“统一多视角聚合”更保守。  
它是合理工程化，但 **不是 paper-exact reward**。

---

## 4.6 中风险偏差二：curriculum 被简化成三档切片，不是论文描述的双轴渐进式课程

### 论文怎么写

论文 §3.5.2 的 curriculum 是双轴：

- 轴 1：reference teacher error
- 轴 2：normalized permutation entropy

并且训练顺序是：

1. 低复杂度 + 低预测难度
2. 更高预测难度，但结构仍规整
3. 更高随机性 / 噪声

### 当前代码怎么做

- `recipe/time_series_forecast/build_etth1_rl_dataset.py:175-180`
  - `_resolve_difficulty_stage(...)` 只是取 `error_band` 和 `entropy_band` 的最大 rank
  - 直接映射成 `easy / medium / hard`

- `recipe/time_series_forecast/build_etth1_rl_dataset.py:319-325`
  - `train_stage1 = easy`
  - `train_stage12 = easy + medium`
  - `train_stage123 = full`

### 结论

当前 curriculum 仍然用了论文的两类难度信号，但它的 staged schedule 是：

- 三档切片版 curriculum

而不是论文文字描述的：

- “先低误差低熵，再高误差低熵，再高熵噪声”的更细双轴渐进过程

所以这部分是 **方向对了，但调度形态没完全复原**。

---

## 4.7 中风险偏差三：当前 runtime Turn 2 prompt 不再等价于论文 appendix 的专家映射提示

### 论文 appendix E.5

论文的 Turn 2 prompt 明确写了：

- `PatchTST: local temporal patterns with long-range dependencies`
- `iTransformer: cross-channel dependency dominant`
- `ARIMA: linear trends and stable seasonality`
- `Chronos2: irregular, noisy, zero-shot scenarios`

### 当前代码

- `recipe/time_series_forecast/prompts.py:304-310`
  - 强调：
  - `Base the decision on the maintained analysis state, not on a fixed model-to-pattern template.`

虽然 tool schema 里仍保留了模型描述：

- `recipe/time_series_forecast/prompts.py:320-325`

但 runtime user prompt 已不再等价于论文 appendix 的 Turn 2 文案。

### 结论

这是 **有意为之的 anti-collapse 改动**，不是 accidental bug。  
但如果目标是“严格按论文 prompt 复刻”，那它仍然是偏差。

换句话说：

- 它可能更利于去 heuristic leak
- 但它不再是 appendix-faithful prompt

---

## 4.8 中风险偏差四：当前 formal profile 默认仍指向 `mv1` 工程链路，而不是 paper-named 数据链路

### 当前 profile

- `examples/time_series_forecast/configs/etth1_ot_qwen3_gpu012.sh:45`
  - `RL_CURRICULUM_DATASET_DIR = dataset/ett_rl_etth1_mv1`

- `examples/time_series_forecast/configs/etth1_ot_qwen3_gpu012.sh:106`
  - `SFT_DATASET_DIR = dataset/ett_sft_etth1_runtime_ot_teacher200_mv1_stepwise_r25_tsfix`

### 结论

这说明当前正式 profile 默认跑的是：

- `mv1 / tsfix` 工程版链路

而不是 repo 内更接近 paper naming 的：

- `paper_same2`
- `paper_aligned_ot_curriculum_same2`

这不一定说明 `mv1` 错，但如果对外表述是“paper exact reproduction”，那么 profile 默认值本身仍不够严谨。

---

## 4.9 低风险说明：`routing_only / refinement_only` builder 模式是研究工具，不是论文训练配方

当前 builder 已加入：

- `routing_only`
- `refinement_only`

见：

- `recipe/time_series_forecast/build_etth1_sft_dataset.py:2151-2159`

这类模式对定位坍塌根因很有用，但它们不是论文中声明的正式两阶段训练配方。  
因此：

- 用它们做 ablation 是合理的
- 但不能把它们当成“论文本来就是这么训的”

---

## 5. 论文里的 GRPO 到底是怎么做的

按论文 Appendix C.2，Cast-R1 的 RL 是：

1. 对每个 input query 采样 `G=8` 条 trajectory。
2. 用这些 trajectory 的 group scores 计算相对优势。
3. reward 是 episode-level delayed reward。
4. generation temperature = `1.0`，用于鼓励探索。
5. KL coefficient = `0.04`。
6. RL global batch size = `64`。
7. max prompt / response length = `8192 / 4096`。

所以论文里的 GRPO 本质上是：

> 同一输入下，采样 8 条完整 episode，等 episode 结束后按最终 reward 做组内相对比较，再更新策略。

### 5.1 论文里采样的不是“随机动作”，而是 policy-induced decision trajectories

论文 §3.1 明确写了：

- 在每个 step，agent 观察 state `s_k`
- 然后从 `structured action space` 里选择 action `a_k`
- action 可能是 feature extraction、model invocation、update prediction、refine prediction

论文 §3.4 / Appendix D.2 又说明：

- 不同 turn 会限制可行动作集合
- Turn 1 只允许诊断类动作
- Turn 2 只允许 forecasting model invocation
- Turn 3 只允许 reasoning / refinement / final output

所以论文里 “8 trajectories are sampled for each input query” 的最合理含义是：

> 从当前 policy 在这个 structured action space 上诱导出的轨迹分布里，采样 8 条完整决策轨迹。

它不是：

- 对所有动作做 uniform random 采样
- 也不是穷举动作空间后随机挑一个 action id

而是：

- stochastic rollout
- + group-relative ranking
- + delayed reward credit assignment

### 5.2 论文没有要求“显式 action-id 采样器”，但要求 sampled object 是完整决策轨迹

论文文本只说 `sampled decision trajectories`，并没有要求实现上必须写成：

- 一个单独的离散 action head
- 一个显式 high-level action categorical sampler

因此，只要实现满足：

- 采到的是完整多轮决策轨迹
- action 受 structured action space / stage constraint 约束
- 组比较用的是终局 trajectory reward

就属于论文语义上的 GRPO rollout。

---

## 6. 当前代码里的 GRPO 实际上在做什么

## 6.1 训练脚本的意图其实是对的

训练脚本确实有这些 paper-consistent 设置：

- `algorithm.adv_estimator=grpo`
  - `examples/time_series_forecast/run_qwen3-1.7B.sh:232`

- `actor_rollout_ref.rollout.n=$ROLLOUT_N`
  - `examples/time_series_forecast/run_qwen3-1.7B.sh:277`

- train profile 默认 `RL_ROLLOUT_N=8`
  - `examples/time_series_forecast/configs/etth1_ot_qwen3_gpu012.sh:203`

- train profile 默认 `RL_TEMPERATURE=1.0`
  - `examples/time_series_forecast/configs/etth1_ot_qwen3_gpu012.sh:92`

所以从 launch config 看，**它本来不是随机单采样 PPO**。

## 6.2 当前代码也不是“uniform random sampling”，而是 token-level policy rollout

当前 rollout 的实际采样链路是：

- `arft/agent_flow/agent_flow.py:565-575`
  - 从 rollout config 读 `temperature` / `top_p`

- `recipe/time_series_forecast/time_series_forecast_agent_flow.py:232-255`
  - 每一 turn 先构造当前 stage 的 prompt 和 tool schemas
  - 再调用 `server_manager.generate(...)`

- `recipe/time_series_forecast/time_series_forecast_agent_flow.py:286-305`
  - 把生成出来的 token 序列交给 `tool_parser`
  - 解析出 tool calls / final answer

也就是说，当前代码采样的并不是“随机 action id”，而是：

> 在 stage-aware action constraint 下，对 LLM response token 序列做 stochastic rollout，再由 parser 把 response 映射成 action。

从语义上说，这仍然属于：

- policy rollout
- induced action trajectory sampling

而不是“完全随机瞎试动作”。

## 6.3 但它和论文仍有四层关键差异

### 问题 A：终局 reward 没被正确喂进 GRPO 组比较

见 §4.2。  
这是当前最严重的问题。

### 问题 B：当前是 `norm_adv_by_std_in_grpo=False`

- `examples/time_series_forecast/run_qwen3-1.7B.sh:285`

这让它更接近：

- centered outcome
- 非原始 GRPO 标准化优势

### 问题 C：trajectory 的采样自由度被额外缩窄

当前代码并不是单纯在 paper-style structured action space 上采样，而是还叠加了额外工程约束：

- `recipe/time_series_forecast/time_series_forecast_agent_flow.py:188-200`
  - Turn 1 要执行的 feature tools 先由 `diagnostic_policy.py` 决定

- `recipe/time_series_forecast/prompts.py:278-288`
  - diagnostic stage 明确要求跟随现成 diagnostic plan

- `recipe/time_series_forecast/time_series_forecast_agent_flow.py:757-769`
  - refinement stage 强制 `temperature=0.0`

这意味着当前 trajectory sampling 虽然不是 uniform random，但也不是论文里那种更“由 policy 自主决定 planning / routing / refine”的采样。

### 问题 D：debug / val-only 经常是 `n=1` 且 `temperature=0.0`

- `examples/time_series_forecast/configs/etth1_ot_qwen3_gpu012.sh:127-144`
  - `val_only` 默认 `RL_ROLLOUT_N=1`

- `examples/time_series_forecast/configs/etth1_ot_qwen3_gpu012.sh:94`
  - `RL_VAL_TEMPERATURE=0.0`

这意味着很多 debug run 只是在做：

- deterministic validation
- 不是训练态的 GRPO rollout

所以：

- 不能拿 `val_only` 的行为去判断“训练时是不是做了 G=8 的 GRPO”

---

## 7. 为什么你会感觉“更像随机采样”

你的直觉并不空穴来风，但更准确的表述应该是：

> 当前实现看起来像“采了很多条，但像没在按动作学”，不是因为它在做 uniform random action sampling，而是因为它没有把论文里最关键的“终局 trajectory reward 组比较”跑通，同时又把 planning / refinement 的自由度额外压窄了。

造成这种感觉的原因至少有四个：

1. **如果首步零分被拿去做组比较，优势会接近全零。**
2. **中间 `0.0` 占位 reward 被保留下来，本身就破坏了论文的 trajectory-level outcome 语义。**
3. **`norm_adv_by_std_in_grpo=False` 进一步削弱了原始 GRPO 的归一化对比。**
4. **Turn 3 deterministic decoding 限制了完整 episode 的探索差异。**
5. **当前 warm start / heuristic scaffold 本来就很强，rollout 多样性已经被压窄。**

所以主问题不是：

- “论文的 GRPO 其实就是 random sampling”

而是：

- “我们当前的实现没有把论文的 GRPO 关键信号链跑通”

---

## 8. 优先级排序

### P0：必须先修

1. 修 GRPO episode score 取值逻辑  
   当前必须改成：每条 trajectory 参与 group comparison 的是 **终局 delayed reward**，不是首个 step reward。

2. 从 GRPO outcome 中删除所有中间 reward 占位  
   论文比较的是最终轨迹 reward，所以 non-terminal step 的 `0.0` 不应再进入 GRPO 可见分数。

### P1：紧随其后

3. 恢复 paper-faithful 的 GRPO 配置
   - `norm_adv_by_std_in_grpo=True`
   - `KL=0.04`
   - 尽量靠近论文的 batch / response length

4. 决定是否要 paper-faithful prompt
   - 如果目标是严格复现论文，应恢复 appendix 风格 Turn 2 prompt
   - 如果目标是先止损坍塌，则保留去模板化 prompt，但要明确这是“偏离论文的工程修正”

5. 把 planning 从外置 heuristic 逐步收回 agent

6. 若要继续追 paper fidelity，再细化 curriculum 调度

---

## 9. 最终判断

如果现在只回答两个问题：

### 问题一：我们目前还有哪些点偏离论文？

最重要的剩余偏差是：

1. Turn 1 planning 仍是外置 heuristic。
2. 当前 runtime prompt 已不再完全等价于论文 appendix。
3. curriculum 是简化版三档切片，不是论文的双轴渐进式课程。
4. reward 结构项被降成 MSE-gated tie-break。
5. RL 超参数不是 paper-exact。
6. 当前 formal profile 默认跑的是 `mv1` 工程链路。

### 问题二：论文里的 GRPO 是怎么做的，我们现在的问题是什么？

论文里的 GRPO 是：

- 对同一输入采样 `8` 条完整 trajectory
- 用终局 delayed reward 做组内相对优势
- 这些 trajectory 是 policy 在 structured action space 上诱导出来的多轮决策轨迹，不是 uniform random action 样本

当前代码的问题不是“只是随机采样”，而是更严重的：

> 当前 GRPO 的组比较很可能拿错了每条 trajectory 的 reward，而且还把中间 `0.0` 占位 reward 一起送进了 outcome 链路，导致它没有真正按论文方式用终局 trajectory reward 做 group-relative optimization。

这应该被视为 **当前 RL 复现链路的首要修复点**。

## 10. 与《最新代码与论文隐藏偏差复查报告》的交叉核验

本节专门回答：`Cast-R1_ETTh1_最新代码与论文隐藏偏差复查报告.md` 里提到的点，哪些在当前代码中确实存在，哪些需要收窄，哪些不应继续算作“当前隐藏偏差”。

### 10.1 核验后确认属实的问题

下面这些判断，经代码复查后确认成立，且仍应保留为当前主审计结论：

1. **GRPO 组内比较没有稳定对齐到“每条 trajectory 的终局 delayed reward”**  
   这一点是对的，而且是当前最严重的问题。  
   代码链路仍然是：
   - `recipe/time_series_forecast/time_series_forecast_agent_flow.py:919-926`
     - 中间 step reward = `0.0`
   - `arft/agent_flow/agent_flow.py:689-694`
     - 每个 step 都会被打成 reward tensor
   - `arft/core_algos.py:144-147`
     - `compute_grpo_outcome_advantage(...)` 只取每个 `trajectory_uid` 第一次出现的 score

2. **Turn 1 planning 仍然是外置 heuristic，而不是 agent 自主 planning**  
   这一点也成立。  
   当前 runtime 仍然在 episode 开头调用：
   - `recipe/time_series_forecast/diagnostic_policy.py:229-291`
   - `recipe/time_series_forecast/time_series_forecast_agent_flow.py:188-200`
   并在 prompt 中要求：
   - `recipe/time_series_forecast/prompts.py:284-288`
     - `Follow the diagnostic plan and call only the feature tools exposed in this turn.`

3. **RL 配置仍非 paper-exact**  
   这条也成立，主要包括：
   - `examples/time_series_forecast/run_qwen3-1.7B.sh:285`
     - `algorithm.norm_adv_by_std_in_grpo=False`
   - `examples/time_series_forecast/configs/etth1_ot_qwen3_gpu012.sh:90`
     - `RL_KL_LOSS_COEF=0.01`
   - `recipe/time_series_forecast/time_series_forecast_agent_flow.py:757-769`
     - refinement turn 强制 `temperature=0.0`

4. **reward 目前是 `MSE-first + structural gated tie-break`，不是 paper-exact 的统一聚合**  
   这条成立。  
   当前门控逻辑仍在：
   - `recipe/time_series_forecast/reward.py:443-453`
   - `recipe/time_series_forecast/reward_metrics.py:19-20`

5. **curriculum 仍是三档切片简化版，而不是论文文字描述的双轴渐进过程**  
   这条成立。  
   当前核心映射仍是：
   - `recipe/time_series_forecast/build_etth1_rl_dataset.py:175-180`
   - `recipe/time_series_forecast/build_etth1_rl_dataset.py:318-325`

6. **默认 formal profile 仍然连到 `mv1/tsfix` 工程链路**  
   这条也成立。  
   当前默认值仍是：
   - `examples/time_series_forecast/configs/etth1_ot_qwen3_gpu012.sh:106-108`
   - `examples/time_series_forecast/configs/etth1_ot_qwen3_gpu012.sh:198-204`

### 10.2 核验后认为“基本属实，但表述需要收窄”的问题

1. **“runtime Turn 2 prompt 仍偏离论文 appendix”是对的，但它现在更像 paper-fidelity 偏差，而不是当前坍塌主因。**  
   当前 Turn 2 prompt 确实已经不再等价于 Appendix E.5：
   - `recipe/time_series_forecast/prompts.py:304-310`
   但这属于有意的 anti-collapse 工程修正。  
   因此更准确的表述应是：
   - 它确实偏离 appendix prompt
   - 但它不是当前最严重的 correctness 问题
   - 当前首要问题仍是 GRPO outcome 取值链路

2. **“当前仍是 delayed reward”只能算语义上成立，训练实现层仍有关键缺口。**  
   也就是说：
   - final reward 仍然只在末步结算，这一点没有问题
   - 但 non-terminal `0.0` placeholder 仍被送入 GRPO 可见链路，这一点与论文不一致
   所以更准确的说法应是：
   - `reward semantics` 仍然是 delayed final reward
   - `GRPO implementation` 还没有真正只比较 terminal trajectory outcome

3. **“当前硬件 / backbone /长度配置与论文不一致”是对的，但这更像实验 setting 差异，不是算法 correctness bug。**  
   这一点在对外陈述时要保留，但优先级应低于：
   - GRPO outcome 取值
   - planning 外置
   - reward / curriculum / sampling 配置偏差

### 10.3 核验后不应继续算作“当前隐藏偏差”的问题

1. **token-level 生成 + tool parser 本身不构成论文偏差**  
   论文要求的是：
   - structured action space
   - stage-aware admissible actions
   - sampled decision trajectories
   并没有要求必须实现独立离散 action head。  
   因此当前：
   - LLM token sampling
   - tool parser 映射成 action
   这个实现方式本身不违规。

2. **heuristic-curated SFT 本身不构成论文偏差**  
   论文 §3.5.1 明确允许用 forecasting heuristics / rule-based strategies 构造 curated trajectories。  
   因此：
   - `heuristic-curated SFT` 不是偏差
   - 真正的偏差在于 runtime RL 仍把 heuristic planning 外置到了 agent 之外

3. **训练态 `G=8` 本身没有丢**  
   当前训练脚本仍有：
   - `examples/time_series_forecast/run_qwen3-1.7B.sh:277`
   - `examples/time_series_forecast/configs/etth1_ot_qwen3_gpu012.sh:203`
   所以不能把当前问题归因为“训练时已经退化成单条 rollout”。

4. **上一轮最明显的 runtime heuristic 泄露，默认链路下已不是主问题。**  
   这部分在最新代码里确实已显著收敛：
   - `recipe/time_series_forecast/prompts.py:260-268`
     - `diagnostic_primary_model / runner_up_model` 默认不再注入
   - `recipe/time_series_forecast/diagnostic_policy.py:221-226`
     - rationale 已改成 feature-oriented 文本
   因此它不应继续被列为“当前最高优先级隐藏偏差”。

### 10.4 交叉核验后的总判断

`Cast-R1_ETTh1_最新代码与论文隐藏偏差复查报告.md` 的主判断 **大体是对的**，但要做两点收敛：

1. 它指出的最高优先级问题里，**GRPO terminal reward 取值错误** 与 **planning 外置** 确实是现在最应该保留的核心结论。
2. 它列出的 prompt / profile / setting 偏差很多都成立，但这些应被视为：
   - `paper-fidelity gap`
   而不是：
   - `当前训练 correctness 的首要破坏项`

---

## 11. 不偏离论文的修改方案

下面给出一套 **paper-exact** 的修改顺序。这里的目标不是继续做 anti-collapse 工程修补，而是把实现尽可能改回论文所描述的训练与采样语义。

### 11.0 先明确：采样方式要改成论文同款，不是“随机乱采样”

如果要求和论文 **一模一样**，那么需要对齐的不是“是否显式写了动作 id 采样器”，而是下面这个采样语义：

- 对同一个输入 query，训练时采样 `G=8` 条完整 trajectory。
- 每条 trajectory 都是在 `structured action space` 上，由当前 policy 按 state 逐步作出决策。
- Turn 1 是否调用哪些 feature tools，必须由 policy 决定。
- Turn 2 选择哪个 forecasting model，必须由 policy 决定。
- Turn 3 是否 `KEEP` 或做怎样的 refinement，也必须由 policy 在训练态采样决定。
- 组内比较只比较这 8 条 trajectory 的 **terminal delayed reward**。

这意味着论文同款做法不是：

- 在所有动作里做 `uniform random` 抽样；
- 也不是外部 heuristic 先把工具计划好，然后模型只在一个被裁剪过的轨迹空间里补全文本；
- 更不是前两步有随机性、最后一步 deterministic，再拿中间 `0.0` 占位分去做 GRPO。

#### 论文同款采样例子

同一个输入样本 `x`，训练时采样 8 条 trajectory，其中两条可能是：

- trajectory A
  1. Turn 1: policy 决定先调 `extract_basic_statistics` 和 `extract_event_summary`
  2. Turn 2: policy 依据更新后的 state 选择 `itransformer`
  3. Turn 3: policy 决定做一次小幅 local refine
  4. terminal reward = `0.63`

- trajectory B
  1. Turn 1: policy 只调 `extract_basic_statistics`
  2. Turn 2: policy 选择 `arima`
  3. Turn 3: policy 选择 `KEEP`
  4. terminal reward = `0.22`

GRPO 比较的是：

- `0.63`
- `0.22`
- 以及同组其它 6 条 trajectory 的最终 reward

而不是比较：

- Turn 1 的 reward
- Turn 2 的 reward
- 或某个外部 planner 先验分数

#### 当前代码的实际方式

当前代码更接近：

- 外部 heuristic 先通过 `build_diagnostic_plan(...)` 决定 Turn 1 工具集合；
- 模型在这个已裁剪空间里按 token 采样；
- Turn 3 训练态还会被强制 `temperature=0.0`；
- GRPO 组比较还可能拿到了每条 trajectory 的首步 `0.0`，而不是 terminal reward。

所以当前问题不该表述成“我们是在随机采样动作”，更准确的是：

- **我们没有按论文要求采样完整、自由的 policy-induced trajectories**
- **并且没有把 terminal trajectory reward 正确送进 GRPO**

#### 因此修改目标必须写死为

1. 不改成 `uniform random action sampling`。
2. 改成 **policy over structured action space 的 complete-trajectory sampling**。
3. 训练态三轮都允许参与 trajectory sampling。
4. GRPO 只看 terminal trajectory reward。

### 11.1 P0：先修 GRPO，只让组比较看到 terminal trajectory reward

这是必须先做的修改；不修这一条，后续 prompt、reward、curriculum 调整都无法正确解释。

#### 需要修改的文件

- `arft/core_algos.py`
- `arft/ray_agent_trainer.py`
- `arft/agent_flow/agent_flow.py`
- `recipe/time_series_forecast/time_series_forecast_agent_flow.py`

#### 如何修改

1. **`arft/ray_agent_trainer.py`**  
   修改 `compute_grpo_outcome_advantage(...)` 的调用，把：
   - `step_indices=valid_data.non_tensor_batch["step_indices"]`
   一并传进去。  
   当前 trainer 已经有这个字段，只是 GRPO 分支没有使用。

2. **`arft/core_algos.py`**  
   重写 `compute_grpo_outcome_advantage(...)` 的 trajectory score 提取逻辑，不能再用：
   - `第一次出现的 score`
   而要改成：
   - `同一 trajectory_uid 中 step_indices 最大的那个 step 的 score`
   - 或显式使用 `terminal_step_mask`

   计算顺序应改为：
   - 先按 `trajectory_uid` 聚合出唯一的 terminal outcome
   - 再按 `uid` 做组内 mean/std
   - 再把得到的 trajectory-level advantage 回填到该 trajectory 的全部 step token 上

   这样才和论文的：
   - `sample G trajectories`
   - `compare terminal rewards`
   - `assign group-relative advantage`
   一致。

3. **`recipe/time_series_forecast/time_series_forecast_agent_flow.py`**  
   把 non-terminal step 的 reward 从“训练信号”里移除。  
   如果要完全贴齐论文语义，建议直接：
   - 废弃 `_compute_intermediate_reward()` 的 GRPO 用途
   - non-terminal step 不再携带 outcome reward
   - terminal step 才写入唯一 final `reward_score`

   注意这里的目标不是“中间 reward 改成别的数”，而是：
   - **中间 reward 不参与 GRPO outcome**
   - **每条 trajectory 只有一个 terminal outcome**

4. **`arft/agent_flow/agent_flow.py`**  
   现有：
   - `assert step.reward_score is not None`
   与论文的 trajectory-level delayed reward 不一致。  
   修改方向应是：
   - 允许 non-terminal step 没有 outcome reward
   - 或引入独立的 `terminal_reward_score` / `terminal_step_mask`
   - batch 打包时只让 terminal outcome 进入 GRPO outcome 链路

#### 这一步改完后的验证标准

- 同一条三阶段 trajectory 的 GRPO group score 只能等于末步 final reward。
- `trainer_reward_input` 不应再以“每步一个分数”的方式误导 GRPO outcome。
- 对一个人工构造的三步 episode，若奖励为 `[0.0, 0.0, 0.43]`，GRPO 组比较使用的必须是 `0.43`。
- 对同一个 query 采样 `8` 条 trajectory 时，组内比较对象必须是 `8` 个 terminal rewards，而不是 `24` 个 step rewards。

### 11.2 P1：把 Turn 1 planning 收回 agent，而不是由 heuristic 预先决定

#### 需要修改的文件

- `recipe/time_series_forecast/diagnostic_policy.py`
- `recipe/time_series_forecast/time_series_forecast_agent_flow.py`
- `recipe/time_series_forecast/prompts.py`

#### 如何修改

1. **runtime 链路不要再调用 `build_diagnostic_plan(...)` 来决定必调工具集合。**  
   当前这段逻辑应从 RL runtime 主路径移除：
   - `recipe/time_series_forecast/time_series_forecast_agent_flow.py:188-200`

2. **Turn 1 应直接暴露完整 diagnostic tool set。**  
   `time_series_forecast_agent_flow.py` 中 Turn 1 的 tool schema 应改成：
   - 给 agent 完整的 feature extraction action set
   - agent 自己决定调哪些 tool、一次调几个、按什么顺序调
   - 不再预先生成 `required_feature_tools`
   - 不再把 heuristic 生成的 tool batches 当作 runtime 行动边界

3. **`prompts.py` 的 Turn 1 prompt 改成 paper-style planning instruction。**  
   不再写：
   - `Follow the diagnostic plan and call only the feature tools exposed in this turn.`
   而改成更接近论文 §3.4 / Appendix D.2 的语义：
   - 先基于当前 state 制定信息采集策略
   - 再调用你认为必要的 feature tools
   - 工具输出将更新 state，并用于后续 routing

4. **`diagnostic_policy.py` 保留为 offline SFT/ablation 辅助工具，不再作为 runtime action controller。**  
   这不偏离论文，因为：
   - heuristic-curated SFT 是允许的
   - 外置 heuristic 决定 runtime planning 则不是

#### 这一步改完后的验证标准

- runtime Turn 1 不再预先写死 `required_feature_tools`。
- 同一输入在不同 rollout 下，Turn 1 允许出现不同但合理的 tool selection trajectory。
- 诊断工具调用的多样性来自 policy，而不是来自外置 heuristic planner。
- 对同一个 query 的 8 条 trajectory，Turn 1 应允许出现不同的工具组合与调用顺序。

### 11.3 P1：恢复 paper-faithful 的 GRPO 训练配置

#### 需要修改的文件

- `examples/time_series_forecast/run_qwen3-1.7B.sh`
- `examples/time_series_forecast/configs/etth1_ot_qwen3_gpu012.sh`
- `recipe/time_series_forecast/time_series_forecast_agent_flow.py`

#### 如何修改

1. **恢复原始 GRPO 标准化**
   - `run_qwen3-1.7B.sh`
   - 把 `algorithm.norm_adv_by_std_in_grpo=False` 改成 `True`

2. **恢复论文 Appendix C.2 的 KL 系数**
   - `etth1_ot_qwen3_gpu012.sh`
   - 把 `RL_KL_LOSS_COEF` 默认值从 `0.01` 调回 `0.04`

3. **训练态三轮都要按论文采样，不要对 refinement turn 强制 `temperature=0.0`**
   - `time_series_forecast_agent_flow.py`
   - `_prepare_sampling_params(...)` 在训练态不应再对 `refinement` 做单独 deterministic override
   - 训练态 Turn 1 / Turn 2 / Turn 3 都应共享 rollout sampling 语义
   - `temperature` 训练态应回到论文 Appendix C.2 的 `1.0`
   - `val_only / debug deterministic eval` 才能保留单独的 deterministic 分支

4. **长度配置恢复 appendix 默认**
   - `etth1_ot_qwen3_gpu012.sh`
   - `RL_MAX_RESPONSE_LENGTH` 调回 `4096`
   - prompt length 与 appendix 保持一致时，再单独记录工程版 profile

5. **不要引入论文没有要求的“随机动作采样器”**
   - 这里的“改成论文同款”不是新加一个 uniform random action sampler
   - 而是保持当前 LLM policy rollout 形式，同时取消外置 heuristic 预裁剪与 turn-specific deterministic override
   - 使得采样对象真正变成：`policy-induced complete decision trajectories`

#### 这一步改完后的验证标准

- 训练 profile 默认值能直接对应到论文 Appendix C.2。
- 训练态三轮 episode 都允许参与 trajectory sampling，而不是只有前两轮有随机性。
- 对同一个 query 的训练组 rollout，8 条 trajectory 应可能在：
  - Turn 1 工具选择
  - Turn 2 expert 选择
  - Turn 3 refinement 决策
  三个层面出现可归因于 policy 的差异。

### 11.4 P2：把 reward 恢复成统一 multi-view aggregation，而不是 gated tie-break

#### 需要修改的文件

- `recipe/time_series_forecast/reward.py`
- `recipe/time_series_forecast/reward_metrics.py`

#### 如何修改

1. 移除 `compute_structural_tie_break_gate(norm_mse)` 对结构项的门控。
2. 让下面几项始终参与最终 reward：
   - normalized/log-transformed MSE
   - trend consistency
   - seasonal consistency
   - turning-point alignment
   - format validity
   - output length consistency
3. 保留 fixed predefined weights，不再让结构项只在低 MSE 区域出场。
4. 若仍需要工程版 reward，请单独保留 `engineering_reward_profile`，不要覆盖 paper reproduction 路径。

#### 这一步改完后的验证标准

- reward debug 日志中，结构项在全域样本上都可见，而不是仅在低 `norm_mse` 时出现。
- paper reproduction profile 与 engineering profile 的 reward 配方分开。

### 11.5 P2：把 curriculum 从三档切片改回双轴渐进式课程

#### 需要修改的文件

- `recipe/time_series_forecast/build_etth1_rl_dataset.py`

#### 如何修改

1. 替换当前：
   - `_resolve_difficulty_stage(error_band, entropy_band) -> max(rank)`
   这套映射。

2. 改成显式双轴课程：
   - `stage1`: low teacher error + low entropy
   - `stage2`: 更高 teacher error，但 entropy 仍低到中等，保持结构规整
   - `stage3`: high entropy / noisy / stochastic windows

3. `build_train_stage_slices(...)` 不再简单做：
   - `easy`
   - `easy + medium`
   - `full`
   而是按上述双轴规则生成 cumulative stage slices。

#### 这一步改完后的验证标准

- metadata 中同时保留 `error_band` 与 `entropy_band`。
- 每个 curriculum stage 的样本组成能解释为论文中的“低难度 -> 高预测难度 -> 高随机性”。

### 11.6 P2：把 appendix-faithful prompt 与工程 prompt 分开，不再混用

#### 需要修改的文件

- `recipe/time_series_forecast/prompts.py`
- `examples/time_series_forecast/configs/etth1_ot_qwen3_gpu012.sh`

#### 如何修改

1. 在 `prompts.py` 中引入显式的 prompt profile：
   - `paper_appendix`
   - `engineering_debias`

2. `paper_appendix` 模式下：
   - Turn 2 prompt 恢复 Appendix E.5 的 expert-role wording
   - 但不要恢复论文里没有写过的 `primary_model / runner_up_model` runtime 泄露

3. config 层新增独立的 paper reproduction profile：
   - 指向 paper-aligned dataset / checkpoint / prompt profile
   - 不再默认复用 `mv1/tsfix` 工程链路

#### 这一步改完后的验证标准

- “paper reproduction” 与 “engineering anti-collapse” 能通过 profile 明确区分。
- 对外汇报时不会再把工程 prompt 误说成 appendix-faithful prompt。

### 11.7 修改顺序建议

如果按最小风险推进，顺序应固定为：

1. 先修 **GRPO terminal outcome**。
2. 再修 **planning agentization**。
3. 再恢复 **paper-faithful GRPO config**。
4. 然后分别处理 **reward** 与 **curriculum**。
5. 最后把 **paper profile** 与 **engineering profile** 拆开。

原因很简单：

- `GRPO outcome` 不修，后续 RL 现象都不可靠。
- `planning` 不收回 agent，paper 的 sequential decision 语义就还没落地。
- `prompt / reward / curriculum` 的 paper-fidelity 调整应建立在前两项已经正确的前提上。

### 11.8 一句话的整改目标

不偏离论文的整改目标不是“继续压 prompt 泄露”，而是把实现重新拉回这三条主线：

1. **trajectory-level delayed reward**
2. **agent-owned planning over the structured action space**
3. **paper-faithful complete-trajectory sampling + GRPO / reward / curriculum configuration**

## 12. 2026-03-25 已落实整改

本节记录已经实际落到代码中的整改，而不是停留在“建议修改”。

### 12.1 已完成项

1. **GRPO 已改成按 terminal trajectory reward 做组内比较**
   - `arft/core_algos.py`
   - `arft/ray_agent_trainer.py`
   - `arft/agent_flow/agent_flow.py`
   - 现在每条 `trajectory_uid` 只取末步 reward 参与 GRPO outcome advantage。
   - non-terminal step 不再向 GRPO outcome 提供伪中间分数。

2. **Turn 1 planning 已收回 agent**
   - `recipe/time_series_forecast/time_series_forecast_agent_flow.py`
   - `recipe/time_series_forecast/prompts.py`
   - runtime 不再由外置 heuristic 预先裁剪 diagnostic tool batches。
   - agent 直接面对完整 diagnostic tool set，自主决定调用哪些 feature tools。

3. **训练态采样已恢复为三阶段都可参与 trajectory sampling**
   - `recipe/time_series_forecast/time_series_forecast_agent_flow.py`
   - refinement turn 不再被 flow 强制 `temperature=0.0`。

4. **reward 已恢复成统一 multi-view aggregation**
   - `recipe/time_series_forecast/reward.py`
   - `recipe/time_series_forecast/reward_metrics.py`
   - 已删除 `compute_structural_tie_break_gate(norm_mse)` 及其所有调用。
   - `normalized/log MSE`、`season/trend consistency`、`turning-point alignment`、`format validity`、`length consistency` 现在都直接按固定权重进入终局 reward。

5. **curriculum 已从 `max(rank)` 简化版改成显式双轴规则**
   - `recipe/time_series_forecast/build_etth1_rl_dataset.py`
   - 现在使用：
     - `easy = low teacher error + low entropy`
     - `medium = higher teacher error with low/medium entropy`
     - `hard = high entropy / stochastic windows`
   - metadata 中新增：
     - `curriculum_policy = teacher_error_entropy_two_axis`
     - `curriculum_stage_definitions`
     - split-level `curriculum_stage_distribution`

6. **formal profile 默认值已切回 paper-named 数据链路**
   - `examples/time_series_forecast/configs/etth1_ot_qwen3_gpu012.sh`
   - 当前默认值已切到：
     - `RL_CURRICULUM_DATASET_DIR = dataset/ett_rl_etth1_paper_aligned_ot_curriculum_same2`
     - `RL_MODEL_PATH = artifacts/checkpoints/sft/time_series_forecast_sft_paper_strict_formal_20260323/global_step_33/huggingface`
     - `SFT_DATASET_DIR = dataset/ett_sft_etth1_runtime_ot_teacher200_paper_same4_stepwise`

### 12.2 代码级验证结果

- `python -m py_compile recipe/time_series_forecast/reward_metrics.py recipe/time_series_forecast/reward.py tests/test_compact_protocol.py`
  - 通过

- `pytest -q tests/test_compact_protocol.py`
  - `17 passed`

- `python -m py_compile recipe/time_series_forecast/build_etth1_rl_dataset.py tests/test_rl_dataset_builder.py`
  - 通过

- `bash -n examples/time_series_forecast/configs/etth1_ot_qwen3_gpu012.sh`
  - 通过

- `pytest -q tests/test_rl_dataset_builder.py tests/test_curriculum_utils.py tests/test_heuristic_curated_sft_builder.py tests/test_high_quality_sft_builder.py`
  - `36 passed`

- 综合回归：
  - `pytest -q tests/test_grpo_terminal_outcome.py tests/test_compact_protocol.py tests/test_time_series_forecast_agent_flow.py tests/test_final_answer_parsing.py tests/test_diagnostic_policy.py tests/test_sft_dataset_builder.py tests/test_high_quality_sft_builder.py tests/test_ray_agent_trainer_validation.py tests/test_rl_dataset_builder.py tests/test_curriculum_utils.py tests/test_heuristic_curated_sft_builder.py`
  - `135 passed, 1 warning`

### 12.3 直接 sanity check

- reward sanity：
  - 在 `norm_mse = 4.0100` 的高误差样本上，当前仍有：
    - `change_point_score = 0.05`
    - `season_trend_score = 0.02686`
  - 说明结构项已不再只在低 MSE 区域出现。

- curriculum sanity：
  - `low/low -> easy`
  - `high/medium -> medium`
  - `low/high -> hard`

- profile sanity：
  - 当前 formal profile 实际导出的默认路径已经是 paper-named 链路，而不是 `mv1/tsfix`。

### 12.4 真实 RL smoke 的 GRPO 采样验证

为了确认不只是“代码看起来像论文”，还实际按论文语义采样，额外执行了一轮真实训练态 smoke：

- 环境：
  - `cast-r1-ts`
  - `RUN_MODE=smoke`
  - `RL_CURRICULUM_PHASE=stage1`
  - `train_batch_size=1`
  - `rollout.n=8`
  - `temperature=1.0`
  - `NUM_GPUS=1`
  - 独立预测服务运行在 `GPU 3 / :8994`

- chain debug 文件：
  - `logs/debug/grpo_real_smoke_20260325/ts_chain_debug.jsonl`

- 真实 rollout 结果显示：
  - 同一个 `sample_uid = 8f6efc97-c8b9-43d1-bb7c-34979bf8096d`
  - 在本次 smoke 中实际展开成 `8` 个不同的 `request_id`
  - 即：
    - `60b410d7`
    - `4b62544f`
    - `5f1f99c6`
    - `95588014`
    - `4902c1b9`
    - `8672eecb`
    - `1d0d93f5`
    - `fbb07a6b`

- 这 8 条 trajectory 的行为并不相同，说明训练态确实在做 policy-induced trajectory sampling，而不是 deterministic 单轨迹复用。
  例子：
  - 有的轨迹路由到 `arima`
  - 有的轨迹路由到 `chronos2`
  - 有的轨迹在 routing stage 因 tool failure / retry 仍未完成到 refinement
  - 至少有多条轨迹进入了 `refinement`

- 因为 `train_batch_size=1`，这说明同一个 query 在真实训练态确实被采样成了 `G=8` 条 trajectory，符合论文的 GRPO rollout 语义。

- 结合代码链路可确认：
  - `verl/verl/trainer/ppo/ray_trainer.py`
    - 先给原始 batch 分配一个 `uid`
    - 再执行 `batch.repeat(repeat_times=rollout.n, interleave=True)`
  - `arft/agent_flow/agent_flow.py`
    - 每条 rollout trajectory 再单独生成自己的 `trajectory_uid`
  - `arft/core_algos.py`
    - GRPO advantage 以共同的 `uid` 分组
    - 以各自不同的 `trajectory_uid` 区分 trajectory
    - 只取各 trajectory 的 terminal reward 做组内比较

- 因此当前 GRPO 的训练态输入输出语义已经满足论文要求：
  - **同一个 query**
  - **采样 8 条 trajectory**
  - **每条 trajectory 有自己的完整多轮决策路径**
  - **最终只用 terminal reward 做 group-relative advantage**

- 本次 smoke 也额外暴露出一个非 GRPO 语义问题：
  - 预测服务在并发 `predict` 时会返回部分 `503 Service Unavailable`
  - 这会影响部分 trajectory 的完整 rollout
  - 但它属于 expert service 并发稳定性问题，不属于 GRPO 采样语义偏差

### 12.5 非法请求根因复盘与当前修复状态

在继续排查 `503` 之后，已确认之前的非法请求主因不是：

- `patchtst / itransformer` 只支持单变量
- 或论文/服务端不支持 multivariate ETTh1
- 或 GRPO 本身的 rollout 并发语义错误

真实根因是：

- `patchtst` 与 `itransformer` checkpoint 本身就是 **ETTh1 七变量 expert**
  - `enc_in = 7`
- 但 formal profile 当时使用的磁盘数据产物仍是 **value-only prompt**
  - runtime 从 prompt 里只能解析出 `OT` 单变量
- 一旦 routing 选到 `patchtst / itransformer`
  - 就会向七变量 expert 发送一变量请求
  - 服务端于是返回：
    - `patchtst is loaded with enc_in=7, but received 1-variable input`
    - `itransformer is loaded with enc_in=7, but received 1-variable input`

这说明问题本质上是 **paper-aligned 数据协议与 expert 契约脱节**，不是模型本身不支持多变量。

当前已落实的修复包括：

1. **RL / SFT 数据已按论文重建为 multivariate timestamped named rows**
   - `dataset/ett_rl_etth1_paper_same2`
   - `dataset/ett_sft_etth1_runtime_teacher200_paper_same2`
   - `dataset/ett_sft_etth1_runtime_ot_teacher200_paper_same4_stepwise`
   - `dataset/ett_rl_etth1_paper_aligned_ot_curriculum_same2`

2. **所有相关 metadata 现在都显式声明 multivariate ETTh1 契约**
   - `task_type = multivariate time-series forecasting`
   - `historical_data_protocol = timestamped_named_rows`
   - `target_column = OT`
   - `observed_feature_columns = [HUFL, HULL, MUFL, MULL, LUFL, LULL, OT]`
   - `observed_covariates = [HUFL, HULL, MUFL, MULL, LUFL, LULL]`
   - `model_input_width = 7`

3. **训练脚本入口已强制校验 metadata**
   - `examples/time_series_forecast/run_qwen3-1.7B.sh`
   - `examples/time_series_forecast/run_qwen3-1.7B_sft.sh`

4. **本地预测客户端会在发请求前校验 width / seq_len**
   - 非法单变量请求不会再被发往 `patchtst / itransformer`

5. **服务端已把契约错误改成显式 `400`**
   - 不再把输入宽度或 lookback 不合法误报成 `503`

#### 直接验证结果

- 回归测试：
  - `pytest -q tests/test_dataset_identity.py tests/test_model_server_batch.py tests/test_time_series_utils.py tests/test_etth1_feature_smoke.py tests/test_sft_dataset_builder.py tests/test_high_quality_sft_builder.py`
  - `62 passed, 1 warning`

- 真实服务端到端验证：
  - 从 `dataset/ett_rl_etth1_paper_aligned_ot_curriculum_same2/train_stage123.jsonl` 读取第一条样本
  - runtime 解析结果为：
    - `rows = 96`
    - `feature_columns = [HUFL, HULL, MUFL, MULL, LUFL, LULL, OT]`
  - 对同一条样本直接调用：
    - `patchtst` -> 成功返回 `96` 行预测
    - `itransformer` -> 成功返回 `96` 行预测
    - `chronos2` -> 成功返回 `96` 行预测

- 脚本级 metadata 校验：
  - `PRINT_CMD_ONLY=1 RUN_MODE=val_only RL_CURRICULUM_PHASE=stage123 bash examples/time_series_forecast/run_qwen3-1.7B.sh`
    - 通过
  - `PRINT_CMD_ONLY=1 MODEL_PATH=/data/linyujie/models/Qwen3-1.7B bash examples/time_series_forecast/run_qwen3-1.7B_sft.sh`
    - 通过
    - `train.parquet / val.parquet` 的 Turn 3 protocol 校验也通过

#### 当前剩余风险

这条非法请求链已经修掉，但还剩一个与“论文完整复现质量”相关的风险：

- 目前重建 `dataset/ett_rl_etth1_paper_aligned_ot_curriculum_same2` 时使用的是 **部分 teacher metadata 覆盖**
- 因此当前 curriculum split 的覆盖度偏低：
  - `train_stage1 = 27`
  - `train_stage12 = 174`
  - `train_stage123 = 12060`

这不会再导致非法请求，也不影响当前 multivariate paper contract；  
但如果目标是尽量贴近论文的 curriculum quality，后续仍应使用更高覆盖的 teacher-eval 结果重建一次 curriculum 数据。
