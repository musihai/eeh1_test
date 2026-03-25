# ETTh1 剩余论文偏差与 GRPO 审计报告

日期：2026-03-25

## 1. 结论先行

当前代码在三阶段工作流、memory-based state、delayed reward、`G=8` 组采样这些主框架上，已经基本保留了论文主线；但如果目标是“严格按论文复现”，现在仍有几处重要偏差。

其中最严重的不是 prompt 细节，而是 **GRPO 的 episode score 取值链路存在实现级错误**：

> 当前 `compute_grpo_outcome_advantage(...)` 很可能拿到的是每条 trajectory 的首个 step 分数，而不是终局 delayed reward。

由于当前中间 step 的 reward 被显式设为 `0.0`，这意味着在正常三阶段轨迹里，GRPO 组内比较极可能在比较一组 `0.0`，而不是比较最终 forecast 的 reward。  
这不是“论文里的 GRPO 本来就像随机采样”，而是 **当前实现没有把论文里的“按终局 episode reward 做组相对优势”真正落下来**。

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

## 4.2 高风险偏差二：当前 GRPO 实现没有按论文使用终局 episode reward 做组内比较

这是本轮发现的最重要问题。

### 论文怎么写

论文 §3.5.2 / Appendix C.2 写得很清楚：

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

#### B. agent flow 在 postprocess 时按 step 顺序写入 trajectory

- `arft/agent_flow/agent_flow.py:672-696`
  - 每条 trajectory 的 step 按顺序写入 batch
  - 每条 trajectory 的多个 step 共用同一个 `trajectory_uid`

#### C. 但 GRPO 只取每个 `trajectory_uid` 第一次出现的 score

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

它不是“纯随机采样”，而是：

- stochastic rollout
- + group-relative ranking
- + delayed reward credit assignment

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

## 6.2 但实现上有三层关键问题

### 问题 A：终局 reward 没被正确喂进 GRPO 组比较

见 §4.2。  
这是当前最严重的问题。

### 问题 B：当前是 `norm_adv_by_std_in_grpo=False`

- `examples/time_series_forecast/run_qwen3-1.7B.sh:285`

这让它更接近：

- centered outcome
- 非原始 GRPO 标准化优势

### 问题 C：debug / val-only 经常是 `n=1` 且 `temperature=0.0`

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

> 当前实现让 GRPO 失去了论文里最关键的“终局 reward 组比较”这一层，因此看起来像是在做很多采样，但没有形成有效的 group-relative learning signal。

造成这种感觉的原因至少有四个：

1. **如果首步零分被拿去做组比较，优势会接近全零。**
2. **`norm_adv_by_std_in_grpo=False` 进一步削弱了原始 GRPO 的归一化对比。**
3. **Turn 3 deterministic decoding 限制了完整 episode 的探索差异。**
4. **当前 warm start / heuristic scaffold 本来就很强，rollout 多样性已经被压窄。**

所以主问题不是：

- “论文的 GRPO 其实就是 random sampling”

而是：

- “我们当前的实现没有把论文的 GRPO 关键信号链跑通”

---

## 8. 优先级排序

### P0：必须先修

1. 修 GRPO episode score 取值逻辑  
   当前必须改成：每条 trajectory 参与 group comparison 的是 **终局 delayed reward**，不是首个 step reward。

### P1：紧随其后

2. 恢复 paper-faithful 的 GRPO 配置
   - `norm_adv_by_std_in_grpo=True`
   - `KL=0.04`
   - 尽量靠近论文的 batch / response length

3. 决定是否要 paper-faithful prompt
   - 如果目标是严格复现论文，应恢复 appendix 风格 Turn 2 prompt
   - 如果目标是先止损坍塌，则保留去模板化 prompt，但要明确这是“偏离论文的工程修正”

4. 把 planning 从外置 heuristic 逐步收回 agent

5. 若要继续追 paper fidelity，再细化 curriculum 调度

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

当前代码的问题不是“只是随机采样”，而是更严重的：

> 当前 GRPO 的组比较很可能拿错了每条 trajectory 的 reward，导致它没有真正按论文方式用终局 reward 做 group-relative optimization。

这应该被视为 **当前 RL 复现链路的首要修复点**。
