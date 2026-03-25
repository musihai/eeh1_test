# ETTh1 代码对齐与坍塌排查报告

日期：2026-03-25

## high_risk_deviations

- runtime 诊断 prompt 在进入 Turn 2 前就泄露候选 expert 假设。
  现象：
  `diagnostic_plan_reason` 本身会写出 ``arima looks strongest`` 或 ``distinguish patchtst from itransformer``，并且 `diagnostic_primary_model` / `diagnostic_runner_up_model` 还能继续把候选模型塞进 prompt。
  影响：
  这会把论文里的 “先做 feature extraction，再基于 updated state adaptive routing” 退化成 “先由 heuristic 预路由，再让 LLM 顺着提示执行”。

- step-wise SFT 的 routing teacher 主要来自 heuristic，而不是 offline best/reference teacher。
  现象：
  `build_etth1_sft_dataset.py` 在原始逻辑里直接调用 `_select_prediction_model_by_heuristic(...)`，并记录 `routing_policy_source = "heuristic_rule_based"`。
  定量证据：
  `dataset/ett_sft_etth1_runtime_teacher200_mv1/train_curated.jsonl` 里，`selected_prediction_model` 与 `reference_teacher_model` 的源样本级一致率仅 `0.27`。
  影响：
  SFT warm start 学到的是 heuristic 行为先验，不是离线最优路由。

- 当前坍塌与 expert 环境最优解不一致。
  定量证据：
  `dataset/ett_sft_etth1_runtime_teacher200_mv1/val_teacher_eval.jsonl` 上，四个 expert 均无失败，平均 `orig_mse` 约为：
  `itransformer=5.003`、`patchtst=5.070`、`chronos2=5.270`、`arima=5.850`。
  但 `logs/debug/mv1tf_val32_tsfix_20260325/eval_step_aggregate.jsonl` 只选了 `arima=19, patchtst=13`，完全不选 `itransformer/chronos2`。
  影响：
  当前路由坍塌不能解释为 “环境自然逼着策略选单一最优 expert”。

## medium_risk_deviations

- Turn 3 被压缩成 `KEEP / LOCAL_REFINE`，且 prompt 明示 `If unsure, choose KEEP.`。
  现象：
  当前日志里 `refinement_changed_ratio` 基本在 `0.0 ~ 0.03125`，说明 Turn 3 几乎不做修正。
  影响：
  一旦 Turn 2 选错 expert，Turn 3 几乎没有补救空间，会放大单一 expert 的吸引力。

- routing stage prompt 仍含较强 expert 适用启发式映射。
  现象：
  prompt 直接写出 `patchtst` 对 local motifs、`arima` 对 stable autocorrelation、`chronos2` 对 irregular windows、`itransformer` 对 structural drift。
  影响：
  这与论文 appendix 的说明相近，但在当前 heuristic-heavy 训练链路下，会继续强化模板化选模而不是纯状态决策。

## non_blocking_deviations

- memory-based state management 主线基本还在。
  现象：
  `history_analysis`、`prediction_results`、stage-aware prompt 组装都存在；final reflection 阶段也会用截断历史窗口。

- feature analysis 覆盖不足不是当前主因。
  定量证据：
  当前 debug 聚合里 `analysis_coverage_ratio_mean = 1.0`，`missing_required_feature_tool_count_mean = 0.0`。

- expert 服务稳定性不是当前主因。
  定量证据：
  当前 debug 聚合里 `prediction_model_defaulted_ratio = 0.0`、`prediction_tool_error_count = 0`；teacher-eval 四个 expert 也都是 `failures = 0`。

## likely_root_cause_ranked

1. SFT behavioral prior 偏向 heuristic routing 与 KEEP-style refinement，不止是 routing label
2. runtime heuristic 泄露
3. RL exploitation 放大了偏置 warm start
4. Turn 3 反思修正过弱
5. expert 环境不对等
6. feature analysis 覆盖不足

## evidence

- 论文方法主线强调：
  `memory-based state management`、Turn 1 诊断、Turn 2 adaptive model selection、Turn 3 reflection，以及 `SFT + multi-turn RL + curriculum RL`。

- 当前 `logs/debug/mv1tf_val32_tsfix_20260325/ts_chain_debug.jsonl` 中，32 个样本的 Turn 1 诊断输出里：
  `looks strongest` 出现 11 次，`distinguish X from Y` 出现 21 次。
  说明 runtime 在 feature tool 执行前已经形成显式候选模型暗示。

- 当前 `mv1` curated 源数据中：
  `selected_prediction_model` 分布为 `patchtst=98, itransformer=60, arima=28, chronos2=14`；
  `reference_teacher_model` 分布为 `itransformer=54, patchtst=51, chronos2=49, arima=46`；
  二者显著不一致。

- 当前 `mv1` step-wise SFT parquet metadata 中：
  `train_routing_policy_source_distribution = {'heuristic_rule_based': 200}`。
  说明路由 teacher 来源单一且是 heuristic。

- 当前 `mv1` teacher-eval 全量统计显示：
  train best-model 分布为 `itransformer=4869, chronos2=2661, patchtst=2404, arima=2126`；
  val best-model 分布为 `arima=526, itransformer=506, patchtst=349, chronos2=341`；
  test best-model 分布为 `itransformer=1104, arima=798, chronos2=623, patchtst=540`。
  没有任何 split 呈现 “arima 绝对统治且别的模型明显不可用”。

- 现有不同 debug run 的坍塌目标不稳定。
  例子：
  `logs/debug/mv1rv_val32_20260325/eval_step_aggregate.jsonl` 为 `itransformer=32/32`；
  `logs/debug/mv1tf_val32_tsfix_20260325/eval_step_aggregate.jsonl` 为 `arima=19, patchtst=13`。
  这更像 checkpoint / prompt / warm-start 偏置驱动，而不是环境单峰最优。

## experiment_results_20260325

### 实验 A：去掉 runtime 候选模型提示

- 运行：
  `logs/debug/mv1_hintdrop_val32_20260325_152246/eval_step_aggregate.jsonl`

- 对比基线：
  基线 `logs/debug/mv1tf_val32_tsfix_20260325/eval_step_aggregate.jsonl`
  `validation_reward_mean: 0.03335 -> 0.22120`
  `final_answer_accept_ratio: 0.75 -> 0.875`
  `strict_length_match_ratio: 0.75 -> 0.875`
  `orig_mse_mean: 4.5220 -> 4.4488`

- 路由分布变化：
  `selected_model_distribution`
  基线：`{'arima': 19, 'patchtst': 13}`
  hint-drop：`{'arima': 16, 'patchtst': 15, 'itransformer': 1}`

- 结论：
  runtime 提示泄露确实在放大坍塌和格式问题，但它不是唯一根因。
  去掉提示后，策略仍几乎只在 `arima/patchtst` 之间摆动，`chronos2` 仍然完全缺席。

### 实验 B：routing teacher 改成 reference teacher

- 先做的数据侧对照：
  原 step-wise parquet
  `dataset/ett_sft_etth1_runtime_ot_teacher200_mv1_stepwise_r25_tsfix/metadata.json`
  里
  `train_routing_policy_source_distribution = {'heuristic_rule_based': 200}`。

- 新构建数据集：
  `dataset/ett_sft_etth1_runtime_ot_teacher200_mv1_stepwise_r25_tsfix_refteacher_oslr_20260325/metadata.json`

- 关键结果：
  `routing_label_source = reference_teacher`
  `train_selected_prediction_model_reference_teacher_agreement_ratio = 1.0`
  `val_selected_prediction_model_reference_teacher_agreement_ratio = 1.0`
  `test_selected_prediction_model_reference_teacher_agreement_ratio = 1.0`
  `train_routing_row_selected_prediction_model_distribution = {'arima': 77, 'chronos2': 77, 'itransformer': 77, 'patchtst': 77}`
  `train_routing_policy_source_distribution = {'reference_teacher_offline_best': 246}`
  `train_source_samples = 200`

- 结论：
  SFT 数据层面的 routing teacher 确实已经把行为先验带偏。
  一旦把 routing label 改成 offline best/reference teacher，并保留训练源样本覆盖，step-wise SFT 数据就能恢复为四 expert 均衡路由，而不是 heuristic 单边偏置。

### 实验 B-2：reference-teacher SFT warm start 的 model-side fixed-set 对照

- SFT 训练：
  `artifacts/checkpoints/sft/time_series_forecast_sft_mv1ref_20260325/global_step_66/huggingface`
  训练完成时
  `val/loss = 0.005503471940755844`

- 为避免随机抽样混杂，额外构造固定验证集：
  `dataset/ett_rl_etth1_mv1/val_fixed32_refcmp_20260325.jsonl`
  它由
  `logs/debug/mv1ref_hintdrop_val32_20260325_160021/ts_chain_debug.jsonl`
  中实际出现的 32 个 `sample_uid` 反查得到。

- 同集对照：
  基线 checkpoint：
  `logs/debug/mv1hintdrop_fixedref32_20260325_160819/eval_step_aggregate.jsonl`
  reference-teacher SFT：
  `logs/debug/mv1ref_hintdrop_val32_20260325_160021/eval_step_aggregate.jsonl`

- 关键结果：
  `validation_reward_mean: 0.35586 -> 0.30412`
  `final_answer_accept_ratio: 0.96875 -> 0.9375`
  `selected_forecast_orig_mse_mean: 5.24539 -> 5.78155`
  `orig_mse_mean: 5.36926 -> 5.62267`
  `strict_length_match_ratio: 0.96875 -> 0.9375`

- 同集路由分布：
  基线：
  `{'patchtst': 20, 'arima': 10, 'itransformer': 2}`
  reference-teacher SFT：
  `{'arima': 23, 'patchtst': 9}`

- 同集样本级切换：
  32 个样本中有 17 个样本改了 expert。
  其中 15 个样本被改成 `arima`，平均 `selected_forecast_orig_mse` 变化为 `+1.2363`；
  只有 2 个样本改成 `patchtst`，平均变化为 `-0.6935`。

- 结论：
  “只改 routing label source” 还不足以修复 warm start 的坍塌。
  在当前 SFT 配方下，reference-teacher routing label 甚至会把更多样本推向 `arima`，并在固定验证集上拉坏 forecast 质量。
  这说明当前问题不是单独的 `heuristic routing label`，而是更广义的 SFT 行为先验：
  可能还包括 routing reasoning 文本、non-routing stage supervision、以及 `KEEP` 主导的 Turn 3 目标。

### 实验 C：轻微放宽 Turn 3 prompt 的 fixed-set 对照

- 运行：
  `logs/debug/mv1hintdrop_turn3relax_fixedref32_20260325_162554/eval_step_aggregate.jsonl`
  只额外打开
  `TS_RELAX_TURN3_KEEP_BIAS=1`
  其余 checkpoint、固定验证集、runtime hint-drop 设置都与基线保持一致。

- 对比基线：
  基线
  `logs/debug/mv1hintdrop_fixedref32_20260325_160819/eval_step_aggregate.jsonl`
  `validation_reward_mean: 0.35586 -> 0.31188`
  `final_answer_accept_ratio: 0.96875 -> 0.9375`
  `strict_length_match_ratio: 0.96875 -> 0.9375`
  `selected_forecast_orig_mse_mean: 5.24539 -> 5.24539`
  `orig_mse_mean: 5.36926 -> 5.51305`
  `refinement_changed_ratio: 0.0 -> 0.0`
  `refinement_improved_ratio: 0.0 -> 0.0`
  `final_vs_selected_mse_mean: 0.0 -> 0.0`

- 同集路由分布：
  基线与 Turn 3 放宽版完全一致：
  `{'patchtst': 20, 'arima': 10, 'itransformer': 2}`

- 同集样本级变化：
  32 个样本里，selected expert 没有任何变化。
  只有 1 个样本的最终输出发生变化：
  `etth1-val-01045`
  它从可接受输出退化成
  `invalid_answer_shape:lines=97,expected=96`。

- 结论：
  “只放宽 Turn 3 prompt 文案” 对当前 checkpoint 几乎没有行为影响。
  策略既没有新增任何真实数值修正，也没有改变 expert 选择；
  反而只引入了额外的格式波动。
  这说明 Turn 3 的 `KEEP` 偏置主要不在 runtime wording，而更像是已经固化在 SFT / policy prior 里。

- 数据侧补证：
  step-wise SFT train parquet 的 refinement rows 并不是“全 KEEP”。
  baseline 数据集里
  `validated_keep=149, local_refine=51`；
  reference-teacher 数据集里
  `validated_keep=184, local_refine=62`。
  也就是说，Turn 3 监督本身虽然 keep-heavy，但并非没有 `LOCAL_REFINE` 正例；
  当前 `0/32` 的运行时改值，更像是相对权重太弱或 imitation gap 太大，而不只是标签缺失。

### 实验 D：严格 paper-aligned 的 Turn 3 stronger-SFT（r50 local_refine ratio）对照

- 新构建数据集：
  `dataset/ett_sft_etth1_runtime_ot_teacher200_mv1_stepwise_r50_tsfix_oslr_20260325/metadata.json`

- 保持不变：
  `turn3_target_mode = paper_strict`
  `routing_label_source = heuristic`
  三阶段协议、reward、runtime prompt 都不改。

- 只改：
  `train_min_local_refine_ratio: 0.25 -> 0.5`
  并使用
  `train_turn3_rebalance_mode = oversample_local_refine`
  保持 `train_source_samples = 200`。

- 关键数据侧结果：
  `train_turn3_target_type_distribution: {'local_refine': 175, 'validated_keep': 175}`
  `train_turn_stage_distribution: {'diagnostic': 350, 'refinement': 350, 'routing': 504}`
  说明这轮确实把 Turn 3 监督从 keep-heavy 拉到了 1:1，但没有改协议本身。

- SFT 训练：
  `artifacts/checkpoints/sft/time_series_forecast_sft_mv1r50_20260325/global_step_100/hf_merged`
  训练完成时
  `val/loss = 0.005357787013053894`

- 同集对照：
  基线 checkpoint：
  `logs/debug/mv1hintdrop_fixedref32_20260325_160819/eval_step_aggregate.jsonl`
  r50 stronger-SFT：
  `logs/debug/mv1r50_fixedref32_20260325_170337/eval_step_aggregate.jsonl`

- 关键结果：
  `validation_reward_mean: 0.35586 -> 0.09655`
  `final_answer_accept_ratio: 0.96875 -> 0.78125`
  `strict_length_match_ratio: 0.96875 -> 0.78125`
  `selected_forecast_orig_mse_mean: 5.24539 -> 4.83006`
  `orig_mse_mean: 5.36926 -> 5.36761`
  `refinement_changed_ratio: 0.0 -> 0.03125`
  `refinement_degraded_ratio: 0.0 -> 0.03125`

- 同集路由分布：
  基线：
  `{'patchtst': 20, 'arima': 10, 'itransformer': 2}`
  r50 stronger-SFT：
  `{'patchtst': 32}`

- 同集样本级切换：
  32 个样本里有 12 个样本从 `arima/itransformer` 切到 `patchtst`，
  平均 `selected_forecast_orig_mse` 变化为 `-1.1075`；
  其中 9 个样本改善，3 个样本恶化。

- 唯一一次真实 refinement：
  `etth1-val-01613`
  在 `patchtst` 基础上改了 14 个值，
  `refinement_delta_orig_mse = -0.0171`，
  说明更强的 Turn 3 SFT 的确开始让策略偶尔执行真实修正。

- 主要副作用：
  `missing_answer_close_tag = 7/32`
  `generation_finish_reason = length` 的样本也有 7 个。
  也就是说，这轮虽然第一次触发了非零 refinement，但同时明显伤害了格式稳定性。

- 结论：
  在严格不改论文协议的前提下，单独强化 Turn 3 SFT 监督并不能修复坍塌。
  它会让策略更强地偏向 `patchtst`，并出现少量真实 refinement，但总体 reward 和格式通过率显著变差。
  因此当前问题仍然更像是广义的 SFT 行为先验，而不是“Turn 3 local_refine 正例太少”这一单因子。
  同时，这轮也提示了一个更细的工程结论：
  如果继续做 paper-aligned 修复，应优先做“只增强调优 refinement rows、不要连带复制 routing/diagnostic rows”的 stage-local weighting，
  否则很容易把 routing prior 一起带偏。

## next_experiments

1. Turn 3 refinement-only row weighting
   保持 `paper_strict` 协议不变，但不要再通过 source-level oversampling 复制整条轨迹。
   只重复 `turn_stage=refinement` 且 `turn3_target_type=local_refine` 的行，
   保证 routing / diagnostic rows 的分布不被一起带偏。
   重点观察：
   `refinement_changed_ratio`
   `refinement_improved_ratio`
   `selected_model_distribution`
   `final_answer_accept_ratio`

2. 更窄的 routing-stage SFT ablation
   不再只改 routing label source，而是同时控制：
   routing reasoning 文本里不出现强 expert 模板；
   routing rows 的 loss / repeat factor 提升；
   其余 diagnostic/refinement supervision 尽量不把策略重新拉回 `KEEP + arima/patchtst`。
   重点观察：
   fixed-set `selected_model_distribution`
   `selected_forecast_orig_mse_mean`
   样本级 route switch 是否仍大面积流向 `arima`

3. 所有后续对照都固定验证子集
   优先复用：
   `dataset/ett_rl_etth1_mv1/val_fixed32_refcmp_20260325.jsonl`
   避免随机抽样掩盖 checkpoint 真正差异。

## minimal_patches_applied

- Patch A: runtime 诊断 prompt 去模型提示化
  文件：
  `recipe/time_series_forecast/prompts.py`
  变更：
  默认会清洗 `diagnostic_plan_reason` 中的 ``looks strongest`` / ``distinguish X from Y`` 语句，并且不再把 `diagnostic_primary_model` / `diagnostic_runner_up_model` 注入 prompt。
  对照开关：
  `TS_INCLUDE_DIAGNOSTIC_MODEL_HINTS=1` 可恢复旧行为。

- Patch B: step-wise SFT 支持 reference teacher 路由标签
  文件：
  `recipe/time_series_forecast/build_etth1_sft_dataset.py`
  变更：
  新增 `--routing-label-source {heuristic,reference_teacher}`。
  `reference_teacher` 模式会直接用 `reference_teacher_model/offline_best_model` 作为 routing label，并把结果写入 parquet metadata。

- Patch C: Turn 3 rebalancing 支持保留训练源样本覆盖
  文件：
  `recipe/time_series_forecast/build_etth1_sft_dataset.py`
  变更：
  新增 `--train-turn3-rebalance-mode {downsample_keep,oversample_local_refine}`。
  对实验 B 使用 `oversample_local_refine`，避免 `reference_teacher` 路由标签导致训练源样本从 `200` 被误砍到 `64`，引入额外混杂因素。

- Patch D: Turn 3 prompt 支持放宽 KEEP 偏置的 opt-in 开关
  文件：
  `recipe/time_series_forecast/prompts.py`
  变更：
  新增
  `TS_RELAX_TURN3_KEEP_BIAS=1`
  后的 Turn 3 文案分支。
  开启时允许多个短的非连续修正或受限重写，并把
  `If unsure, choose KEEP.`
  改成更中性的判断条件。
  默认关闭，因此不会影响现有正式链路。

## minimal_patch_verification

- `python -m py_compile recipe/time_series_forecast/prompts.py recipe/time_series_forecast/build_etth1_sft_dataset.py tests/test_compact_protocol.py`
  已通过。

- prompt 验证：
  构造一个含 ``arima looks strongest`` 的诊断 plan 后，默认生成的诊断 prompt 已只保留 feature-oriented 文本，不再暴露候选模型。

- routing label 验证：
  对 `train_curated.jsonl` 中 `index=81` 的样本：
  heuristic 模式得到 `selected_prediction_model = arima`
  reference_teacher 模式得到 `selected_prediction_model = chronos2`
  与 `reference_teacher_model` 一致。

- builder 单测验证：
  `pytest -q tests/test_compact_protocol.py tests/test_sft_dataset_builder.py`
  已通过，`41 passed`。

- reference-teacher 数据集验证：
  `dataset/ett_sft_etth1_runtime_ot_teacher200_mv1_stepwise_r25_tsfix_refteacher_oslr_20260325/train.parquet`
  上：
  `unique_source_sample_index = 200`
  `routing_policy_source = {'reference_teacher_offline_best': 308}`
  `routing selected_prediction_model = {'arima': 77, 'chronos2': 77, 'itransformer': 77, 'patchtst': 77}`。

- model-side fixed-set 对照验证：
  基线：
  `logs/debug/mv1hintdrop_fixedref32_20260325_160819/eval_step_aggregate.jsonl`
  reference-teacher SFT：
  `logs/debug/mv1ref_hintdrop_val32_20260325_160021/eval_step_aggregate.jsonl`
  在同一批 32 个样本上，reference-teacher SFT 没有恢复多 expert 路由，反而把分布推到 `arima=23, patchtst=9`，并把 `selected_forecast_orig_mse_mean` 从 `5.24539` 拉高到 `5.78155`。

- Turn 3 prompt-only 放宽验证：
  基线：
  `logs/debug/mv1hintdrop_fixedref32_20260325_160819/eval_step_aggregate.jsonl`
  Turn 3 放宽版：
  `logs/debug/mv1hintdrop_turn3relax_fixedref32_20260325_162554/eval_step_aggregate.jsonl`
  在同一批 32 个样本上，selected expert 分布完全不变，
  `selected_forecast_orig_mse_mean` 完全不变，
  `refinement_changed_ratio` 仍为 `0.0`；
  但 `validation_reward_mean` 从 `0.35586` 降到 `0.31188`，
  额外多出 1 个 `invalid_answer_shape:lines=97,expected=96`。

- 严格 paper-aligned 的 Turn 3 stronger-SFT 验证：
  新数据集：
  `dataset/ett_sft_etth1_runtime_ot_teacher200_mv1_stepwise_r50_tsfix_oslr_20260325/metadata.json`
  新 warm start：
  `artifacts/checkpoints/sft/time_series_forecast_sft_mv1r50_20260325/global_step_100/hf_merged`
  fixed-set 对照：
  `logs/debug/mv1r50_fixedref32_20260325_170337/eval_step_aggregate.jsonl`
  在同一批 32 个样本上，
  `refinement_changed_ratio` 首次从 `0.0` 升到 `0.03125`，
  `selected_forecast_orig_mse_mean` 从 `5.24539` 降到 `4.83006`；
  但路由完全坍到 `patchtst=32/32`，
  `final_answer_accept_ratio` 降到 `0.78125`，
  并出现 `7` 个 `missing_answer_close_tag`。
