# Archive Candidates

日期：`2026-03-22`

这份清单只列出建议归档或删除的历史产物，不包含当前 paper-aligned 主线必须保留的代码。

## 已删除代码

- `recipe/time_series_forecast/analyze_chain_debug.py`
  - 原因：纯离线 debug helper，无主链路引用，无测试覆盖。
- `recipe/time_series_forecast/benchmark_models_on_rl_samples.py`
  - 原因：纯离线 benchmark helper，无 launcher / 主流程引用，无测试覆盖。

## 建议归档的旧数据产物

- `dataset/ett_sft_etth1_runtime_ot_teacher200_paper_same3/`
  - 旧版非 step-wise SFT parquet，已被 `same4_stepwise` 取代。
- `dataset/ett_sft_etth1_runtime_ot_teacher200_paper_same3_multiturn/`
  - 旧版 transcript 风格 SFT 产物，已被 runtime step-wise 数据取代。

## 建议保留的当前数据主线

- `dataset/ett_rl_etth1_paper_same2/`
- `dataset/ett_rl_etth1_paper_aligned_ot_curriculum_same2/`
- `dataset/ett_sft_etth1_runtime_teacher200_paper_same2/`
- `dataset/ett_sft_etth1_runtime_ot_teacher200_paper_same4_stepwise/`

## 建议归档的 debug 目录

- `logs/debug/recheck_20260322_smoke`
- `logs/debug/recheck_20260322_201609_smoke`
- `logs/debug/diag_budgetfix_temp0_valonly_20260322`
- `logs/debug/diag_maxsteps4_valonly2_20260322`
- `logs/debug/diag_maxsteps4_valonly_20260322`
- `logs/debug/diag_maxsteps6_valonly_20260322`
- `logs/debug/diag_paper_multiturnsft_temp0_valonly_20260322`
- `logs/debug/diag_paper_multiturnsft_fix1_temp0_valonly_20260322`
- `logs/debug/diag_paper_multiturnsft_fix2_temp0_valonly_20260322`
- `logs/debug/diag_patched_base_valonly_20260322`
- `logs/debug/diag_patched_base2_valonly_20260322`
- `logs/debug/diag_refine_temp0_valonly_20260322`
- `logs/debug/diag_stepwise_sft_temp0_valonly_20260322`

## 建议保留的当前 debug 基线

- `logs/debug/diag_stepwise_sft_fulltemplate_paper3stage_temp0_valonly_20260322/`
  - 当前 paper-aligned 三阶段 fresh smoke 基线。

## 仍建议人工确认后再删的代码

- `recipe/time_series_forecast/retrain_expert_models_train_split.py`
  - 如果后续不再重训本地专家模型，可归档。
- `recipe/time_series_forecast/build_etth1_sft_subset.py`
  - 如果后续不再做小规模调试子集，可归档。
