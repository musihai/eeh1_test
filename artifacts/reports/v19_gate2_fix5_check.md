## v19 Gate 2 Fix-5 Recheck

Date: 2026-03-31

### Scope

This report records the Gate 2 recheck after applying **Fix 5** from
`v19_Gate2_单一candidate坍塌_代码级排障+修复方案.md`:

- stronger visible candidate features in the Turn 3 prompt
- rebuilt `ett_sft_etth1_v19_final_select_only`
- reran final-select warm-up for 100 steps
- probed checkpoints at steps `25 / 50 / 75 / 100`

### Code / Data Changes

- Added visible feature computation in
  `recipe/time_series_forecast/candidate_selection_support.py`
- Injected structured candidate comparison fields in
  `recipe/time_series_forecast/prompts.py`
- Stored visible candidate metrics in v19 SFT records in
  `recipe/time_series_forecast/build_etth1_v19_sft_dataset.py`
- Extended coverage in `tests/test_v19_candidate_selection.py`

### Validation of the Fix Itself

- Dataset rebuild completed successfully:
  - `dataset/ett_sft_etth1_v19_final_select_only`
- Protocol validity remained stable:
  - `turn3_protocol_valid_ratio = 1.0`
- Class balancing remained correct:
  - train labels: `54 x 4 classes`
- Think-text diversity remained high after fix 4:
  - `unique_think_count = 153`
- New prompt now contains structured visible comparison fields:
  - `recent_match=...`
  - `vs_default_gain=...`
  - `direction_check=...`

Visible-feature audit report:

- `artifacts/reports/v19_gate2_fix5_visible_feature_audit.md`

### Warm-up Run

Run:

- `artifacts/checkpoints/sft/qwen3-1.7b-etth1-final-select-v19-gate2fix-r5-20260331_160000`

Merged probeable checkpoints:

- `global_step_25/huggingface`
- `global_step_50/huggingface`
- `global_step_75/huggingface`
- `global_step_100/huggingface`

Training-side validation loss:

| step | val/loss |
| --- | ---: |
| 25 | 0.1216 |
| 50 | 0.0772 |
| 75 | 0.0776 |
| 100 | 0.0798 |

For reference, this is materially better than the previous r4 run, whose
validation loss stayed around `0.2178 ~ 0.2763`.

### Probe Results

| step | single max share | exact | top2 | final vs default | risky vs default | default_ok vs default | distribution |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 25 | 0.4219 | 0.1875 | 0.5000 | +0.5423 | +0.3896 | +0.6858 | patchtst 27 / arima 18 / chronos2 13 / itransformer 6 |
| 50 | 0.4688 | 0.1562 | 0.5156 | +1.0572 | +0.5180 | +1.5637 | itransformer 30 / arima 24 / patchtst 8 / chronos2 2 |
| 75 | 0.3125 | 0.2344 | 0.5156 | +0.7028 | +0.2065 | +1.1689 | arima 20 / patchtst 19 / itransformer 19 / chronos2 6 |
| 100 | 0.3594 | 0.1875 | 0.5156 | +0.7386 | +0.2904 | +1.1596 | patchtst 23 / itransformer 19 / arima 16 / chronos2 6 |

### Gate 2 Threshold Check

Required thresholds from the fix plan:

- `protocol_ok_rate = 1.0`
- `materialize_ok_rate = 1.0`
- `single_candidate_max_share < 0.7`
- `selected_candidate_distribution` covers at least 2 classes
- `final_vs_default_mean < 0`
- risky subset should also satisfy `final_vs_default_mean < 0`
- `default_ok` subset should stay near `0` or below

Observed:

- `protocol_ok_rate = 1.0` at all checked steps
- `materialize_ok_rate = 1.0` at all checked steps
- `single_candidate_max_share < 0.7` at all checked steps
- all checkpoints cover at least 4 visible classes
- but `final_vs_default_mean` is **positive at all steps**
- and `risky_final_vs_default_mean` is also **positive at all steps**
- `default_ok_final_vs_default_mean` remains clearly positive

### Decision

Gate 2 still fails after Fix 5.

### What Improved

- The original single-candidate collapse was successfully broken.
- The model no longer emits a single fixed candidate like
  `patchtst__keep: 64 / 64`.
- Candidate usage is now clearly diversified across all four keep candidates.
- Training optimization is healthier:
  - validation loss fell sharply versus r4
  - prompt truncation / protocol issues did not return

### What Did Not Improve Enough

- The policy is still not selecting candidates that beat the default baseline.
- Even the `default_risky` subset remains positive, meaning the model is not
  reliably using the added visible features to identify when overriding the
  default path helps.
- The `default_ok` subset is still being over-overridden.

### Interpretation

Fix 5 solved a **representation / collapse** problem, but not the deeper
**decision-quality** problem.

Current evidence suggests:

1. Turn 3 now has enough signal to avoid a single-answer shortcut.
2. But the visible candidate features are still not sufficient for the model to
   learn a reliable "override only when helpful" selection policy.
3. Therefore, Gate 2 failure is no longer explained by:
   - prompt truncation
   - zero loss-mask rows
   - class imbalance
   - candidate order shortcut
   - thin reasoning supervision alone

The remaining bottleneck is more likely the task formulation itself:

- Turn 3 is still asked to make a final candidate choice from summaries that do
  not expose enough decision-relevant information to beat the default baseline
  consistently.

### Consequence

Per `v19_Gate2_单一candidate坍塌_代码级排障+修复方案.md`:

- do **not** build `full_stepwise_v19`
- do **not** enter Gate 3
- do **not** enter RL

