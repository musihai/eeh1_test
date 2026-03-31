# v19 Gate 2 Root Cause Audit

Date: 2026-03-31

## Scope

This audit records the Step 1-5 code-level investigation required by
`v19_Gate2_单一candidate坍塌_代码级排障+修复方案.md` before any new warm-up or
Gate 2 re-check is run.

Current failure under investigation:

- Gate 2 protocol is healthy
- materialization is healthy
- but final selection collapsed to a single candidate:
  - `selected_candidate_distribution = {"patchtst__keep": 64}`
  - `single_candidate_max_share = 1.0`

## Step 1: final_select_only Train Distribution

Dataset:

- `dataset/ett_sft_etth1_v19_final_select_only/train.parquet`

Train rows:

- `200`

Global `final_candidate_label` distribution:

- `arima__keep = 50`
- `chronos2__keep = 48`
- `itransformer__keep = 48`
- `patchtst__keep = 54`

`risk_label` distribution:

- `default_ok = 84`
- `default_risky = 116`

Within `default_ok`:

- `arima__keep = 4`
- `chronos2__keep = 7`
- `itransformer__keep = 48`
- `patchtst__keep = 25`

Within `default_risky`:

- `arima__keep = 46`
- `chronos2__keep = 41`
- `patchtst__keep = 29`

Interpretation:

- There is no global single-label majority strong enough to explain an
  automatic collapse to `patchtst__keep`.
- However, the train split is not subgroup-balanced:
  - `default_ok` is heavily skewed toward `itransformer__keep`
  - `default_risky` has no `itransformer__keep` gold at all
- This supports adding explicit train balancing for `final_candidate_label`
  instead of assuming the current 200-row train split is already safe.

## Step 2: Effective Training-Step Audit

Previous warm-up checkpoint:

- `artifacts/checkpoints/sft/qwen3-1.7b-etth1-final-select-v19-20260331_142354/global_step_16/huggingface`

Observed training scale:

- `train_rows = 200`
- `global_batch_size = 12`
- `world_size = 3`
- `gradient_accumulation_steps = 1`
- `steps_per_epoch (drop_last) = 16`
- observed warm-up `global_step = 16`

Interpretation:

- The previous warm-up was effectively a single short epoch.
- `16` optimizer steps are enough to teach schema compliance, but not enough to
  conclude that candidate discrimination has failed as a method.
- The plan's recommendation to switch from epoch-controlled warm-up to
  step-controlled warm-up (`100~200` steps, eval/probe every `25`) is justified.

## Step 3: Candidate Order Bias Audit

Current final-selection prompt candidate order is fixed.

Example order observed in the train prompt:

1. `itransformer__keep`
2. `patchtst__keep`
3. `arima__keep`
4. `chronos2__keep`

Implications:

- `patchtst__keep` is always shown in slot 2
- `itransformer__keep` is always shown in slot 1
- candidate order is currently learnable as a shortcut

Interpretation:

- The current builder exposes a deterministic slot pattern.
- Candidate randomization with a fixed reproducible seed is required for the
  next warm-up to test and remove slot bias.

## Step 4: Turn 3 Supervision Thinness Audit

Inspection target:

- assistant response in `train.parquet`

Result:

- only `4` unique `<think>` texts appear across all `200` training rows

The four unique reasoning texts are just label-specific templates:

- `I compare the visible default-path and alternative candidates and choose arima__keep as the final forecast.`
- `I compare the visible default-path and alternative candidates and choose chronos2__keep as the final forecast.`
- `I compare the visible default-path and alternative candidates and choose itransformer__keep as the final forecast.`
- `I compare the visible default-path and alternative candidates and choose patchtst__keep as the final forecast.`

Interpretation:

- Current Turn 3 reasoning supervision is extremely thin.
- Almost all discriminative supervision is concentrated in the final
  `candidate_id=...` token span.
- This directly supports the plan's suspicion that the model can learn a single
  safe answer string without learning meaningful candidate comparison.

## Step 5: Visible Candidate Learnability Audit

Task:

- Predict gold `final_candidate_label` using only currently visible candidate-
  level information from the v19 data pipeline.

Models:

- Logistic Regression
- Random Forest
- HistGradientBoosting

Validation results from the current visible feature set:

### Logistic Regression

- `accuracy = 0.28125`
- `macro_f1 = 0.23083959899749373`
- `top2 = 0.5`
- predicted distribution:
  - `arima__keep = 34`
  - `chronos2__keep = 8`
  - `patchtst__keep = 19`
  - `itransformer__keep = 3`

### Random Forest

- `accuracy = 0.1875`
- `macro_f1 = 0.14333242183939476`
- `top2 = 0.5`
- predicted distribution:
  - `arima__keep = 7`
  - `chronos2__keep = 20`
  - `patchtst__keep = 37`

### HistGradientBoosting

- `accuracy = 0.203125`
- `macro_f1 = 0.17590682836949645`
- `top2 = 0.40625`
- predicted distribution:
  - `arima__keep = 7`
  - `chronos2__keep = 35`
  - `patchtst__keep = 7`
  - `itransformer__keep = 15`

Interpretation:

- The currently visible candidate summaries are not highly learnable even for
  simple supervised models.
- This means the collapse cannot be blamed only on short warm-up; the current
  visible representation and/or supervision shape is still weak.

## Combined Conclusion

The v19 Gate 2 single-candidate collapse is not explained by a single bug.
The strongest confirmed factors are:

1. The previous warm-up was far too short (`16` optimizer steps).
2. Candidate order is fixed and therefore shortcut-learnable.
3. Turn 3 reasoning supervision is almost constant.
4. The current visible candidate summaries are only weakly predictive of the
   gold `final_candidate_label`.

This supports the exact repair order in
`v19_Gate2_单一candidate坍塌_代码级排障+修复方案.md`:

1. switch to step-based warm-up (`100~200` steps)
2. balance `final_select_only` train labels
3. randomize candidate order
4. if still needed, strengthen Turn 3 reasoning supervision
5. if still needed, strengthen visible candidate features

## Decision

Proceed with fixes 1-3 first.

Do not enter Gate 3 or RL until Gate 2 is rerun and passes with:

- `protocol_ok_rate = 1.0`
- `materialize_ok_rate = 1.0`
- `single_candidate_max_share < 0.7`
- `selected_candidate_distribution` covering more than one class
- `final_vs_default_mean < 0`
