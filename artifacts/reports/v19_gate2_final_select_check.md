## v19 Gate 2: Final Select Check

Date: 2026-03-31

### Scope

This report records the first Gate 2 acceptance check for
`v19_重构+测试+验收方案.md`.

Warm-up checkpoint under evaluation:

- `artifacts/checkpoints/sft/qwen3-1.7b-etth1-final-select-v19-20260331_142354/global_step_16/huggingface`

Validation dataset:

- `dataset/ett_sft_etth1_v19_final_select_only/val.parquet`

Probe output:

- `artifacts/reports/v19_final_select_probe_step16_20260331_142354/final_select_probe_summary.json`
- `artifacts/reports/v19_final_select_probe_step16_20260331_142354/final_select_probe_samples.jsonl`

### Closed-Loop Fix Before Gate 2

The original v19 final-select warm-up could not train because the Turn-3 prompt
was too long:

- prompt length was about `10688` tokens for a representative sample
- SFT `max_length` was `9216`
- all training rows were truncated to the max length
- all rows had `loss_mask = 0`
- training immediately produced `train/loss = nan`

This was fixed by compressing the v19 Turn-3 prompt:

- historical context was reduced to a short target summary + recent target rows
- candidate blocks were reduced from full forecast dumps to compact summary / head / tail previews

After rebuilding `dataset/ett_sft_etth1_v19_final_select_only`:

- `loss_mask_zero_count = 0 / 200`
- train sequence length range became `2551 ~ 2644`
- the warm-up run trained normally
- final training metrics were finite: `val/loss = 0.04074`

### Gate 2 Thresholds

Per v19:

- `final_vs_default_mean < 0`
- risky subset should improve more clearly
- candidate selection must not collapse to a single `candidate_id`

### Probe Results

Core summary:

- `count = 64`
- `protocol_ok_rate = 1.0`
- `materialize_ok_rate = 1.0`
- `exact_match_rate = 0.25`
- `top2_hit_rate = 0.6875`
- `selected_candidate_distribution = {'patchtst__keep': 64}`
- `single_candidate_max_share = 1.0`
- `final_vs_default_mean = +0.0589`
- `risky_final_vs_default_mean = -0.3576`

Risk-subset breakdown:

- `default_ok`: `33` samples
  - `selected = patchtst__keep` on all `33`
  - `final_vs_default_mean = +0.4502`
  - `exact_match_rate = 21.21%`
- `default_risky`: `31` samples
  - `selected = patchtst__keep` on all `31`
  - `final_vs_default_mean = -0.3576`
  - `exact_match_rate = 29.03%`

Gold distribution on this validation split:

- `itransformer__keep = 16`
- `patchtst__keep = 16`
- `arima__keep = 16`
- `chronos2__keep = 16`

### Decision

Gate 2 failed.

### Why It Failed

- The policy collapsed to a single candidate: `patchtst__keep`.
- Overall `final_vs_default_mean` stayed positive (`+0.0589`), so the final selector did not beat the default path on average.
- The risky subset improved, but the `default_ok` subset was over-overridden and became substantially worse (`+0.4502`).
- This violates both core Gate 2 requirements:
  - no single-candidate collapse
  - overall `final_vs_default_mean < 0`

### Interpretation

The first v19 repair loop succeeded in fixing the *training pipeline* failure:

- Turn-3 prompts are now short enough
- SFT supervision reaches the model
- protocol / materialization are both `100%` valid

However, the *behavioral* problem remains:

- the final selector learned a single aggressive alternative (`patchtst__keep`)
- it helps on `default_risky` windows
- but it overrides even when the default path is already good

So the current v19 state is:

- data / protocol chain is fixed
- Gate 2 still fails because Turn 3 is now **overriding too broadly**

### Consequence

Per the v19 plan:

- do not proceed to `full_stepwise_v19`
- do not run Gate 3
- do not enter RL

### Current Conclusion

v19 found and fixed a real root-cause bug in the Turn-3 training chain
(prompt overlength leading to zero supervision), but the first behavioral gate
still fails. The current failure mode is no longer "cannot train" or "parser
broken"; it is now a cleaner policy failure:

- a valid but collapsed final selector that always chooses `patchtst__keep`

