## Route Proposal v18 Gate 1 Check

Date: 2026-03-31

### Scope

This report records the first Route Proposal Gate check for
`Route_Proposal_v18_修改测试验收方案.md`.

Checkpoint under evaluation:

- `artifacts/checkpoints/sft/qwen3-1.7b-etth1-route-proposal-v18-20260331_130630/global_step_8/huggingface`

Validation datasets:

- balanced: `dataset/routing_proposal_bootstrap_v18_routing_only_balanced/val.parquet`
- natural: `dataset/routing_proposal_bootstrap_v18_routing_only_natural/val.parquet`

Default expert:

- `itransformer`

### Gate 1 Thresholds

On `val_balanced`:

- `keep_vs_override_f1 >= 70%`
- `override_precision >= 70%`
- `override_recall >= 60%`
- `override_f1 >= 65%`
- `override_subset_exact_agreement >= 60%`
- `override_subset_top2_agreement >= 85%`

On `val_natural`:

- `delta_vs_default_mean < 0`
- `keep_default_share <= 85%`
- must not collapse to a single override model

### Probe Results

Balanced probe summary:

- `count = 192`
- `valid_tool_call_rate = 1.0`
- `predicted_label_distribution = {'override_to_arima': 188, 'keep_default': 4}`
- `requested_model_distribution = {'arima': 188, 'itransformer': 4}`
- `keep_vs_override_f1 = 35.10%`
- `override_precision = 50.00%`
- `override_recall = 97.92%`
- `override_f1 = 66.20%`
- `override_subset_exact_agreement = 32.29%`
- `override_subset_top2_agreement = 56.25%`
- `delta_vs_default_mean = -0.1030`
- `single_model_max_share = 97.92%`

Natural probe summary:

- `count = 192`
- `valid_tool_call_rate = 1.0`
- `predicted_label_distribution = {'override_to_arima': 192}`
- `requested_model_distribution = {'arima': 192}`
- `keep_vs_override_f1 = 17.24%`
- `override_precision = 20.83%`
- `override_recall = 100.00%`
- `override_f1 = 34.48%`
- `override_subset_exact_agreement = 65.00%`
- `override_subset_top2_agreement = 72.50%`
- `delta_vs_default_mean = 3.1322`
- `single_model_max_share = 100.00%`

Raw summaries:

- `artifacts/reports/v18_route_proposal_probe_step8_balanced_20260331_130630/routing_greedy_probe_summary.json`
- `artifacts/reports/v18_route_proposal_probe_step8_natural_20260331_130630/routing_greedy_probe_summary.json`

### Decision

Gate 1 failed.

### Why It Failed

- The policy collapsed to `override_to_arima` almost everywhere on `val_balanced`
  and completely on `val_natural`.
- `keep_vs_override_f1` missed badly on both validation views.
- `override_precision` remained far below threshold.
- `override_subset_top2_agreement` did not reach the required `85%`.
- On natural validation, `delta_vs_default_mean` became strongly positive
  (`+3.1322`), meaning the learned proposal policy was much worse than simply
  keeping the default expert.
- The run violated the anti-collapse rule by converging to a single override
  branch.

### Interpretation

v18 fixed the v17 supervision contradiction at the data layer, but the first
proposal warm-up still collapsed into another single-action attractor:

- not `keep_default`
- not `itransformer`
- but almost always `override_to_arima`

So the failure mode moved, yet the underlying route-task instability remains.
The model is still not learning a stable proposal policy; it is choosing one
globally safe action under the current prompt/state formulation.

### Consequence

Per the v18 plan:

- stop the current route proposal warm-up run
- do not build `full_stepwise_v18`
- do not run short full refresh SFT
- do not enter RL
