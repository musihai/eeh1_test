## Route Relaxation v17 Gate Check

Date: 2026-03-31

### Scope

This report records the first Route Override Gate check for
`Route_Relaxation_v17_修改测试验收方案.md`.

Checkpoint under evaluation:

- `artifacts/checkpoints/sft/qwen3-1.7b-etth1-route-override-v17-20260331_121913/global_step_8/huggingface`

Validation dataset:

- `dataset/routing_override_bootstrap_v17_routing_only/val.parquet`

Default expert:

- `itransformer`

Validation default baseline from `routing_override_bootstrap_v17/val.jsonl`:

- `default_expert_mean_mse = 5.892036220156792`
- `default_expert_mean_regret = 2.005855424804145`

### Gate 1 Thresholds

- `delta_vs_default_mean < 0`
- `override_precision >= 60%`
- `override_recall >= 50%`
- `override_f1 >= 55%`
- `override_subset_exact_agreement >= 50%`
- `override_subset_top2_agreement >= 80%`
- must not collapse to all `keep_default`
- must not collapse to a single override branch

### Probe Results

Core summary from
`artifacts/reports/v17_route_override_probe_step8_20260331_121913/routing_greedy_probe_summary.json`:

- `count = 192`
- `valid_tool_call_rate = 1.0`
- `predicted_label_distribution = {'keep_default': 192}`
- `requested_model_distribution = {'itransformer': 192}`
- `keep_default_share = 1.0`
- `override_share = 0.0`
- `delta_vs_default_mean = 0.0`
- `override_precision = 0.0`
- `override_recall = 0.0`
- `override_f1 = 0.0`
- `override_subset_count = 58`
- `override_subset_exact_agreement = 0.0`
- `override_subset_top2_agreement = 0.06896551724137931`
- `mean_route_regret = 2.005855424804145`

Additional override-subset diagnostics from
`routing_greedy_probe_samples.jsonl`:

- `override_route_regret_p50 = 3.2698830733333333`
- `override_route_regret_p90 = 10.112800785833334`
- `override_route_regret_max = 11.459509819583335`

### Decision

Gate 1 failed.

### Why It Failed

- The policy collapsed to `keep_default` on all 192 validation samples.
- `delta_vs_default_mean` stayed exactly at `0.0`, so the learned route policy
  did not beat the fixed default baseline.
- All override detection metrics were `0.0`.
- On the 58 gold override windows, the model never triggered an override.
- The override subset top-2 agreement was only `6.90%`, far below the required
  `80%`.

### Interpretation

The new v17 route schema is being followed syntactically:

- tool-call validity is `100%`
- the model emits the new `route_time_series` action cleanly

But behaviorally, the policy has collapsed into the safest possible action:

- always choose `keep_default`

This makes the headline `overall_exact_agreement = 69.79%` misleading, because
it is driven almost entirely by the class prior of the validation set
(`keep_default = 134 / 192`), not by successful override behavior.

### Consequence

Per the v17 plan:

- stop the current route-override warm-up run
- do not build `full_stepwise_v17`
- do not run short full refresh SFT
- do not enter RL

### Current Conclusion

Under the current v17 state/action formulation, the route policy still does not
learn the override subtask. The main failure mode has changed from v16's
single-expert exact routing collapse to a new v17 collapse:

- a fully well-formed but always-`keep_default` route policy
