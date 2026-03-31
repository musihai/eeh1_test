## Route Proposal v18 Gate 0 Check

Date: 2026-03-31

### Scope

This report records the data-gate audit for
`Route_Proposal_v18_修改测试验收方案.md`.

Dataset under evaluation:

- `dataset/routing_proposal_bootstrap_v18`

Default expert:

- `itransformer`

### Gate 0 Thresholds

- `contradictory_keep_count = 0`
- train keep/override ratio in `[45%, 55%]`
- each override model in train `>= 64`
- each override model in `val_balanced >= 16`
- no split overlap across `train / val_balanced / val_natural / test`

### Learned Thresholds

- `tau_keep = 0.05`
- `tau_margin = 0.08`
- `tau_override_model(arima) = 0.5495660110589313`
- `tau_override_model(chronos2) = 0.47706335765944097`
- `tau_override_model(patchtst) = 0.35`

### Core Audit Results

- `train_contradictory_keep_count = 0`
- `val_contradictory_keep_count = 0`
- `test_contradictory_keep_count = 0`

Train selection:

- count = `768`
- keep / override = `384 / 384`
- train keep share = `50.00%`
- train override share = `50.00%`
- override distribution = `patchtst 128 / arima 128 / chronos2 128`

Validation balanced selection:

- count = `192`
- keep / override = `96 / 96`
- override distribution = `patchtst 32 / arima 32 / chronos2 32`

Validation natural selection:

- count = `192`
- keep / override = `152 / 40`
- override distribution = `patchtst 12 / arima 26 / chronos2 2`

Test selection:

- count = `384`
- keep / override = `273 / 111`
- override distribution = `patchtst 16 / arima 60 / chronos2 35`

### Split Overlap Audit

- `train ∩ val_balanced = 0`
- `train ∩ val_natural = 0`
- `val_balanced ∩ val_natural = 0`
- `train ∩ test = 0`
- `val_balanced ∩ test = 0`
- `val_natural ∩ test = 0`

### Decision

Gate 0 passed.

### Interpretation

The v18 triage builder cleared the key supervision contradiction that broke
v17:

- no high-gain non-default samples remain inside `must_keep`
- route warm-up train is explicitly balanced `50 / 50`
- override supervision now covers all three non-default experts with adequate
  support
- balanced and natural validation sets are disjoint, so Gate 1 can separately
  measure classification ability and real-world default-regret reduction

### Consequence

Per the v18 plan:

- proceed to `routing_proposal_bootstrap_v18_routing_only`
- start route proposal warm-up SFT
- evaluate Gate 1 on both `val_balanced` and `val_natural`
