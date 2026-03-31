## Route Rescue v16 Gate Check

Date: 2026-03-31

### Scope

This report records the Stage 3.5 route-rescue acceptance check against the
criteria in `Route_Rescue_v16_修改测试验收方案_v2.md`.

### Gate 1 Thresholds

- overall exact agreement >= 50%
- high-confidence exact agreement >= 70%
- top-2 agreement >= 85%
- all 4 experts must be predicted
- no single expert share > 70%
- high-confidence exact agreement must improve by at least 30 points over the
  current full v15 baseline

### Probe Results

| model/checkpoint | overall exact | high-conf exact | top-2 | mean regret | max share | coverage | distribution |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| full v15 baseline | 30.47% | 30.47% | 42.19% | 3.8614 | 61.72% | 2 | arima 79 / chronos2 49 |
| route repair from full v15 | 25.00% | 25.00% | 43.75% | 4.0444 | 47.66% | 3 | arima 61 / patchtst 43 / itransformer 24 |
| old routing-only v15 | 25.00% | 25.00% | 51.56% | 3.3991 | 100.00% | 1 | patchtst 128 |
| route repair from base | 25.00% | 25.00% | 61.72% | 2.5934 | 100.00% | 1 | itransformer 128 |

### Decision

Gate 1 failed.

- The best route-repair attempt did not reach the minimum exact-agreement
  thresholds.
- High-confidence exact agreement did not improve; it fell from 30.47% to
  25.00%.
- The from-base repair improved regret, but collapsed to a single expert and
  still failed the agreement and coverage thresholds.

### Consequence

Per the v16 plan:

- do not build `full_stepwise_v16`
- do not run short full refresh SFT
- do not enter RL

### Interpretation

The route bootstrap dataset and supervision cleanup were applied correctly, but
route policy did not become learnable enough under the current state/action
formulation. The current evidence points to a route-task structure limitation
rather than a simple warm-start contamination issue.
