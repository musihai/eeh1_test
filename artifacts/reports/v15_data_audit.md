# Cast-R1-TS v1.5 Stage 0 Data Audit

Date: 2026-03-31

Scope:
- Curated teacher dataset: `dataset/ett_sft_etth1_runtime_teacher200_paper_same2/{train,val,test}_curated.jsonl`
- Runtime expert path: `recipe/time_series_forecast/time_series_forecast_agent_flow.py`
- Teacher builder path: `recipe/time_series_forecast/build_etth1_high_quality_sft.py`
- Shared expert dispatch: `recipe/time_series_forecast/utils.py`

## Audit Verdict

- Recommend entering Stage 1: `Yes`
- Recommend skipping directly to Stage 2/3 training: `No`

Reason:
- Split leakage was not detected when using stable sample identity (`uid`) or prompt content.
- Teacher labels are internally legal.
- But route signal quality is not yet strong enough for immediate retraining:
  - curated `selected_prediction_model == reference_teacher_model` agreement is far below the v1.5 target
  - current curated train `local_refine` ratio is below the v1.5 target
  - `routing_confidence_tier` is absent in current curated artifacts, so confidence-aware filtering cannot yet be validated offline

## A. Expert Route Value

Reference-teacher winner distribution across all curated splits:

| model | count | share |
| --- | ---: | ---: |
| itransformer | 136 | 34.69% |
| patchtst | 105 | 26.79% |
| chronos2 | 84 | 21.43% |
| arima | 67 | 17.09% |

Assessment:
- No single expert exceeds 80%.
- Route learning still has value; the task has not collapsed into a trivial single-expert policy.

## B. Split Leakage Check

Checked on:
- `uid`
- first `raw_prompt` message content

Results:

| intersection | overlap count |
| --- | ---: |
| train ∩ val (`uid`) | 0 |
| train ∩ test (`uid`) | 0 |
| val ∩ test (`uid`) | 0 |
| train ∩ val (`raw_prompt`) | 0 |
| train ∩ test (`raw_prompt`) | 0 |
| val ∩ test (`raw_prompt`) | 0 |

Note:
- `index` is split-local and shows false overlap across files; it is not a safe global identity field.

Verdict:
- Pass

## C. Teacher Label Legality

Teacher legality was audited with current available curated fields:
- `reference_teacher_model`
- `reference_teacher_error`
- `teacher_prediction_text`
- `teacher_prediction_source`

Results:

| split | n | teacher missing | bad teacher length | bad teacher error | selected==reference_teacher |
| --- | ---: | ---: | ---: | ---: | ---: |
| train | 200 | 0 | 0 | 0 | 0.4850 |
| val | 64 | 0 | 0 | 0 | 0.2500 |
| test | 128 | 0 | 0 | 0 | 0.2891 |

Interpretation:
- Teacher itself is legal.
- But curated `selected_prediction_model` is still far from the reference teacher in the current teacher200 artifacts.
- This confirms the need for v1.5 Stage 2 rebuild with `routing_label_source=reference_teacher`.

Verdict:
- Teacher legality: Pass
- v1.5 agreement target (`>= 0.98`): Fail on current curated artifacts

## D. Route Confidence Tier Distribution

Expected field in v1.5 audit:
- `routing_confidence_tier`

Observed in current curated artifacts:
- Field absent

Implication:
- `mid/high` confidence filtering and repeat-factor design from v1.5 cannot be validated against the current teacher200 artifacts.
- This should be treated as a Stage 2 rebuild requirement, not a silent assumption.

Verdict:
- Not auditable from current artifacts

## E. Refine Sample Imbalance

Turn-3 target type distribution:

| split | validated_keep | local_refine | local_refine ratio |
| --- | ---: | ---: | ---: |
| train | 153 | 47 | 23.50% |
| val | 59 | 5 | 7.81% |
| test | 117 | 11 | 8.59% |

Top train `refine_ops_signature` values:

| signature | count |
| --- | ---: |
| none | 153 |
| isolated_spike_smoothing | 21 |
| local_level_adjust | 21 |
| flat_tail_repair | 2 |
| local_slope_adjust | 2 |
| amplitude_clip | 1 |

Interpretation:
- Current train `local_refine` ratio is below the v1.5 target of 30%.
- Current curated data is still keep-dominant.

Verdict:
- Fail on current teacher200 artifacts

## F. Expert Version Consistency

Observed code path:
- Teacher builder calls shared dispatcher `predict_time_series_async(...)` in `recipe/time_series_forecast/build_etth1_high_quality_sft.py`
- Runtime agent flow also calls shared dispatcher `predict_time_series_async(...)` in `recipe/time_series_forecast/time_series_forecast_agent_flow.py`
- Shared dispatcher lives in `recipe/time_series_forecast/utils.py`

Shared runtime model set:
- `chronos2`
- `arima`
- `patchtst`
- `itransformer`

Available model artifacts:
- `recipe/time_series_forecast/models/patchtst/provenance.json`
- `recipe/time_series_forecast/models/itransformer/provenance.json`
- `recipe/time_series_forecast/models/chronos-2/`
- `arima` is local code-path only, implemented in `utils.py`

Assessment:
- Teacher construction and runtime inference use the same model-name surface and the same dispatch entrypoint.
- PatchTST and iTransformer provenance files are present and point to the same retrained expert root under `artifacts/expert_retrain_multivar_20260324`.
- Chronos2 and ARIMA are consistent at the code-path level, but current curated artifacts do not embed a per-sample provenance stamp, so full per-run checkpoint provenance is still weaker than ideal.

Verdict:
- Pass at code-path level
- Provenance recording should still be strengthened in future rebuild outputs

## Summary Against v1.5 Stage 0 Gates

| audit item | result |
| --- | --- |
| expert routing still meaningful | Pass |
| split overlap | Pass |
| teacher missing/error/length legality | Pass |
| curated selected vs reference teacher agreement | Fail |
| route confidence distribution auditable | Fail |
| local_refine ratio >= 30% | Fail |
| expert version consistency | Pass at code-path level |

## Recommendation

Proceed with Stage 1 code fixes now:
- unify Turn 3 training/runtime protocol
- remove routing prompt leakage
- lower KL and add small entropy
- weaken Turn 3 keep bias

Do not treat the current teacher200 curated artifacts as v1.5-ready training data.
Stage 2 must rebuild new curated + stepwise datasets with:
- `routing_label_source=reference_teacher`
- `turn3_target_mode=engineering_refine`
- stronger `local_refine` coverage
- explicit route-confidence metadata
