# Conservative Refactor Execution Report (2026-03-23)

## Scope

This pass focused on low-risk cleanup and structural normalization that preserve:

- existing public entrypoints
- current training / inference semantics
- current dataset formats and runtime interfaces

Large rewrites and behavior-sensitive compatibility removals were intentionally deferred.

## Core Data Flow

1. Dataset build
   - `recipe/time_series_forecast/build_etth1_sft_dataset.py`
   - `recipe/time_series_forecast/build_etth1_high_quality_sft.py`
2. Runtime agent flow
   - `recipe/time_series_forecast/time_series_forecast_agent_flow.py`
3. Training / validation orchestration
   - `arft/ray_agent_trainer.py`
   - `arft/main_agent_ppo.py`
4. Launchers / configs
   - `examples/time_series_forecast/*`

## Audit Outcome

### Can delete directly

- `arft/agent_flow/tool_agent_flow.py`
  - empty file
  - no references in the repository

### Suggested merge / normalization

- `build_etth1_high_quality_sft.py`
  - was importing private helper names from `build_etth1_sft_dataset.py`
- `time_series_forecast_agent_flow.py`
  - duplicated episode-state reset logic in both `__init__` and `run`
- `tests/test_sft_dataset_builder.py`
  - was depending on private helper names instead of public helper surface

### Defer for later

- `recipe/time_series_forecast/utils.py`
  - still mixes parsing, formatting, diagnostics, service access, sync and async prediction
- `recipe/time_series_forecast/reward.py`
  - remains large and multi-responsibility
- `arft/main_agent_ppo.py`
  - still contains `use_legacy_worker_impl` compatibility branches
- environment-level legacy compatibility such as `CHRONOS_SERVICE_URL`
  - not removed in this pass to avoid behavior drift

## Implemented Changes

### Deleted

- `arft/agent_flow/tool_agent_flow.py`

Reason:
- dead file
- no runtime or test references

### Updated

- `arft/agent_flow/__init__.py`
  - removed side-effect placeholder export pattern
  - exported `SingleStepSingleTurnAgentFlow` explicitly through `__all__`

- `recipe/time_series_forecast/build_etth1_sft_dataset.py`
  - introduced public helper names:
    - `distribution_from_series`
    - `rebalance_train_turn3_targets`
    - `rebalance_train_stage_records`
  - retained backward-compatible aliases for local scripts/tests
  - updated internal use sites to prefer public helper names
  - removed one unused local binding by renaming it to `_routing_feature_snapshot`

- `recipe/time_series_forecast/build_etth1_high_quality_sft.py`
  - stopped importing private helper names from `build_etth1_sft_dataset.py`
  - now uses the public helper surface

- `recipe/time_series_forecast/time_series_forecast_agent_flow.py`
  - extracted duplicated reset logic into:
    - `_reset_prediction_state()`
    - `_reset_feature_state()`
    - `_reset_episode_state()`
  - `__init__` and `run()` now share the same reset path

- `tests/test_sft_dataset_builder.py`
  - switched to public helper names for rebalancing helpers

## Behavior Risk Assessment

### Structure-only changes

- deleting empty `tool_agent_flow.py`
- explicit export cleanup in `arft/agent_flow/__init__.py`
- switching callers from private helper names to public helper names
- test cleanup to stop depending on private helper names

### Low behavior risk changes

- shared state reset helpers in `time_series_forecast_agent_flow.py`

Why low risk:
- the reset values were kept the same
- stable configuration fields were intentionally not moved into the reset helper
- targeted regression tests passed after the change

## Intentionally Untouched

These remain because changing them is medium/high risk and can affect training semantics or deployment compatibility:

- `recipe/time_series_forecast/utils.py`
- `recipe/time_series_forecast/reward.py`
- `arft/main_agent_ppo.py`
- legacy environment compatibility:
  - `CHRONOS_SERVICE_URL`
  - `use_legacy_worker_impl`

## Validation

Executed:

```bash
conda run -n cast-r1-ts pytest \
  tests/test_sft_dataset_builder.py \
  tests/test_high_quality_sft_builder.py \
  tests/test_time_series_forecast_agent_flow.py \
  -q
```

Result:

- `32 passed, 1 warning`

## Recommended Next Pass

1. Split `recipe/time_series_forecast/utils.py` by responsibility
   - service client
   - time-series parsing / formatting
   - diagnostics
2. Split `recipe/time_series_forecast/reward.py`
   - protocol parsing
   - scoring
   - debug output
3. Remove obsolete compatibility branches only after explicit runtime validation
   - `CHRONOS_SERVICE_URL`
   - `use_legacy_worker_impl`
