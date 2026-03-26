# Medium-Risk Refactor Execution Report (2026-03-23)

## Goal

This pass targeted the two most overloaded modules in the time-series stack:

- `recipe/time_series_forecast/utils.py`
- `recipe/time_series_forecast/reward.py`

The refactor preserved the existing public import surface while splitting internal responsibilities into smaller modules.

## Implemented Splits

### 1. Time-Series Utility Split

Added:

- `recipe/time_series_forecast/time_series_io.py`
  - parsing time-series text
  - dataframe conversion
  - prediction string formatting
  - compact tool-output formatting

- `recipe/time_series_forecast/diagnostic_features.py`
  - feature extraction
  - feature sanitization
  - human-readable feature formatting

Updated:

- `recipe/time_series_forecast/utils.py`
  - now acts as the compatibility-facing prediction/runtime module
  - keeps:
    - async/sync prediction entrypoints
    - HTTP client lifecycle
    - legacy environment compatibility fields
  - re-exports:
    - time-series IO helpers
    - diagnostic feature helpers

### 2. Reward Module Split

Added:

- `recipe/time_series_forecast/reward_protocol.py`
  - answer extraction
  - strict protocol parsing
  - recovery parsing
  - suffix repetition / line counting / answer-region helpers
  - numeric value extraction

- `recipe/time_series_forecast/reward_metrics.py`
  - decomposition helpers
  - MSE / normalization helpers
  - length / change-point / season-trend scoring helpers

Updated:

- `recipe/time_series_forecast/reward.py`
  - now keeps the top-level composite reward assembly
  - keeps turn-3 debug payload construction
  - re-exports protocol and metric helpers for compatibility

## Behavior Preservation Strategy

- Existing import paths remain valid:
  - `from recipe.time_series_forecast.utils import ...`
  - `from recipe.time_series_forecast.reward import ...`
- Compatibility-sensitive state remained in `utils.py`:
  - `_httpx_client`
  - `_httpx_client_loop`
  - `_get_httpx_client()`
- `compute_score()` behavior was not redesigned; only helper boundaries changed.

## New Internal Data Flow

### Utils side

1. `utils.py`
   - prediction runtime / service access
2. `time_series_io.py`
   - parse / format / compact IO helpers
3. `diagnostic_features.py`
   - extract / sanitize / format diagnostics

### Reward side

1. `reward.py`
   - final reward orchestration
   - debug payload emission
2. `reward_protocol.py`
   - protocol parsing and answer extraction
3. `reward_metrics.py`
   - scoring primitives

## Validation

Executed:

```bash
conda run -n cast-r1-ts pytest \
  tests/test_time_series_utils.py \
  tests/test_compact_protocol.py \
  tests/test_ray_agent_trainer_validation.py \
  tests/test_time_series_forecast_agent_flow.py \
  tests/test_sft_dataset_builder.py \
  tests/test_high_quality_sft_builder.py \
  tests/test_arima_runtime.py \
  -q
```

Result:

- `56 passed, 2 warnings`

## Deferred High-Risk Items

- Split `time_series_forecast_agent_flow.py` into state / execution / logging submodules
- Split `arft/ray_agent_trainer.py`
- Remove compatibility branches such as:
  - `use_legacy_worker_impl`
  - legacy environment fallbacks
- Normalize debug/report writing across trainer and reward layers
