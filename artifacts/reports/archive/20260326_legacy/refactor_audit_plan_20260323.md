# Refactor Audit Plan 2026-03-23

## Scope

This audit focuses on the active time-series training and inference path:

- `recipe/time_series_forecast/*`
- `examples/time_series_forecast/*`
- `arft/*`
- related tests under `tests/*`

The vendored `verl/` tree is treated conservatively. Only project-local integration points that directly affect the time-series pipeline are candidates in this pass.

## Main Findings

### Can delete directly

- `arft/agent_flow/tool_agent_flow.py`
  - Empty file.
  - No code references in the repository.
  - Safe to remove.

### Should merge or normalize

- `recipe/time_series_forecast/build_etth1_sft_dataset.py`
  - Large mixed-responsibility module.
  - Combines:
    - routing heuristics
    - tool formatting
    - turn-3 target construction
    - parquet conversion
    - train rebalancing
    - metadata writing
    - CLI entrypoint
  - Also exposes helpers that are consumed by `build_etth1_high_quality_sft.py`, but some of those helpers are still treated as private.

- `recipe/time_series_forecast/build_etth1_high_quality_sft.py`
  - Imports private helpers from `build_etth1_sft_dataset.py`.
  - This is a maintainability smell and creates fragile cross-module coupling.

- `recipe/time_series_forecast/time_series_forecast_agent_flow.py`
  - Duplicates state-reset logic between `__init__` and `run`.
  - Keeps too much mutable state in one class without internal grouping.
  - Still readable, but clearly patch-accumulated.

- `recipe/time_series_forecast/utils.py`
  - Mixed responsibilities:
    - async/sync prediction entrypoints
    - parsing
    - formatting
    - diagnostics
    - service communication
  - Contains historical compatibility branches and both async/sync service wrappers.

### Suggest rewrite or staged extraction

- `build_etth1_sft_dataset.py`
  - Not a full rewrite in this pass.
  - Should be split later into:
    - routing policy helpers
    - turn-3 target builder
    - parquet writer / metadata builder

- `time_series_forecast_agent_flow.py`
  - Not a full rewrite in this pass.
  - Should be staged into:
    - episode state
    - prompt/tool-schema assembly
    - tool execution
    - final answer parsing / reward handoff

### Historical compatibility or patch-style paths

- `arft/main_agent_ppo.py`
  - `use_legacy_worker_impl` branches remain.
  - They are still active trainer paths, so not safe to remove in a conservative pass.

- `recipe/time_series_forecast/utils.py`
  - Legacy `CHRONOS_SERVICE_URL` compatibility remains.
  - This is removable, but it changes environment-level compatibility and should be treated as medium risk.

- `tests/test_ray_agent_trainer_validation.py`
  - Still contains a compatibility assertion for ignored `TS_TURN3_DEBUG_FILE`.
  - This is test-only compatibility residue.

## Risk Classification

### Low risk

- Delete empty file `arft/agent_flow/tool_agent_flow.py`
- Clean `arft/agent_flow/__init__.py` exports
- Stop importing private SFT builder helpers from `build_etth1_high_quality_sft.py`
- Rename or expose currently-private shared helpers via stable public names while keeping aliases
- Remove obviously unused locals and small structural duplication
- Deduplicate `TimeSeriesForecastAgentFlow` state-reset code without changing behavior

### Medium risk

- Remove environment-level legacy compatibility such as `CHRONOS_SERVICE_URL`
- Extract routing heuristics into a dedicated module and keep compatibility wrappers
- Split metadata / rebalancing helpers out of `build_etth1_sft_dataset.py`

### High risk

- Rewrite `time_series_forecast_agent_flow.py`
- Rewrite `reward.py`
- Rewrite `utils.py` into multiple modules in one pass
- Remove `use_legacy_worker_impl` support from trainer code
- Change CLI flags, dataset metadata schema, or training entrypoints

## Validation Strategy

- Run focused unit tests for every touched module
- Keep public interfaces stable where possible
- Preserve existing filenames for main entrypoints
- Avoid algorithmic changes in this pass
- Re-check SFT builder outputs and metadata when touching shared builder helpers

## Planned Execution Order

1. Low-risk cleanup and export normalization
2. Conservative structure cleanup in `time_series_forecast_agent_flow.py`
3. Re-run targeted tests
4. Stop and report remaining medium/high-risk debt without forcing it into this pass
