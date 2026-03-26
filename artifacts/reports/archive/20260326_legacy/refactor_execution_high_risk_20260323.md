# High-Risk Refactor Execution Report (2026-03-23)

## Scope

This pass finished the remaining high-risk cleanup for the runtime and trainer entrypoints:

- `recipe/time_series_forecast/time_series_forecast_agent_flow.py`
- `arft/ray_agent_trainer.py`
- `arft/main_agent_ppo.py`

The class entrypoint and external interface were preserved:

- class name unchanged
- module path unchanged
- tool method names unchanged
- existing tests for direct internal methods remained valid

## What Changed

### Added

- `recipe/time_series_forecast/agent_flow_support.py`
  - stage / budget helpers
  - shared reward-tracking payload helpers
  - refinement metric helpers
  - turn-debug payload construction
  - prediction-tool debug payload construction

- `recipe/time_series_forecast/agent_flow_feature_tools.py`
  - feature tool specifications
  - shared metadata for state attribute, extractor, formatter, success log

- `arft/trainer_validation_support.py`
  - validation reward-manager adapter
  - numeric/string/bool coercion helpers
  - percentile / text-tail / value-extraction helpers
  - minimal eval debug file writer

- `arft/task_runner_support.py`
  - worker import specifications
  - actor/critic/reward worker selection logic
  - ref-policy registration predicate
  - resource-pool specification builder

### Updated

- `recipe/time_series_forecast/time_series_forecast_agent_flow.py`
  - removed wildcard prompt import
  - switched to explicit imports
  - delegated stage / budget / shared tracking logic to `agent_flow_support.py`
  - replaced repeated feature-tool implementations with a generic `_run_feature_tool()` path
  - replaced repeated feature-tool state checks with spec-driven logic
  - moved refinement metric calculations to shared helper functions
  - moved debug payload construction to shared helper functions

- `arft/ray_agent_trainer.py`
  - delegated validation reward-manager bridging to `trainer_validation_support.py`
  - delegated eval debug export helpers to `trainer_validation_support.py`
  - retained the public trainer API and validation flow

- `arft/main_agent_ppo.py`
  - added a local `_register_worker()` helper to centralize role registration
  - delegated worker selection / pool building to `task_runner_support.py`
  - kept `TaskRunner` focused on orchestration instead of branch-heavy worker resolution

## Structure After Refactor

### Still in `time_series_forecast_agent_flow.py`

- main `run()` loop
- prompt/token/sampling integration
- workflow validation
- prediction-tool execution orchestration
- parse-failure output handling

### Extracted out

- feature tool registry and per-tool metadata
- episode bookkeeping helpers
- reward-tracking field assembly
- refinement comparison metrics
- agent-turn debug payload building

### Extracted out of `ray_agent_trainer.py`

- validation reward-manager adaptation
- eval debug aggregation helpers
- text/metric coercion helpers used only by validation exports

### Extracted out of `main_agent_ppo.py`

- actor worker selection by strategy / rollout mode / legacy-worker setting
- critic worker selection
- reward-model worker selection and pool assignment
- ref-policy registration condition
- resource-pool specification assembly

## Why This Was Worth Doing

Before this pass, the remaining high-risk modules mixed orchestration with branch-heavy helper logic:

- `time_series_forecast_agent_flow.py` mixed stage progression, feature tool registry, reward tracking, and debug assembly
- `ray_agent_trainer.py` mixed validation orchestration with eval-export helper utilities
- `main_agent_ppo.py` mixed trainer orchestration with worker-strategy and pool-resolution branches

That made small fixes risky because orchestration code and compatibility logic were interleaved.

After this pass:

- the runtime loop is easier to follow
- repeated feature-tool logic is centralized
- debug and refinement metrics can evolve independently from the main loop
- helper logic can now be unit-tested or reused without copying class methods
- trainer validation/export logic is isolated from the trainer lifecycle
- `TaskRunner` now reads as a startup pipeline instead of a worker-resolution switchboard

## Behavior Risk

This was a high-risk refactor because the agent flow is on the critical path for:

- SFT/validation debug
- RL rollout behavior
- workflow gating

Risk was controlled by:

- preserving the class/module interface
- keeping `run()` in place
- only extracting helper logic that can be expressed as pure functions
- preserving internal method names that tests call directly

## Validation

Executed in stages:

```bash
python -m py_compile arft/trainer_validation_support.py arft/ray_agent_trainer.py
conda run -n cast-r1-ts pytest \
  tests/test_ray_agent_trainer_validation.py \
  tests/test_time_series_forecast_agent_flow.py \
  tests/test_compact_protocol.py \
  -q
```

Result:

- `30 passed, 1 warning`

```bash
python -m py_compile arft/task_runner_support.py arft/main_agent_ppo.py
conda run -n cast-r1-ts pytest \
  tests/test_task_runner_support.py \
  tests/test_main_agent_ppo.py \
  tests/test_main_agent_ppo_config.py \
  tests/test_ray_agent_trainer_validation.py \
  -q
```

Result:

- `20 passed, 1 warning`

Final broader regression:

```bash
conda run -n cast-r1-ts pytest \
  tests/test_task_runner_support.py \
  tests/test_main_agent_ppo.py \
  tests/test_main_agent_ppo_config.py \
  tests/test_ray_agent_trainer_validation.py \
  tests/test_time_series_forecast_agent_flow.py \
  tests/test_final_answer_parsing.py \
  tests/test_compact_protocol.py \
  tests/test_sft_dataset_builder.py \
  tests/test_high_quality_sft_builder.py \
  tests/test_time_series_utils.py \
  tests/test_arima_runtime.py \
  -q
```

Result:

- `93 passed, 2 warnings`

## Still Deferred

- split `run()` itself into stage handlers
- remove `use_legacy_worker_impl` compatibility from `main_agent_ppo.py` and downstream workers
- simplify remaining environment/runtime compatibility branches
