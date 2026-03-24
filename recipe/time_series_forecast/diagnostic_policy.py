from __future__ import annotations

FEATURE_TOOL_ORDER = (
    "extract_basic_statistics",
    "extract_within_channel_dynamics",
    "extract_forecast_residuals",
    "extract_data_quality",
    "extract_event_summary",
)


def select_feature_tool_names(values: list[float]) -> list[str]:
    del values
    # The current paper-aligned mainline uses a full diagnostic pass before
    # routing so that SFT trajectories, runtime prompts, and workflow
    # validation all share the same memory state.
    return list(FEATURE_TOOL_ORDER)


def plan_diagnostic_tool_batches(
    required_tool_names: list[str] | tuple[str, ...],
    *,
    max_parallel_calls: int = 5,
) -> list[list[str]]:
    required = [name for name in FEATURE_TOOL_ORDER if name in set(required_tool_names or [])]
    if not required:
        required = ["extract_basic_statistics"]

    batch_cap = max(1, int(max_parallel_calls or 1))
    batches: list[list[str]] = []
    remaining = list(required)

    while remaining:
        batches.append(remaining[:batch_cap])
        remaining = remaining[batch_cap:]

    return [batch for batch in batches if batch]
