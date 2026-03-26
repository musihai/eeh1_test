# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Optional, Tuple


ETTH1_TARGET_COLUMN = "OT"
ETTH1_FEATURE_COLUMNS = ("HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", ETTH1_TARGET_COLUMN)
ETTH1_COVARIATE_COLUMNS = ETTH1_FEATURE_COLUMNS[:-1]


def _parse_env_int(name: str) -> Optional[int]:
    value = os.environ.get(name)
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _load_lengths_from_base_yaml() -> Tuple[Optional[int], Optional[int]]:
    base_path = Path(__file__).parent / "base.yaml"
    if not base_path.exists():
        return None, None
    try:
        text = base_path.read_text(encoding="utf-8")
    except Exception:
        return None, None

    lookback_match = re.search(r"lookback_window:\s*(\d+)", text)
    horizon_match = re.search(r"forecast_horizon:\s*(\d+)", text)
    lookback = int(lookback_match.group(1)) if lookback_match else None
    horizon = int(horizon_match.group(1)) if horizon_match else None
    return lookback, horizon


def get_default_lengths() -> Tuple[int, int]:
    """
    Resolve default lookback/forecast lengths.
    Priority: ENV overrides -> base.yaml -> 96.
    """
    lookback = _parse_env_int("LOOKBACK_WINDOW")
    horizon = _parse_env_int("FORECAST_HORIZON")

    if lookback is None or horizon is None:
        base_lookback, base_horizon = _load_lengths_from_base_yaml()
        if lookback is None:
            lookback = base_lookback
        if horizon is None:
            horizon = base_horizon

    if lookback is None:
        lookback = 96
    if horizon is None:
        horizon = 96

    return lookback, horizon


def load_model_config_json(model_name: str) -> dict[str, Any]:
    config_path = Path(__file__).parent / "models" / str(model_name).strip().lower() / "config.json"
    if not config_path.exists():
        return {}
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def expected_model_input_width(model_name: str) -> Optional[int]:
    config = load_model_config_json(model_name)
    value = config.get("enc_in")
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def expected_model_seq_len(model_name: str) -> Optional[int]:
    config = load_model_config_json(model_name)
    value = config.get("seq_len")
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None
