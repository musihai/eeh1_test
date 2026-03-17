#!/bin/bash
set -euo pipefail

if [ "${TRACE:-0}" = "1" ]; then
    set -x
fi
# =============================================================================
# Unified Time Series Prediction Service Startup Script
# =============================================================================
#
# This script starts the unified prediction service that loads all available
# models (Chronos2, PatchTST, iTransformer) on a dedicated GPU.
#
# Run this BEFORE starting the training script.
#
# Usage:
#   ./start_model_server.sh           # Uses GPU 0 by default, port 8994
#   ./start_model_server.sh 2         # Uses GPU 2
#   ./start_model_server.sh 3 8995    # Uses GPU 3, port 8995
#
# =============================================================================

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Get the project root directory (two levels up from script dir)
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"

# Add project root to PYTHONPATH so that imports like 'recipe.time_series_forecast...' work
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

resolve_profile_path() {
    local candidate="$1"
    if [ -z "$candidate" ]; then
        return 1
    fi
    if [ -f "$candidate" ]; then
        printf '%s\n' "$candidate"
        return 0
    fi
    if [ -f "$PROJECT_ROOT/$candidate" ]; then
        printf '%s\n' "$PROJECT_ROOT/$candidate"
        return 0
    fi
    return 1
}

PROFILE_PATH="${PROFILE_PATH:-}"
if [ -n "$PROFILE_PATH" ]; then
    RESOLVED_PROFILE_PATH="$(resolve_profile_path "$PROFILE_PATH")" || {
        echo "PROFILE_PATH not found: $PROFILE_PATH"
        exit 1
    }
    # shellcheck disable=SC1090
    source "$RESOLVED_PROFILE_PATH"
fi

# Configuration
GPU_ID=${1:-${SERVER_GPU_ID:-0}}          # Default: GPU 0
PORT=${2:-${MODEL_SERVICE_PORT:-8994}}    # Default: port 8994
HOST="${HOST:-0.0.0.0}"

echo "============================================================"
echo "  Unified Time Series Prediction Service"
echo "============================================================"
echo "  GPU:  $GPU_ID"
echo "  Port: $PORT"
echo "  Host: $HOST"
echo "============================================================"
echo ""
echo "  Available Models:"
echo "    - Chronos2:     Foundation model for time series"
echo "    - PatchTST:     Patch-based Transformer"
echo "    - iTransformer: Inverted Transformer"
echo ""
echo "  Models will be loaded if checkpoint exists."
echo "============================================================"

# Set CUDA device and start the server
cd "$SCRIPT_DIR"
CMD=(python model_server.py --host "$HOST" --port "$PORT" --device cuda)

if [ "${PRINT_CMD_ONLY:-0}" = "1" ]; then
    printf 'CUDA_VISIBLE_DEVICES=%q ' "$GPU_ID"
    printf '%q ' "${CMD[@]}"
    printf '\n'
    exit 0
fi

CUDA_VISIBLE_DEVICES=$GPU_ID "${CMD[@]}"

# =============================================================================
# Alternative: run with nohup in background
# =============================================================================
# Uncomment the lines below to run in background:
#
# CUDA_VISIBLE_DEVICES=$GPU_ID nohup python model_server.py --host $HOST --port $PORT --device cuda > model_server.log 2>&1 &
# echo "Server started in background. PID: $!"
# echo "Log file: model_server.log"
# echo ""
# echo "To check status: curl http://localhost:$PORT/health"
# echo "To see models:   curl http://localhost:$PORT/models"
