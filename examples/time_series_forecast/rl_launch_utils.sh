#!/bin/bash
# shellcheck shell=bash

is_true() {
    local value="${1:-}"
    case "${value,,}" in
        1|true|yes|on)
            return 0
            ;;
        *)
            return 1
            ;;
    esac
}

collect_stale_vllm_pids() {
    local -a names=("VLLM::Worker" "VLLM::EngineCore")
    local name
    local -a pids=()

    for name in "${names[@]}"; do
        while IFS= read -r pid; do
            [ -n "$pid" ] || continue
            pids+=("$pid")
        done < <(pgrep -u "$(id -u)" -f "^${name}$" || true)
    done

    if [ "${#pids[@]}" -eq 0 ]; then
        return 0
    fi

    printf '%s\n' "${pids[@]}" | sort -u
}

wait_for_pids_exit() {
    local timeout_s="$1"
    shift || true
    local -a pids=("$@")
    local deadline=$((SECONDS + timeout_s))

    while [ "${#pids[@]}" -gt 0 ] && [ "$SECONDS" -lt "$deadline" ]; do
        local -a alive=()
        local pid
        for pid in "${pids[@]}"; do
            if kill -0 "$pid" 2>/dev/null; then
                alive+=("$pid")
            fi
        done
        pids=("${alive[@]}")
        if [ "${#pids[@]}" -eq 0 ]; then
            return 0
        fi
        sleep 1
    done

    [ "${#pids[@]}" -eq 0 ]
}

cleanup_stale_vllm_processes() {
    if ! is_true "${RL_CLEANUP_STALE_VLLM:-1}"; then
        return 0
    fi

    local timeout_s="${RL_CLEANUP_WAIT_SECONDS:-20}"
    local -a stale_pids=()
    mapfile -t stale_pids < <(collect_stale_vllm_pids)

    if [ "${#stale_pids[@]}" -eq 0 ]; then
        return 0
    fi

    echo "[RL CLEANUP] stopping stale vLLM processes: ${stale_pids[*]}"
    kill "${stale_pids[@]}" 2>/dev/null || true

    if ! wait_for_pids_exit "$timeout_s" "${stale_pids[@]}"; then
        echo "[RL CLEANUP] force-killing stale vLLM processes: ${stale_pids[*]}"
        kill -9 "${stale_pids[@]}" 2>/dev/null || true
        wait_for_pids_exit 5 "${stale_pids[@]}" || true
    fi
}

cleanup_ray_and_stale_vllm() {
    ray stop --force >/dev/null 2>&1 || true
    cleanup_stale_vllm_processes
}
