#!/usr/bin/env bash
# Block until this project may use the GPU (rules in docs/DECISIONS.md, 2026-11-18):
#   * no compute process on the GPU and the T-Rex job (PID $TREX_PID) has ended -> go now;
#   * otherwise the GPU must be idle for IDLE_MIN consecutive minutes (30; 10 after 2026-11-21 12:00);
#   * never start between 2026-11-21 01:00 and 12:00.
TREX_PID=${TREX_PID:-3700589}
idle=0
while true; do
    now=$(date +%s)
    if (( now >= $(date -d '2026-11-21 01:00' +%s) && now < $(date -d '2026-11-21 12:00' +%s) )); then
        idle=0; sleep 60; continue
    fi
    need=30; (( now >= $(date -d '2026-11-21 12:00' +%s) )) && need=10
    if [[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]]; then
        idle=$((idle + 1))
        if ! ps -p "$TREX_PID" >/dev/null || (( idle >= need )); then
            date '+GPU free at %F %H:%M'; free -g | sed -n 2p; exit 0
        fi
    else
        idle=0
    fi
    sleep 60
done
