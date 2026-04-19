#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUBMIT="$SCRIPT_DIR/submit_online.sh"

ENVS=(tool_hang transport)
STEPS=(0 500 1000 1500 2000)

for env in "${ENVS[@]}"; do
    for steps in "${STEPS[@]}"; do
        "$SUBMIT" "$env" "$steps"
        "$SUBMIT" "$env" "$steps" --residual
    done
done
