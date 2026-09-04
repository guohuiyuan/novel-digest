#!/bin/bash

set -u

cd "$(dirname "$0")" || exit 1

if ! command -v uv >/dev/null 2>&1; then
    echo "[ERROR] uv is not installed."
    echo "[ERROR] Install uv: pip install uv  (or use the official installer)."
    exit 1
fi

uv run main.py
EXIT_CODE=$?

echo
if [ -t 0 ]; then
    printf 'Task finished. Press any key to exit...'
    read -r -n 1 -s
    echo
else
    echo "Task finished."
fi

exit "$EXIT_CODE"
