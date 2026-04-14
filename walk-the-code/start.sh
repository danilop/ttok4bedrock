#!/usr/bin/env bash
set -e
PORT="${1:-8000}"
CONFIG="$(cd "$(dirname "$0")" && pwd)/config.json"
WTC='uvx --from "walk-the-code @ git+https://github.com/danilop/walk-the-code"'
echo "Opening walkthrough at http://localhost:$PORT"
(sleep 1 && python3 -m webbrowser "http://localhost:$PORT") &
exec $WTC wtc-serve --config "$CONFIG" "$PORT"
