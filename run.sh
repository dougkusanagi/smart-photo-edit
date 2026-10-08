#!/usr/bin/env bash
# Uso: ./run.sh [--port 8765] [--no-browser]
set -euo pipefail
cd "$(dirname "$0")"
if [ ! -x .venv/bin/python ] || ! .venv/bin/python -c 'import aiohttp, PIL, smart_photo_edit' >/dev/null 2>&1; then
  bash install.sh
fi
exec .venv/bin/python -m smart_photo_edit "$@"
