#!/usr/bin/env bash
# Cria o ambiente na primeira vez e inicia o app. Uso: ./run.sh [--port 8765] [--no-browser]
set -euo pipefail
cd "$(dirname "$0")"
if [ ! -x .venv/bin/python ]; then
  echo "Criando ambiente Python (.venv)…"
  if command -v uv >/dev/null 2>&1; then
    uv venv .venv && uv pip install --python .venv/bin/python -e .
  else
    python3 -m venv .venv && .venv/bin/python -m pip install -e .
  fi
fi
exec .venv/bin/python -m smart_photo_edit "$@"
