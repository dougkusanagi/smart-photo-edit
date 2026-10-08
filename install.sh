#!/usr/bin/env bash
# Prepara ou repara o ambiente do aplicativo; os modelos são preparados pela interface.
set -euo pipefail
cd "$(dirname "$0")"
if command -v uv >/dev/null 2>&1; then
  if [ ! -x .venv/bin/python ]; then
    uv venv --python "${SPE_PYTHON:-python3}" .venv
  fi
  uv pip install --python .venv/bin/python -e .
else
  if [ ! -x .venv/bin/python ]; then
    "${SPE_PYTHON:-python3}" -m venv .venv
  fi
  if ! .venv/bin/python -m pip --version >/dev/null 2>&1; then
    .venv/bin/python -m ensurepip --upgrade
  fi
  .venv/bin/python -m pip install -e .
fi
echo 'Instalação concluída. Execute ./run.sh para abrir o aplicativo.'
