#!/usr/bin/env bash
# Atualiza o código e as dependências do app. Feche o aplicativo antes de usar.
set -euo pipefail
cd "$(dirname "$0")"
if [ -x .venv/bin/python ]; then
  .venv/bin/python scripts/update_app.py
else
  "${SPE_PYTHON:-python3}" scripts/update_app.py
fi
bash install.sh
