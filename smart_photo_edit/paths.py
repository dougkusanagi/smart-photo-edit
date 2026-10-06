"""Pastas do usuário, por sistema operacional (Linux, Windows e macOS)."""
from __future__ import annotations

import os
import sys
from pathlib import Path

APP_DIR_NAME = "smart-photo-edit"


def user_data_dir() -> Path:
    """Onde ficam configuração, workflows importados e resultados temporários.

    Pode ser sobrescrito com a variável SPE_HOME (útil em testes e instalações portáteis).
    """
    override = os.environ.get("SPE_HOME")
    if override:
        return Path(override).expanduser()
    if sys.platform == "win32":
        base = os.environ.get("APPDATA") or str(Path.home() / "AppData" / "Roaming")
        return Path(base) / "SmartPhotoEdit"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "SmartPhotoEdit"
    base = os.environ.get("XDG_CONFIG_HOME") or str(Path.home() / ".config")
    return Path(base) / APP_DIR_NAME


def config_file() -> Path:
    return user_data_dir() / "config.json"


def user_workflows_dir() -> Path:
    return user_data_dir() / "workflows"


def results_dir() -> Path:
    return user_data_dir() / "results"


def log_file() -> Path:
    return user_data_dir() / "comfyui.log"


PACKAGE_DIR = Path(__file__).resolve().parent
WEB_DIR = PACKAGE_DIR / "web"
BUILTIN_WORKFLOWS_DIR = PACKAGE_DIR / "builtin_workflows"
