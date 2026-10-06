"""Configuração persistida em JSON (gravação atômica, tolerante a arquivo corrompido)."""
from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

from . import paths

DEFAULT_WORKFLOW = "qwen21-viggle-turbo"


@dataclass
class Settings:
    engine_mode: str = "managed"
    comfy_url: str = "http://127.0.0.1:8188"
    # Comando para iniciar o ComfyUI quando ele não estiver rodando (opcional).
    # Ex.: "/caminho/python /caminho/ComfyUI/main.py --port 8188"
    comfy_command: str = ""
    comfy_cwd: str = ""
    comfy_autostart: bool = False
    workflow: str = DEFAULT_WORKFLOW
    # Valores escolhidos para os parâmetros extras de cada workflow: {workflow_id: {chave: valor}}
    model_choices: dict[str, dict[str, Any]] = field(default_factory=dict)
    workflow_params: dict[str, dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Settings":
        known = {f.name for f in fields(cls)}
        clean = {k: v for k, v in data.items() if k in known}
        s = cls(**clean)
        if s.engine_mode not in ("managed", "external"):
            s.engine_mode = "managed"
        s.comfy_url = normalize_url(s.comfy_url)
        if not isinstance(s.workflow_params, dict):
            s.workflow_params = {}
        if not isinstance(s.model_choices, dict):
            s.model_choices = {}
        from .models import choices
        from .workflows import WorkflowError
        valid = {}
        for wf_id, values in s.model_choices.items():
            try:
                valid[wf_id] = choices(wf_id, values)
            except WorkflowError:
                continue
        s.model_choices = valid
        return s


def normalize_url(url: str) -> str:
    url = (url or "").strip().rstrip("/")
    if not url:
        return Settings.comfy_url
    if "://" not in url:
        url = "http://" + url
    return url


def load(path: Path | None = None) -> Settings:
    path = path or paths.config_file()
    data: dict[str, Any] = {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            data = {}
    except (OSError, ValueError):
        data = {}
    settings = Settings.from_dict(data)
    env_url = os.environ.get("SPE_COMFY_URL")
    if env_url:
        settings.comfy_url = normalize_url(env_url)
        settings.engine_mode = "external"
    return settings


def save(settings: Settings, path: Path | None = None) -> None:
    path = path or paths.config_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".config-", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(settings.to_dict(), fh, indent=2, ensure_ascii=False)
        os.replace(tmp, path)  # atômico também no Windows
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
