"""Baixa os requisitos de um workflow (nós customizados e modelos) para uma instalação do ComfyUI."""
from __future__ import annotations

import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable

from .workflows import Workflow

CHUNK = 1024 * 256
UA = {"User-Agent": "smart-photo-edit/0.1"}


def download(url: str, dest: Path, progress: Callable[[int, int], None] | None = None) -> None:
    """Baixa com retomada (.part) e troca atômica no final."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    have = part.stat().st_size if part.exists() else 0
    headers = dict(UA)
    if have:
        headers["Range"] = f"bytes={have}-"
    req = urllib.request.Request(url, headers=headers)
    try:
        resp = urllib.request.urlopen(req, timeout=60)
    except urllib.error.HTTPError as exc:
        if exc.code == 416:  # já estava completo
            part.replace(dest)
            return
        raise
    with resp:
        resumed = resp.status == 206
        length = int(resp.headers.get("Content-Length") or 0)
        total = length + (have if resumed else 0)
        mode = "ab" if resumed else "wb"
        done = have if resumed else 0
        with open(part, mode) as fh:
            while chunk := resp.read(CHUNK):
                fh.write(chunk)
                done += len(chunk)
                if progress:
                    progress(done, total)
    part.replace(dest)


def _bar(name: str) -> Callable[[int, int], None]:
    last = {"pct": -1}

    def cb(done: int, total: int) -> None:
        pct = int(done * 100 / total) if total else 0
        if pct != last["pct"]:
            last["pct"] = pct
            sys.stdout.write(f"\r  {name}: {pct:3d}%  ({done / 1e6:,.0f} MB)")
            sys.stdout.flush()

    return cb


def install_requirements(wf: Workflow, comfyui_dir: Path, models_dir: Path | None = None) -> list[str]:
    """Instala o que tem URL e ainda não existe. Devolve avisos sobre itens manuais."""
    notes: list[str] = []
    models_dir = models_dir or comfyui_dir / "models"
    for node in wf.requires.get("custom_nodes", []):
        target = comfyui_dir / "custom_nodes" / node["name"]
        if target.exists():
            print(f"  ✓ nó customizado já instalado: {node['name']}")
            continue
        print(f"  ↓ nó customizado: {node['name']}")
        download(node["url"], target)
        notes.append(f"Reinicie o ComfyUI para carregar {node['name']}.")
    for model in wf.requires.get("models", []):
        target = models_dir / model["folder"] / model["filename"]
        if target.exists():
            print(f"  ✓ {model['folder']}/{model['filename']}")
            continue
        if not model.get("url"):
            notes.append(f"Baixe manualmente: {model['folder']}/{model['filename']} (sem URL conhecida).")
            continue
        print(f"  ↓ {model['folder']}/{model['filename']}")
        download(model["url"], target, _bar(model["filename"]))
        print()
    return notes
