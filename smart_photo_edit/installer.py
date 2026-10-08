"""Baixa os requisitos de um workflow (nós customizados e modelos) para uma instalação do ComfyUI."""
from __future__ import annotations

import hashlib
import json
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
        if exc.code == 416:
            expected = exc.headers.get("Content-Range", "").split("/")[-1]
            if expected.isdigit() and have == int(expected):
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
    if total and done != total:
        raise OSError("Download incompleto; tente novamente para retomar.")
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


def install_requirements(wf: Workflow, comfyui_dir: Path, models_dir: Path | None = None, progress: Callable[[int, int], None] | None = None,
                         on_model: Callable[[dict], None] | None = None) -> list[str]:
    """Instala o que tem URL e ainda não existe. Devolve avisos sobre itens manuais."""
    notes: list[str] = []
    models_dir = models_dir or comfyui_dir / "models"
    def safe_target(root: Path, *parts: str) -> Path:
        target = root.joinpath(*parts)
        if not target.resolve().is_relative_to(root.resolve()):
            raise ValueError("Requisito do workflow aponta para fora da pasta do motor.")
        return target

    for node in wf.requires.get("custom_nodes", []):
        target = safe_target(comfyui_dir / "custom_nodes", node["name"])
        if target.exists():
            print(f"  ✓ nó customizado já instalado: {node['name']}")
            continue
        print(f"  ↓ nó customizado: {node['name']}")
        download(node["url"], target, progress)
        notes.append(f"Reinicie o ComfyUI para carregar {node['name']}.")
    for model in wf.requires.get("models", []):
        target = safe_target(models_dir, model["folder"], model["filename"])
        if target.exists() and not model.get("sha256"):
            print(f"  ✓ {model['folder']}/{model['filename']}")
            continue
        if not model.get("url"):
            notes.append(f"Baixe manualmente: {model['folder']}/{model['filename']} (sem URL conhecida).")
            continue
        if not target.exists():
            print(f"  ↓ {model['folder']}/{model['filename']}")
            if on_model:
                on_model(model)
            download(model["url"], target, progress or _bar(model["filename"]))
        if model.get("sha256"):
            receipt = target.with_name(target.name + '.verified.json')
            identity = [model['sha256'], target.stat().st_size, target.stat().st_mtime_ns]
            try:
                if json.loads(receipt.read_text()) == identity:
                    continue
            except (OSError, ValueError):
                pass
            digest = hashlib.sha256()
            with target.open("rb") as fh:
                while chunk := fh.read(CHUNK):
                    if progress:
                        progress(0, 0)
                    digest.update(chunk)
            if digest.hexdigest() != model["sha256"]:
                target.unlink()
                raise ValueError(f"Download corrompido: {model['filename']}. Tente novamente.")
            receipt.write_text(json.dumps(identity), encoding="utf-8")
        print()
    return notes


def install_extensions(wf: Workflow, comfyui_dir: Path, progress=None) -> list[str]:
    """Instala extensões do catálogo em diretório próprio, com revisão fixa."""
    import tempfile
    import zipfile

    from . import paths
    for node in wf.requires.get('bundled_nodes', []):
        source = paths.PACKAGE_DIR / 'engine_nodes' / node['name']
        target = comfyui_dir / 'custom_nodes' / node['name']
        if source.parent != paths.PACKAGE_DIR / 'engine_nodes' or not source.is_file():
            raise ValueError('Nó embutido desconhecido.')
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
    packages = list(wf.requires.get('engine_packages', []))
    # Bibliotecas de inferência ficam fora de custom_nodes e usam namespace próprio.
    for library in wf.requires.get('libraries', []):
        root = comfyui_dir / 'spe_libraries'
        root.mkdir(parents=True, exist_ok=True)
        target = root / library['name']
        if not target.resolve().is_relative_to(root.resolve()):
            raise ValueError('Pasta de biblioteca inválida.')
        marker = target / '.spe-revision'
        if target.exists():
            if not marker.is_file() or marker.read_text() != library['revision']:
                raise ValueError('Biblioteca existente tem revisão diferente do catálogo.')
            continue
        archive = comfyui_dir / f"spe-{library['name']}-{library['revision']}.zip"
        download(library['url'], archive, progress)
        if hashlib.sha256(archive.read_bytes()).hexdigest() != library['sha256']:
            archive.unlink()
            raise ValueError('Arquivo da biblioteca corrompido.')
        with tempfile.TemporaryDirectory(dir=root, prefix='.spe-install-') as temp:
            staging = Path(temp)
            with zipfile.ZipFile(archive) as z:
                for entry in z.infolist():
                    if not (staging / entry.filename).resolve().is_relative_to(staging.resolve()):
                        raise ValueError('Arquivo da biblioteca contém caminho inválido.')
                z.extractall(staging)
            folders = list(staging.iterdir())
            if len(folders) != 1:
                raise ValueError('Arquivo da biblioteca inválido.')
            prepared = staging / 'prepared'
            prepared.mkdir()
            import shutil
            import re
            for name in library['include']:
                source = folders[0] / name
                if not source.resolve().is_relative_to(folders[0].resolve()):
                    raise ValueError('Caminho de biblioteca inválido.')
                if source.is_dir():
                    shutil.copytree(source, prepared / name)
                else:
                    shutil.copy2(source, prepared / name)
            for file in prepared.rglob('*.py'):
                data = file.read_text(encoding='utf-8')
                data = re.sub(r'(?m)^(from|import) (ldm|models|utils)(?=[. ])',
                              lambda m: f"{m[1]} {library['namespace']}.{m[2]}", data)
                file.write_text(data, encoding='utf-8')
            (prepared / '.spe-revision').write_text(library['revision'])
            prepared.replace(target)
        archive.unlink()
    for extension in wf.requires.get('extensions', []):
        root = comfyui_dir / 'custom_nodes'
        root.mkdir(parents=True, exist_ok=True)
        target = root / extension['name']
        if not target.resolve().is_relative_to(root.resolve()):
            raise ValueError('Pasta de extensão inválida.')
        marker = target / '.spe-revision'
        revision = extension['revision']
        if target.exists():
            if not marker.is_file() or marker.read_text() != revision:
                raise ValueError(f"A extensão {extension['name']} existente tem uma revisão diferente do catálogo.")
        else:
            archive = comfyui_dir / f"{extension['name']}-{revision}.zip"
            download(extension['url'], archive, progress)
            with tempfile.TemporaryDirectory(dir=root, prefix='.spe-install-') as temp:
                staging = Path(temp)
                with zipfile.ZipFile(archive) as z:
                    for entry in z.infolist():
                        if not (staging / entry.filename).resolve().is_relative_to(staging.resolve()):
                            raise ValueError('Arquivo da extensão contém caminho inválido.')
                    z.extractall(staging)
                folders = list(staging.iterdir())
                if len(folders) != 1 or not (folders[0] / '__init__.py').is_file():
                    raise ValueError('Arquivo da extensão inválido.')
                (folders[0] / '.spe-revision').write_text(revision)
                folders[0].replace(target)
            archive.unlink()
        packages.extend(extension['packages'])
    return packages
