#!/usr/bin/env python3
"""Atualiza clones Git ou instalações por ZIP sem tocar nos dados do usuário."""
from __future__ import annotations

import io
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import tempfile
import urllib.request
import zipfile

if __package__:
    from .build_zip import ROOT, package_files
else:
    from build_zip import ROOT, package_files

ARCHIVE_URL = 'https://github.com/dougkusanagi/smart-photo-edit/archive/refs/heads/main.zip'
MAX_ARCHIVE_BYTES = 32 * 1024 * 1024
MAX_EXTRACTED_BYTES = 64 * 1024 * 1024


def apply_archive(data: bytes, root: Path) -> None:
    """Valida e prepara todos os arquivos antes de substituir o código local."""
    with tempfile.TemporaryDirectory(prefix='spe-update-') as directory:
        staging = Path(directory) / 'source'
        backup = Path(directory) / 'backup'
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            names = archive.namelist()
            if not names or sum(info.file_size for info in archive.infolist()) > MAX_EXTRACTED_BYTES:
                raise ValueError('Pacote vazio ou grande demais.')
            for name in names:
                path = PurePosixPath(name)
                if path.is_absolute() or '..' in path.parts or '\\' in name or ':' in name:
                    raise ValueError('Caminho inseguro no pacote de atualização.')
            prefixes = {PurePosixPath(name).parts[0] for name in names}
            if len(prefixes) != 1:
                raise ValueError('Estrutura inesperada no pacote de atualização.')
            archive.extractall(staging)
        source = staging / prefixes.pop()
        files = package_files(source)
        # Não siga links na instalação local nem substitua diretórios por arquivos.
        for file in files:
            relative = file.relative_to(source)
            target = root / relative
            if target.is_symlink() or any((root / parent).is_symlink() for parent in relative.parents):
                raise ValueError(f'Link simbólico no destino: {relative}')
            if target.exists() and not target.is_file():
                raise ValueError(f'Destino não é um arquivo: {relative}')
            if target.exists():
                saved = backup / relative
                saved.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(target, saved)
        changed = []
        try:
            for file in files:
                relative = file.relative_to(source)
                target = root / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as tmp:
                    pending = Path(tmp.name)
                try:
                    shutil.copyfile(file, pending)
                    pending.chmod(0o755 if relative.as_posix() in ('install.sh', 'update.sh', 'run.sh') else 0o644)
                    os.replace(pending, target)
                    changed.append(relative)
                finally:
                    pending.unlink(missing_ok=True)
        except BaseException:
            for relative in reversed(changed):
                saved = backup / relative
                if saved.exists():
                    shutil.copy2(saved, root / relative)
                else:
                    (root / relative).unlink(missing_ok=True)
            raise


def update(root: Path) -> None:
    if (root / '.git').exists():
        dirty = subprocess.check_output(
            ['git', 'status', '--porcelain', '--untracked-files=normal'], cwd=root, text=True)
        if dirty.strip():
            raise ValueError('Há alterações locais. Salve-as em um commit ou stash antes de atualizar.')
        subprocess.run(['git', 'pull', '--ff-only'], cwd=root, check=True)
    else:
        print('Baixando o código público do GitHub…', flush=True)
        request = urllib.request.Request(ARCHIVE_URL, headers={'User-Agent': 'SmartPhotoEdit-Updater'})
        with urllib.request.urlopen(request, timeout=60) as response:
            data = response.read(MAX_ARCHIVE_BYTES + 1)
        if len(data) > MAX_ARCHIVE_BYTES:
            raise ValueError('Download grande demais para um pacote de código.')
        apply_archive(data, root)
    print('Código atualizado. Modelos, fotos e histórico foram preservados.')


if __name__ == '__main__':
    try:
        update(ROOT)
    except (OSError, ValueError, subprocess.CalledProcessError, zipfile.BadZipFile) as exc:
        raise SystemExit(f'Não consegui atualizar: {exc}') from exc
