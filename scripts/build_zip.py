#!/usr/bin/env python3
"""Gera o pacote de instalação usando apenas a biblioteca padrão do Python."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
PACKAGE_NAME = 'smart-photo-edit'
ROOT_FILES = ('README.md', 'pyproject.toml', 'run.sh', 'run.bat', 'install-windows.txt',
              'install.sh', 'install.bat', 'update.sh', 'update.bat',
              'scripts/build_zip.py', 'scripts/update_app.py')
PATTERNS = ('smart_photo_edit/*.py', 'smart_photo_edit/builtin_workflows/*.json',
            'smart_photo_edit/model_data/*.json', 'smart_photo_edit/engine_nodes/*.py',
            'smart_photo_edit/engine_nodes/*LICENSE.txt')
WEB_EXTENSIONS = {'.html', '.css', '.js', '.json', '.txt', '.svg', '.png', '.jpg', '.jpeg',
                  '.webp', '.ico', '.woff', '.woff2', '.ttf'}


def package_files(root: Path) -> list[Path]:
    files = {root / name for name in ROOT_FILES}
    for pattern in PATTERNS:
        files.update(root.glob(pattern))
    web = root / 'smart_photo_edit' / 'web'
    files.update(p for p in web.rglob('*') if p.suffix.lower() in WEB_EXTENSIONS
                 and not any(part.startswith('.') for part in p.relative_to(web).parts))
    for required in (web / 'index.html', web / 'tailwind.css'):
        if required not in files:
            raise ValueError(f'Arquivo obrigatório ausente: {required}')
    for path in files:
        if not path.is_file() or path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
            raise ValueError(f'Arquivo ausente ou caminho inseguro: {path}')
    return sorted(files, key=lambda p: p.relative_to(root).as_posix())


def build_zip(root: Path, output: Path) -> tuple[int, str]:
    files = package_files(root)
    if output.resolve() in {p.resolve() for p in files}:
        raise ValueError('O destino não pode substituir um arquivo do pacote.')
    output.parent.mkdir(parents=True, exist_ok=True)
    # Arquivo temporário no mesmo volume: um erro não deixa um ZIP final incompleto.
    with tempfile.NamedTemporaryFile(dir=output.parent, suffix='.zip.part', delete=False) as tmp:
        temporary = Path(tmp.name)
    try:
        with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
            for path in files:
                relative = path.relative_to(root).as_posix()
                info = zipfile.ZipInfo(f'{PACKAGE_NAME}/{relative}', date_time=(2020, 1, 1, 0, 0, 0))
                info.create_system = 3
                info.external_attr = (0o100755 if relative in ('run.sh', 'install.sh', 'update.sh') else 0o100644) << 16
                info.compress_type = zipfile.ZIP_DEFLATED
                archive.writestr(info, path.read_bytes(), compresslevel=9)
        with zipfile.ZipFile(temporary) as archive:
            bad = archive.testzip()
            if bad:
                raise ValueError(f'Falha ao verificar o ZIP: {bad}')
        digest = hashlib.sha256(temporary.read_bytes()).hexdigest()
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    return len(files), digest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'dist' / 'smart-photo-edit.zip',
                        help='Destino do ZIP (padrão: dist/smart-photo-edit.zip).')
    args = parser.parse_args()
    try:
        count, digest = build_zip(ROOT, args.output.expanduser().resolve())
    except (OSError, ValueError) as exc:
        parser.exit(1, f'Não consegui gerar o pacote: {exc}\n')
    print(f'ZIP: {args.output.expanduser().resolve()}')
    print(f'{count} arquivos · {args.output.expanduser().stat().st_size:,} bytes')
    print(f'SHA-256: {digest}')


if __name__ == '__main__':
    main()
