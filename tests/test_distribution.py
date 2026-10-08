"""Empacotamento e atualização sem modelos, rede ou dados pessoais."""
import io
import os
from pathlib import Path
import subprocess
import zipfile

import pytest

from scripts.build_zip import ROOT, build_zip
from scripts.update_app import apply_archive, update


def archive_bytes(tmp_path):
    archive = tmp_path / 'app.zip'
    build_zip(ROOT, archive)
    return archive.read_bytes()


def test_zip_update_preserves_data_and_installs_launchers(tmp_path):
    root = tmp_path / 'installation'
    root.mkdir()
    for name in ('.venv/marker', 'results/photo.png', 'config.json', 'models/weights.bin'):
        file = root / name
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_bytes(b'personal')
    (root / 'run.sh').write_text('old')
    apply_archive(archive_bytes(tmp_path), root)
    for name in ('.venv/marker', 'results/photo.png', 'config.json', 'models/weights.bin'):
        assert (root / name).read_bytes() == b'personal'
    for name in ('install.sh', 'update.sh', 'run.sh', 'install.bat', 'update.bat', 'run.bat'):
        assert (root / name).read_bytes() == (ROOT / name).read_bytes()
    if os.name != 'nt':
        assert (root / 'run.sh').stat().st_mode & 0o111


def test_zip_update_rejects_traversal_before_writing(tmp_path):
    data = io.BytesIO()
    with zipfile.ZipFile(data, 'w') as archive:
        archive.writestr('app/../../outside.txt', 'bad')
    with pytest.raises(ValueError, match='inseguro'):
        apply_archive(data.getvalue(), tmp_path)
    assert not (tmp_path.parent / 'outside.txt').exists()


@pytest.mark.skipif(os.name == 'nt', reason='Links no Windows dependem de privilégios locais')
def test_zip_update_rejects_local_symlink(tmp_path):
    root = tmp_path / 'installation'
    root.mkdir()
    external = tmp_path / 'external'
    external.mkdir()
    (root / 'scripts').symlink_to(external, target_is_directory=True)
    with pytest.raises(ValueError, match='simbólico'):
        apply_archive(archive_bytes(tmp_path), root)
    assert list(external.iterdir()) == []
    assert not (root / 'README.md').exists()


def test_zip_update_rolls_back_after_write_failure(tmp_path, monkeypatch):
    from scripts import update_app
    root = tmp_path / 'installation'
    root.mkdir()
    (root / 'README.md').write_text('previous README')
    replace = update_app.os.replace
    calls = 0
    data = archive_bytes(tmp_path)

    def fail_second(source, target):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError('simulated write failure')
        return replace(source, target)

    monkeypatch.setattr(update_app.os, 'replace', fail_second)
    with pytest.raises(OSError, match='simulated'):
        apply_archive(data, root)
    assert (root / 'README.md').read_text() == 'previous README'
    assert not (root / 'install.bat').exists()


def test_git_update_refuses_local_changes(tmp_path):
    subprocess.run(['git', 'init', str(tmp_path)], check=True, capture_output=True)
    (tmp_path / 'local.txt').write_text('local work')
    with pytest.raises(ValueError, match='alterações locais'):
        update(tmp_path)
    assert (tmp_path / 'local.txt').read_text() == 'local work'
