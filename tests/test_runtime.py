import asyncio
import hashlib
import zipfile
from unittest.mock import Mock

import pytest
from aiohttp.test_utils import TestClient, TestServer

from smart_photo_edit import config, installer, workflows, models
from smart_photo_edit.comfy import ComfyError
from smart_photo_edit.runtime import ManagedRuntime
from smart_photo_edit.server import AppState, create_app
from smart_photo_edit.service import ResultStore
from .fake_comfy import FakeComfy
from .test_server import edit_form, read_events


def workflow():
    return workflows.discover()[0][config.DEFAULT_WORKFLOW]


async def test_first_edit_prepares_private_engine(monkeypatch, tmp_path, png):
    fake = TestServer(FakeComfy().app)
    await fake.start_server()
    calls = []

    async def prepare(runtime, wf):
        calls.append(wf.id)
        runtime.url = str(fake.make_url('')).rstrip('/')
        runtime.workflow_id = wf.id
        runtime.workflow_signature = models.signature(wf)
        runtime.launcher = Mock(state='running')

    monkeypatch.setattr(ManagedRuntime, '_prepare', prepare)
    state = AppState(config.Settings(), ResultStore(tmp_path / 'results'))
    client = TestClient(TestServer(create_app(state)))
    await client.start_server()
    try:
        status = await (await client.get('/api/status')).json()
        assert status['comfy']['managed'] and not status['comfy']['online']
        assert calls == []  # abrir o app não baixa gigabytes
        for _ in range(2):
            result = await client.post('/api/edit', data=edit_form(png))
            assert (await read_events(result))[-1]['type'] == 'done'
        assert calls == [config.DEFAULT_WORKFLOW]
    finally:
        await client.close()
        await fake.close()
    state.runtime.launcher.stop.assert_called_once()


async def test_prepare_single_task_and_retry(monkeypatch):
    runtime = ManagedRuntime()
    gate = asyncio.Event()

    async def prepare(wf):
        await gate.wait()
        raise ComfyError('Falha de download', 'engine_setup_failed')

    monkeypatch.setattr(runtime, '_prepare', prepare)
    first = runtime.start(workflow())
    assert runtime.start(workflow()) is first
    gate.set()
    with pytest.raises(ComfyError):
        await first
    second = runtime.start(workflow())
    assert second is not first
    await asyncio.gather(second, return_exceptions=True)
    await runtime.close()


async def test_cancel_edit_during_preparation_keeps_task_for_next_edit(monkeypatch):
    runtime = ManagedRuntime()
    gate = asyncio.Event()

    async def prepare(wf):
        await gate.wait()

    monkeypatch.setattr(runtime, '_prepare', prepare)
    cancel = asyncio.Event()
    cancel.set()
    assert not await runtime.ensure(workflow(), None, cancel, Mock())
    assert not runtime.task.done()
    gate.set()
    await runtime.close()


def test_source_archive_cannot_escape_engine(monkeypatch):
    runtime = ManagedRuntime()

    def archive(url, dest, progress):
        with zipfile.ZipFile(dest, 'w') as z:
            z.writestr('../../escape.py', 'bad')

    monkeypatch.setattr('smart_photo_edit.runtime.download', archive)
    with pytest.raises(ValueError, match='caminho inválido'):
        runtime._source()
    assert not (runtime.root.parent / 'escape.py').exists()


def test_bad_model_checksum_can_be_retried(monkeypatch, tmp_path):
    wf = workflow()
    wf.requires = {'custom_nodes': [], 'models': [{'folder': 'vae', 'filename': 'test.bin', 'url': 'https://test/model', 'sha256': hashlib.sha256(b'good').hexdigest()}]}
    def bad_download(url, target, progress):
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b'bad')
    monkeypatch.setattr(installer, 'download', bad_download)
    with pytest.raises(ValueError, match='corrompido'):
        installer.install_requirements(wf, tmp_path)
    assert not (tmp_path / 'models/vae/test.bin').exists()
    def good_download(url, target, progress):
        target.write_bytes(b'good')
    monkeypatch.setattr(installer, 'download', good_download)
    installer.install_requirements(wf, tmp_path)
    monkeypatch.setattr(installer, 'download', Mock(side_effect=AssertionError('não deve baixar novamente')))
    installer.install_requirements(wf, tmp_path)


def test_imported_requirements_cannot_escape_data_dir(tmp_path):
    wf = workflow()
    wf.requires = {'custom_nodes': [{'name': '../../../outside.py', 'url': 'https://test/node'}], 'models': []}
    with pytest.raises(ValueError, match='fora da pasta'):
        installer.install_requirements(wf, tmp_path)


async def test_preparation_installs_in_private_environment_and_launches_argv(monkeypatch):
    runtime = ManagedRuntime()
    wf = workflow()
    runtime.root.mkdir(parents=True)
    runtime.source.mkdir()
    (runtime.source / 'main.py').write_text('')
    runtime.python.parent.mkdir(parents=True)
    runtime.python.write_text('')
    (runtime.root / 'environment.ready').write_text('ready')
    pip_calls = []
    async def pip(*args):
        pip_calls.append(args)
    monkeypatch.setattr(runtime, '_pip', pip)
    install = Mock(return_value=[])
    monkeypatch.setattr('smart_photo_edit.runtime.install_requirements', install)
    monkeypatch.setattr(runtime.launcher, 'start_argv', Mock())
    await runtime.start(wf)
    assert (runtime.root / 'dependencies.ready').exists()
    assert pip_calls[-1] == ('-r', str(runtime.source / 'requirements.txt'))
    install.assert_called_once()
    assert install.call_args.args[:3] == (wf, runtime.source, None) and install.call_args.args[4] == runtime._next_model
    argv, cwd = runtime.launcher.start_argv.call_args.args
    assert argv[:2] == [str(runtime.python), str(runtime.source / 'main.py')]
    assert argv[argv.index('--listen') + 1] == '127.0.0.1'
    assert runtime.url.endswith(':' + argv[argv.index('--port') + 1])
    assert '--disable-api-nodes' in argv and cwd == str(runtime.source)
    await runtime.close()


async def test_pip_failure_does_not_mark_engine_ready(monkeypatch):
    runtime = ManagedRuntime()
    runtime.source.mkdir(parents=True)
    (runtime.source / 'main.py').write_text('')
    runtime.python.parent.mkdir(parents=True)
    runtime.python.write_text('')
    (runtime.root / 'environment.ready').write_text('ready')
    async def fail(*args):
        raise RuntimeError('Falha pip')
    monkeypatch.setattr(runtime, '_pip', fail)
    with pytest.raises(ComfyError, match='Falha pip'):
        await runtime.start(workflow())
    assert not (runtime.root / 'dependencies.ready').exists()
    assert runtime.error == 'Falha pip'
    await runtime.close()


async def test_managed_mode_ignores_old_external_configuration(tmp_path):
    state = AppState(config.Settings(comfy_url='http://some-external-server:8188', comfy_command='invalid-command', comfy_autostart=True), ResultStore(tmp_path / 'r'))
    client = TestClient(TestServer(create_app(state)))
    await client.start_server()
    try:
        data = await (await client.get('/api/status')).json()
        assert data['comfy']['managed'] and data['comfy']['url'] == ''
        assert state.launcher.proc is None and state.runtime.task is None
        response = await client.get('/assets/tailwind.css')
        assert response.status == 200 and '.btn-primary' in await response.text()
        html = await (await client.get('/')).text()
        assert 'cdn.tailwindcss.com' not in html and '@apply' not in html
    finally:
        await client.close()


def test_inference_library_is_pinned_namespaced_and_reused(monkeypatch, tmp_path):
    import io
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, 'w') as z:
        z.writestr('repo/models/unet.py', 'from ldm.util import fn\nfrom .ops import x\n')
        z.writestr('repo/ldm/util.py', 'def fn(): pass\n')
        z.writestr('repo/LICENSE', 'License text')
        z.writestr('repo/train.py', 'training code')
    content = archive.getvalue()
    wf = workflow()
    wf.requires = {'libraries': [{'name': 'sinsr', 'revision': 'pinned', 'namespace': 'spe_sinsr',
                    'url': 'https://test/library.zip', 'sha256': hashlib.sha256(content).hexdigest(),
                    'include': ['models', 'ldm', 'LICENSE']}], 'engine_packages': ['timm==1.0.30']}
    download = Mock(side_effect=lambda url, dest, progress: dest.write_bytes(content))
    monkeypatch.setattr(installer, 'download', download)
    assert installer.install_extensions(wf, tmp_path) == ['timm==1.0.30']
    root = tmp_path / 'spe_libraries/sinsr'
    assert 'from spe_sinsr.ldm.util' in (root / 'models/unet.py').read_text()
    assert 'from .ops import x' in (root / 'models/unet.py').read_text()
    assert (root / 'LICENSE').read_text() == 'License text'
    assert not (root / 'train.py').exists()
    installer.install_extensions(wf, tmp_path)
    assert download.call_count == 1


@pytest.mark.parametrize('bad_hash,escape', [(True, False), (False, True)])
def test_inference_library_rejects_corrupt_and_unsafe_archive(monkeypatch, tmp_path, bad_hash, escape):
    import io
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, 'w') as z:
        z.writestr('../../outside.py' if escape else 'repo/LICENSE', 'content')
    content = archive.getvalue()
    wf = workflow()
    wf.requires = {'libraries': [{'name': 'sinsr', 'revision': 'pinned', 'namespace': 'spe_sinsr',
                   'url': 'https://test/library.zip',
                   'sha256': 'wrong' if bad_hash else hashlib.sha256(content).hexdigest(),
                   'include': ['LICENSE']}]}
    monkeypatch.setattr(installer, 'download', lambda url, dest, progress: dest.write_bytes(content))
    with pytest.raises(ValueError, match='corrompido|caminho inválido'):
        installer.install_extensions(wf, tmp_path)
    assert not (tmp_path / 'spe_libraries/sinsr').exists()
