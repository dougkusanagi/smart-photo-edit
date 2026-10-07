"""Motor privado: instala, inicia e encerra sem depender de outro ComfyUI."""
from __future__ import annotations

import asyncio
import os
import shutil
import socket
import sys
import threading
import venv
import zipfile
from pathlib import Path

from . import paths
from .comfy import ComfyClient, ComfyError
from .installer import download, install_requirements, install_extensions
from .models import missing_bytes, signature
from .launcher import ComfyLauncher
from .workflows import Workflow

COMFY_VERSION = 'v0.37.4'
COMFY_ARCHIVE = f'https://github.com/Comfy-Org/ComfyUI/archive/refs/tags/{COMFY_VERSION}.zip'


class ManagedRuntime:
    def __init__(self):
        self.root = paths.user_data_dir() / 'engine'
        self.source = self.root / f'ComfyUI-{COMFY_VERSION}'
        self.python = self.root / 'venv' / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
        self.launcher = ComfyLauncher()
        self.url = ''
        self.phase = 'Motor local será preparado na primeira edição.'
        self.error = ''
        self.task: asyncio.Task | None = None
        self.install_proc: asyncio.subprocess.Process | None = None
        self.stopping = threading.Event()
        self.workflow_id = ''
        self.workflow_signature = ''
        self.download: dict | None = None  # {'done', 'total'} em bytes enquanto baixa modelos
        self._file = {'base': 0, 'size': 0}

    @property
    def state(self):
        if self.task and not self.task.done():
            return 'preparing'
        return self.launcher.state

    def _progress(self, done, total):
        if self.stopping.is_set():
            raise RuntimeError('Preparo interrompido.')
        pct = f' · {done * 100 // total}%' if total else ''
        self.phase = 'Baixando componentes do motor local' + pct

    def _source(self):
        self.root.mkdir(parents=True, exist_ok=True)
        if (self.source / 'main.py').exists():
            return
        archive = self.root / 'comfyui.zip'
        download(COMFY_ARCHIVE, archive, self._progress)
        staging = self.root / 'extract'
        shutil.rmtree(staging, ignore_errors=True)
        staging.mkdir()
        with zipfile.ZipFile(archive) as z:
            for entry in z.infolist():
                target = (staging / entry.filename).resolve()
                if not target.is_relative_to(staging.resolve()):
                    raise ValueError('Arquivo do motor contém caminho inválido.')
            z.extractall(staging)
        extracted = staging / f'ComfyUI-{COMFY_VERSION[1:]}'
        if not (extracted / 'main.py').is_file():
            raise ValueError('Arquivo do motor não contém main.py.')
        extracted.replace(self.source)
        shutil.rmtree(staging)
        archive.unlink()

    async def _pip(self, *args: str):
        if self.stopping.is_set():
            raise RuntimeError('Preparo interrompido.')
        with paths.log_file().open('ab') as log:
            self.install_proc = await asyncio.create_subprocess_exec(
                str(self.python), '-m', 'pip', 'install', *args,
                stdout=log, stderr=asyncio.subprocess.STDOUT,
            )
            code = await self.install_proc.wait()
        self.install_proc = None
        if code:
            raise RuntimeError(f'Falha ao instalar dependências. Consulte {paths.log_file()}.')

    async def _prepare(self, wf: Workflow):
        self.error = ''
        try:
            self.phase = 'Preparando motor local'
            await asyncio.to_thread(self._source)
            if self.stopping.is_set():
                raise RuntimeError("Preparo interrompido.")
            marker = self.root / 'dependencies.ready'
            if not self.python.exists() or not (self.root / 'environment.ready').exists():
                self.phase = 'Criando ambiente privado do motor'
                await asyncio.to_thread(venv.EnvBuilder(with_pip=True).create, self.root / 'venv')
                (self.root / 'environment.ready').write_text('ready', encoding='utf-8')
            if self.stopping.is_set():
                raise RuntimeError("Preparo interrompido.")
            if not marker.exists():
                self.phase = 'Instalando dependências do motor (pode levar vários minutos)'
                if sys.platform in ('linux', 'win32'):
                    await self._pip('torch', 'torchvision', 'torchaudio', '--index-url', 'https://download.pytorch.org/whl/cu130')
                await self._pip('-r', str(self.source / 'requirements.txt'))
                marker.write_text(COMFY_VERSION, encoding='utf-8')
            self.phase = 'Preparando extensões compatíveis'
            packages = await asyncio.to_thread(install_extensions, wf, self.source, self._progress)
            extensions_marker = self.root / 'extensions.ready'
            expected = repr(packages)
            if packages and (not extensions_marker.exists() or extensions_marker.read_text() != expected):
                await self._pip(*packages)
                extensions_marker.write_text(expected)
            self.phase = 'Preparando os modelos selecionados'
            notes = await asyncio.to_thread(install_requirements, wf, self.source, None, self._models_progress(wf), self._next_model)
            self.download = None
            missing = [n for n in notes if n.startswith('Baixe manualmente:')]
            if missing:
                raise RuntimeError('Este workflow precisa de arquivos sem download automático: ' + '; '.join(missing))
            if self.stopping.is_set():
                raise RuntimeError("Preparo interrompido.")
            # Reiniciar carrega novos nós de workflows importados.
            if self.launcher.state == 'running':
                await asyncio.to_thread(self.launcher.stop)
            with socket.socket() as sock:
                sock.bind(('127.0.0.1', 0))
                port = sock.getsockname()[1]
            self.url = f'http://127.0.0.1:{port}'
            self.phase = 'Iniciando motor local'
            self.launcher.start_argv([
                str(self.python), str(self.source / 'main.py'), '--listen', '127.0.0.1',
                '--port', str(port), '--disable-auto-launch', '--disable-api-nodes', *wf.runtime_args,
            ], str(self.source))
            self.workflow_id = wf.id
            self.workflow_signature = signature(wf)
        except Exception as exc:
            self.download = None
            self.error = str(exc)
            self.phase = self.error
            raise ComfyError(f'Não consegui preparar o motor local: {exc}', 'engine_setup_failed') from exc

    def _models_progress(self, wf: Workflow):
        """Progresso somado de todos os modelos que faltam, em vez de um percentual por arquivo."""
        models_dir = self.source / 'models'
        pending = [m for m in wf.requires.get('models', []) if not (models_dir / m['folder'] / m['filename']).exists()]
        total = sum(m.get('size_bytes', 0) for m in pending)
        self._file = {'base': 0, 'size': 0}
        if total:
            self.download = {'done': total - missing_bytes(pending, models_dir), 'total': total}

        def progress(done, size):
            if self.stopping.is_set():
                raise RuntimeError('Preparo interrompido.')
            if not size or not self.download:
                self.phase = 'Verificando os modelos baixados'
                return
            self.download = {'done': min(total, self._file['base'] + done), 'total': total}
            gb = lambda n: f'{n / 1e9:.1f}'.replace('.', ',')
            self.phase = f'Baixando modelos · {gb(self.download["done"])} de {gb(total)} GB'
        return progress

    def _next_model(self, model: dict):
        self._file['base'] += self._file['size']
        self._file['size'] = model.get('size_bytes', 0)

    async def cancel(self):
        """Interrompe o preparo; os arquivos .part permitem retomar o download depois."""
        if not (self.task and not self.task.done()):
            return
        self.stopping.set()
        if self.install_proc and self.install_proc.returncode is None:
            self.install_proc.terminate()
        await asyncio.gather(self.task, return_exceptions=True)
        self.stopping.clear()
        self.error = ''
        self.phase = 'Preparo cancelado. O download continua de onde parou.'

    def start(self, wf: Workflow):
        if self.task and not self.task.done():
            return self.task
        if self.launcher.state == 'running' and self.workflow_signature == signature(wf):
            return self.task
        self.task = asyncio.create_task(self._prepare(wf))
        # A rota de preparo responde imediatamente; consome falhas para evitar task exception warnings.
        self.task.add_done_callback(lambda t: t.exception() if not t.cancelled() else None)
        return self.task

    async def ensure(self, wf: Workflow, session, cancel, emit):
        task = self.start(wf)
        while task and not task.done():
            if cancel.is_set():
                return False
            await emit({'type': 'progress', 'p': 0, 'phase': self.phase})
            await asyncio.sleep(.5)
        if task:
            await task
        if self.workflow_signature != signature(wf):
            return await self.ensure(wf, session, cancel, emit)
        client = ComfyClient(self.url, session)
        for _ in range(240):
            if cancel.is_set():
                return False
            if await client.system_stats() is not None:
                self.phase = 'Motor local pronto'
                return True
            if self.launcher.state != 'running':
                self.error = self.phase = f'O motor local encerrou. Consulte {paths.log_file()}.'
                raise ComfyError(self.error, 'launch_failed')
            await emit({'type': 'progress', 'p': 0, 'phase': self.phase})
            await asyncio.sleep(.5)
        await asyncio.to_thread(self.launcher.stop)
        self.error = self.phase = f'O motor demorou para iniciar. Consulte {paths.log_file()}.'
        raise ComfyError(self.error, 'launch_failed')

    async def ensure_addons(self, wf: Workflow, cancel, emit):
        """Baixa as LoRAs adicionais da edição (sem reiniciar o motor). False se cancelado."""
        if not any(m.get('addon') for m in wf.requires.get('models', [])):
            return True
        task = asyncio.create_task(asyncio.to_thread(install_requirements, wf, self.source, None, self._progress))
        try:
            while not task.done():
                if cancel.is_set():
                    self.stopping.set()  # interrompe o download; o .part permite retomar
                    await asyncio.gather(task, return_exceptions=True)
                    self.stopping.clear()
                    return False
                await emit({'type': 'progress', 'p': 0, 'phase': self.phase.replace('componentes do motor local', 'LoRA adicional')})
                await asyncio.sleep(.5)
            notes = await task
        except Exception as exc:
            raise ComfyError(f'Não consegui preparar a LoRA adicional: {exc}', 'addon_setup_failed') from exc
        if any(n.startswith('Baixe manualmente:') for n in notes):
            raise ComfyError('A LoRA adicional não tem download automático.', 'addon_setup_failed')
        return True

    async def close(self):
        self.stopping.set()
        if self.install_proc and self.install_proc.returncode is None:
            self.install_proc.terminate()
            try:
                await asyncio.wait_for(self.install_proc.wait(), 10)
            except asyncio.TimeoutError:
                self.install_proc.kill()
                await self.install_proc.wait()
        if self.task and not self.task.done():
            # Espera o download parar antes de encerrar; nunca deixa um instalador órfão.
            await asyncio.gather(self.task, return_exceptions=True)
        await asyncio.to_thread(self.launcher.stop)
