"""IA de texto opcional: processo privado na CPU, liberado após cada sugestão."""
from __future__ import annotations

import asyncio
import json
import os
import secrets
import sys
from pathlib import Path

from . import paths

MODEL_ID = 'Qwen/Qwen3-0.6B'
MODEL_REVISION = 'c1899de289a04d12100db370d81485cdf75e47ca'


class PromptEnhancer:
    def __init__(self):
        self.task = None
        self.proc = None
        self.job = {}

    @property
    def busy(self):
        return self.task is not None and not self.task.done()

    def start(self, prompt, has_reference, edit_lock):
        if self.busy:
            raise ValueError('O aprimorador já está trabalhando.')
        self.job = {'id': secrets.token_hex(12), 'state': 'running', 'phase': 'Preparando IA de texto na CPU'}
        self.task = asyncio.create_task(self._run(prompt, has_reference, edit_lock))
        return dict(self.job)

    async def _process(self, argv, payload=None):
        env = {**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'HF_HUB_DISABLE_PROGRESS_BARS': '1', 'TOKENIZERS_PARALLELISM': 'false', 'PYTHONIOENCODING': 'utf-8'}
        log = paths.user_data_dir() / 'prompt-ai.log'
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open('ab') as stderr:
            self.proc = await asyncio.create_subprocess_exec(*map(str, argv), stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE if payload else stderr, stderr=stderr, env=env)
            try:
                output, _ = await self.proc.communicate(json.dumps(payload, ensure_ascii=False).encode() if payload else None)
                if self.proc.returncode:
                    raise RuntimeError(f'Falha ao executar a IA local. Consulte {log}.')
                return output
            finally:
                if self.proc and self.proc.returncode is None:
                    self.proc.terminate()
                    try:
                        await asyncio.wait_for(self.proc.wait(), 5)
                    except asyncio.TimeoutError:
                        self.proc.kill()
                        await self.proc.wait()
                self.proc = None

    async def _run(self, prompt, has_reference, edit_lock):
        try:
            async with edit_lock:
                root = paths.user_data_dir() / 'prompt-ai'
                python = root / 'venv' / ('Scripts/python.exe' if sys.platform == 'win32' else 'bin/python')
                marker = root / 'ready'
                if not marker.exists():
                    root.mkdir(parents=True, exist_ok=True)
                    if not python.exists():
                        await self._process([sys.executable, '-m', 'venv', root / 'venv'])
                    self.job['phase'] = 'Instalando dependências opcionais da IA de texto'
                    await self._process([python, '-m', 'pip', 'install', 'torch>=2.6,<3', '--index-url', 'https://download.pytorch.org/whl/cpu'])
                    await self._process([python, '-m', 'pip', 'install', 'transformers>=4.51,<5', 'safetensors'])
                    marker.write_text('qwen3-cpu-v1', encoding='utf-8')
                self.job['phase'] = 'Carregando IA na CPU e aprimorando o texto (primeiro uso baixa cerca de 1,2 GB)'
                worker = paths.PACKAGE_DIR / 'prompt_worker.py'
                output = await self._process([python, worker], {'prompt': prompt, 'has_reference': has_reference, 'model': MODEL_ID, 'revision': MODEL_REVISION, 'cache': str(root / 'models')})
                result = json.loads(output)
                if result.get('error'):
                    raise ValueError(result['error'])
                if not isinstance(result.get('prompt'), str) or not result['prompt'].strip():
                    raise ValueError('A IA não retornou uma sugestão de prompt.')
                self.job.update(state='done', phase='Sugestão pronta', prompt=result['prompt'].strip(), model=MODEL_ID)
        except asyncio.CancelledError:
            self.job.update(state='cancelled', phase='Aprimoramento cancelado')
        except Exception as exc:
            self.job.update(state='error', phase=str(exc), error=str(exc))

    async def cancel(self):
        if self.busy:
            self.task.cancel()
            await asyncio.gather(self.task, return_exceptions=True)

    async def close(self):
        await self.cancel()
