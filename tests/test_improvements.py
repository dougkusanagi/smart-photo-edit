import asyncio
import json
from unittest.mock import AsyncMock

import pytest
from aiohttp import FormData

from smart_photo_edit import workflows, models
from smart_photo_edit.prompt_ai import PromptEnhancer
from .test_server import make_env, read_events


@pytest.mark.parametrize('wf_id', ['qwen21-base', 'qwen21-viggle-turbo'])
async def test_reference_reaches_graph_and_effective_prompt_metadata(make_env, png, wf_id):
    env = await make_env()
    env.state.settings.workflow = wf_id
    form = FormData()
    form.add_field('image', png, filename='pessoa.png', content_type='image/png')
    form.add_field('reference', png, filename='referencia.png', content_type='image/png')
    form.add_field('options', json.dumps({'prompt':'Coloque em <image1> o boné branco de <image2>.'}), content_type='application/json')
    events = await read_events(await env.http.post('/api/edit', data=form))
    assert events[-1]['type'] == 'done'
    result = events[-1]['images'][0]
    assert isinstance(result['duration_seconds'], (int, float)) and result['duration_seconds'] >= 0
    saved = await (await env.http.get(result['url'])).read()
    from smart_photo_edit.history import image_metadata
    assert image_metadata(saved)['duration_seconds'] == result['duration_seconds']
    assert result['user_prompt'] == 'Coloque em <image1> o boné branco de <image2>.'
    assert result['reference']['tag'] == '<image2>'
    assert result['reference']['filename'] == 'referencia.png'
    assert len(env.comfy.uploads) == 2
    graph = next(iter(env.comfy.prompts.values()))
    encoder = next(node for node in graph.values() if 'images.image_1' in node['inputs'])
    assert encoder['inputs']['images.image_2'] == ['_spe_reference', 0]
    assert graph['_spe_reference']['inputs']['image'].endswith('spe_reference.png [temp]')
    assert encoder['inputs']['prompt'] == result['prompt']
    assert result['prompt'] == result['user_prompt']


def test_reference_not_added_without_request_and_imported_workflows_are_rejected():
    wf = workflows.discover()[0]['qwen21-viggle-turbo']
    chosen = models.configure(wf, {'image':'turbo_q4_k_m'})
    graph = chosen.apply(image_name='source.png', prompt='edit')
    assert '_spe_reference' not in graph
    assert graph['3']['inputs']['model'] == ['1',0]
    wf.source = 'user'
    with pytest.raises(workflows.WorkflowError, match='segunda referência'):
        wf.apply(image_name='source.png', prompt='edit', reference_name='reference.png')


async def test_prompt_job_review_errors_cancel_and_busy_guard(make_env, monkeypatch):
    env = await make_env()
    enhancer = env.state.prompt_ai
    async def fake_run(prompt, role, edit_lock):
        async with edit_lock:
            await asyncio.sleep(.01)
            enhancer.job.update(state='done', phase='Sugestão pronta', prompt='Preserve a identidade; integre luz e sombras.')
    monkeypatch.setattr(enhancer, '_run', fake_run)
    response = await env.http.post('/api/prompt/enhance', json={'prompt':'Pôr do sol', 'has_reference':True})
    assert response.status == 202
    job = await response.json()
    while enhancer.busy:
        await asyncio.sleep(.01)
    ready = await (await env.http.get('/api/prompt/jobs/' + job['id'])).json()
    assert ready['state'] == 'done' and ready['prompt'].startswith('Preserve')
    assert (await env.http.get('/api/prompt/jobs/unknown')).status == 404
    assert (await env.http.post('/api/prompt/enhance', json={'prompt':''})).status == 400
    async with env.state.edit_lock:
        assert (await env.http.post('/api/prompt/enhance', json={'prompt':'edit'})).status == 409
    async def waiting(*args):
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            enhancer.job.update(state='cancelled')
    monkeypatch.setattr(enhancer, '_run', waiting)
    job = await (await env.http.post('/api/prompt/enhance', json={'prompt':'edit'})).json()
    await asyncio.sleep(0)
    assert (await env.http.post('/api/prompt/enhance', json={'prompt':'other'})).status == 409
    assert (await (await env.http.delete('/api/prompt/jobs/' + job['id'])).json())['state'] == 'cancelled'
    assert not enhancer.busy


@pytest.mark.parametrize('result, expected', [({'prompt':'Integre luz e sombras.'}, 'done'), ({'error':'Memória insuficiente'}, 'error'), ({'prompt':''}, 'error')])
async def test_private_prompt_worker_result_and_cpu_install(tmp_path, monkeypatch, result, expected):
    from smart_photo_edit import paths
    monkeypatch.setattr(paths, 'user_data_dir', lambda: tmp_path)
    enhancer = PromptEnhancer()
    calls = []
    async def process(argv, payload=None):
        calls.append((list(map(str, argv)), payload))
        return json.dumps(result).encode() if payload else None
    monkeypatch.setattr(enhancer, '_process', process)
    enhancer.start('Troque o fundo.', True, asyncio.Lock())
    await enhancer.task
    assert enhancer.job['state'] == expected
    assert any('https://download.pytorch.org/whl/cpu' in argv for argv, _ in calls)
    assert calls[-1][1]['has_reference'] is True
    assert (tmp_path / 'prompt-ai' / 'ready').exists()
    assert not enhancer.busy


async def test_cancelling_prompt_worker_reaps_process(tmp_path, monkeypatch):
    import sys
    from smart_photo_edit import paths
    monkeypatch.setattr(paths, 'user_data_dir', lambda: tmp_path)
    enhancer = PromptEnhancer()
    task = asyncio.create_task(enhancer._process([sys.executable, '-c', 'import time; time.sleep(60)']))
    while enhancer.proc is None:
        await asyncio.sleep(.01)
    proc = enhancer.proc
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert proc.returncode is not None
    assert enhancer.proc is None
