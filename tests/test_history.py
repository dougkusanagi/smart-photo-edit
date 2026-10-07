import io
import json
import os
import time

import pytest
from PIL import Image
from PIL.PngImagePlugin import PngInfo

from smart_photo_edit.history import ResultStore
from .test_server import make_env, edit_form, read_events


def test_png_export_preserves_prompt_unicode_pixels_and_recovers_index(tmp_path):
    image = Image.new('RGB', (60, 40), (34, 56, 78))
    info = PngInfo()
    info.add_text('prompt', '{"engine": "graph"}')
    buf = io.BytesIO()
    image.save(buf, 'PNG', pnginfo=info)
    store = ResultStore(tmp_path)
    prompt = 'Iluminação suave\nPreserve o rosto e a cor café ☕ <script>'
    rid = store.save(buf.getvalue(), {'prompt': prompt, 'seed': 17, 'workflow': {'id':'qwen21-base'}})
    with Image.open(store.path(rid)) as exported:
        assert exported.tobytes() == image.tobytes()
        assert exported.info['prompt'] == exported.info['Description'] == prompt
        assert exported.info['comfyui_prompt'] == '{"engine": "graph"}'
        assert json.loads(exported.info['SmartPhotoEdit'])['seed'] == 17
    (tmp_path / (rid + '.json')).write_text('invalid json')
    record = store.record(rid)
    assert record['prompt'] == prompt and record['seed'] == 17
    assert ResultStore(tmp_path).list()['items'][0]['id'] == rid


def test_history_persists_old_edits_and_paginates_newest_first(tmp_path, png):
    store = ResultStore(tmp_path)
    older = store.save(png, {'prompt':'primeira'})
    old = time.time() - 7 * 24 * 3600
    os.utime(store.path(older), (old, old))
    latest = store.save(png, {'prompt':'segunda'})
    reopened = ResultStore(tmp_path)
    assert reopened.path(older) is not None
    first = reopened.list(limit=1)
    assert first['items'][0]['id'] == latest and first['has_more']
    assert reopened.list(first['next_offset'], 1)['items'][0]['id'] == older
    thumb = reopened.thumbnail(latest)
    assert thumb.is_file()
    assert len(reopened.list()['items']) == 2
    assert reopened.delete(latest)
    assert not thumb.exists() and not (tmp_path / (latest + '.json')).exists()
    assert reopened.path(older)
    assert not reopened.delete('../config.json')
    assert reopened.record(older + '.json') is None


@pytest.mark.parametrize('format', ['JPEG', 'WEBP'])
def test_non_png_engine_output_is_exported_as_png_with_metadata(tmp_path, format):
    buf = io.BytesIO()
    Image.new('RGB', (30, 20), 'red').save(buf, format)
    store = ResultStore(tmp_path)
    rid = store.save(buf.getvalue(), {'prompt':'Teste'})
    assert rid.endswith('.png')
    with Image.open(store.path(rid)) as image:
        assert image.info['prompt'] == 'Teste'


async def test_history_api_and_each_variation_metadata(make_env, png):
    env = await make_env()
    prompt = 'Troque o céu por um pôr do sol <b>azul</b>'
    events = await read_events(await env.http.post('/api/edit', data=edit_form(png, prompt=prompt, seed=333, n=2, name='Praia', source_result='anterior.png')))
    results = events[-1]['images']
    assert [item['seed'] for item in results] == [333,334]
    for item in results:
        assert item['prompt'] == prompt
        with Image.open(io.BytesIO(await (await env.http.get(item['url'])).read())) as image:
            metadata = json.loads(image.info['SmartPhotoEdit'])
            assert metadata['prompt'] == prompt and metadata['seed'] == item['seed']
            assert metadata['workflow']['id'] == 'qwen21-viggle-turbo'
            assert metadata['params']['resolution'] == 1024
            assert metadata['name'] == 'Praia'
            assert metadata['source_result'] == 'anterior.png'
            assert metadata['model_choices']['image'] == 'image_int8'
            assert metadata['model_choices']['text_device'] == 'default'
            assert metadata['variation'] == item['variation'] and metadata['variations'] == 2
            assert {model['role'] for model in metadata['models']} == {'image', 'text_encoder', 'vae', 'turbo_lora'}
            assert all(len(model['sha256']) == 64 and model['size_bytes'] > 0 for model in metadata['models'])
        # Também recupera os detalhes pelo PNG se o índice JSON se perder.
        (env.state.store.folder / (item['id'] + '.json')).unlink()
        recovered = env.state.store.record(item['id'])
        assert recovered['models'] == item['models'] and recovered['model_choices'] == item['model_choices']
        assert recovered['duration_seconds'] == item['duration_seconds']
        assert (await (await env.http.get(recovered['original_url'])).read()) == png
    history = await (await env.http.get('/api/history?limit=1')).json()
    assert history['has_more'] and history['items'][0]['id'] == results[1]['id']
    thumb = await env.http.get(history['items'][0]['thumbnail_url'])
    assert thumb.status == 200 and thumb.content_type == 'image/webp'
    assert (await env.http.get('/api/history?offset=invalid')).status == 400
    assert (await env.http.delete('/api/history/' + results[0]['id'])).status == 200
    assert (await env.http.get(results[0]['url'])).status == 404
    assert len((await (await env.http.get('/api/history')).json())['items']) == 1

async def test_export_can_be_reopened_with_its_prompt_and_legacy_images_are_listed(make_env, png):
    env = await make_env()
    rid = env.state.store.save(png, {'prompt':'Reabrir: luz de manhã', 'seed':999})
    exported = await (await env.http.get('/api/results/' + rid)).read()
    response = await env.http.post('/api/image-metadata', data=exported, headers={'Content-Type':'image/png'})
    metadata = (await response.json())['metadata']
    assert metadata['prompt'] == 'Reabrir: luz de manhã' and metadata['seed'] == 999
    assert 'url' not in metadata and 'id' not in metadata
    assert (await (await env.http.post('/api/image-metadata', data=png)).json())['metadata'] is None
    assert (await env.http.post('/api/image-metadata', data=b'invalid')).status == 415
    legacy = 'a' * 24 + '.png'
    (env.state.store.folder / legacy).write_bytes(png)
    items = (await (await env.http.get('/api/history')).json())['items']
    old = next(item for item in items if item['id'] == legacy)
    assert old['prompt'] == '' and old['width'] == 64

async def test_long_prompt_reaches_workflow_and_is_preserved_in_export_and_history(make_env, png):
    env = await make_env()
    prompt = ('Refaça iluminação, perspectiva e sombras de contato. ' * 70) + 'Preserve a identidade da pessoa.'
    assert len(prompt) > 2000
    events = await read_events(await env.http.post('/api/edit', data=edit_form(png, prompt=prompt)))
    assert events[-1]['type'] == 'done'
    image = events[-1]['images'][0]
    graph = next(iter(env.comfy.prompts.values()))
    assert graph['7']['inputs']['prompt'] == prompt
    assert image['prompt'] == prompt
    with Image.open(io.BytesIO(await (await env.http.get(image['url'])).read())) as exported:
        assert exported.info['prompt'] == prompt
    items = (await (await env.http.get('/api/history')).json())['items']
    assert items[0]['prompt'] == prompt
