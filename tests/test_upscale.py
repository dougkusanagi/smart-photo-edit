import asyncio
import io
import runpy
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from PIL import Image

from smart_photo_edit import paths, upscale
from smart_photo_edit.history import image_metadata
from smart_photo_edit.workflows import WorkflowError
from .conftest import make_png
from .test_server import edit_form, make_env, read_events


def encoded(image, **kwargs):
    buf = io.BytesIO()
    image.save(buf, 'PNG', **kwargs)
    return buf.getvalue()


@pytest.mark.parametrize('value', [True, False, 2.5, 2.0, 3, '2.0', None, {}, []])
def test_rejects_invalid_scale(value):
    with pytest.raises(WorkflowError):
        upscale.validate_scale(value)


def test_finish_preserves_transparency_profile_and_orientation():
    source = Image.new('RGBA', (3, 2), (20, 40, 60, 128))
    exif = source.getexif()
    exif[274] = 6
    original = encoded(source, exif=exif, icc_profile=b'test-profile')
    assert upscale.dimensions(original, 2) == (2, 3)
    result = Image.open(io.BytesIO(upscale.finish(original, make_png(size=(4, 6)), 2)))
    assert result.size == (4, 6)
    assert result.getchannel('A').getextrema() == (128, 128)
    assert result.info['icc_profile'] == b'test-profile'
    assert result.getexif().get(274, 1) == 1
    with pytest.raises(ValueError, match='resolução'):
        upscale.finish(original, make_png(size=(6, 4)), 2)


@pytest.mark.parametrize('scale', [2, 4])
async def test_upscale_api_history_and_export(make_env, monkeypatch, scale):
    env = await make_env()
    env.state.settings.engine_mode = 'managed'
    env.state.runtime.url = env.state.settings.comfy_url
    ensure = AsyncMock(return_value=True)
    monkeypatch.setattr(env.state.runtime, 'ensure', ensure)
    original = encoded(Image.new('RGBA', (37, 29), (20, 40, 60, 128)))
    events = await read_events(await env.http.post('/api/edit', data=edit_form(
        original, mode='upscale', scale=scale, prompt='ignored', seed='invalid',
        addons=['light-blend'], n=4, strength=1, negative='ignored')))
    assert events[-1]['type'] == 'done', events
    item = events[-1]['images'][0]
    graph = next(iter(env.comfy.prompts.values()))
    assert graph['2']['inputs']['scale_by'] == scale
    assert {n['class_type'] for n in graph.values()} == {
        'LoadImage', 'ImageScaleBy', 'SeedVR2Preprocess', 'VAELoader', 'UNETLoader', 'VAEEncodeTiled',
        'SeedVR2Conditioning', 'KSampler', 'VAEDecodeTiled', 'SeedVR2PostProcessing', 'PreviewImage', 'PrimitiveString'}
    prepared_wf = ensure.call_args.args[0]
    assert prepared_wf.requires['models'] == upscale.OPTIONS['seedvr2']['models']
    assert 'bundled_nodes' not in prepared_wf.requires
    assert '--force-fp32' not in prepared_wf.runtime_args
    assert '--cache-none' in prepared_wf.runtime_args
    assert len(env.comfy.prompts) == 1
    assert item['mode'] == 'upscale' and item['params'] == {'scale': scale}
    assert item['seed'] == 0 and item['addons'] == []
    assert item['prompt'] == '' and item['negative'] == ''
    assert item['upscale']['algorithm'] == 'seedvr2' and item['upscale']['steps'] == 1
    assert item['upscale']['color_correction'] == 'lab'
    assert graph['8']['inputs']['steps'] == 1 and graph['8']['inputs']['denoise'] == 1
    assert any('detalhes · passo 3/3' in e.get('phase', '') for e in events)
    exported_bytes = await (await env.http.get(item['url'])).read()
    exported = Image.open(io.BytesIO(exported_bytes))
    assert exported.size == (37 * scale, 29 * scale)
    assert exported.getchannel('A').getextrema() == (128, 128)
    assert image_metadata(exported_bytes)['upscale']['precision'] == 'UNet FP8 (e4m3fn) / VAE FP16'
    history = (await (await env.http.get('/api/history')).json())['items'][0]
    assert history['models'][0]['sha256'] == upscale.OPTIONS['seedvr2']['models'][0]['sha256']
    assert await (await env.http.get(history['original_url'])).read() == original


async def test_invalid_requests_fail_before_preparing_engine(make_env, monkeypatch, png):
    env = await make_env()
    response = await env.http.post('/api/edit', data=edit_form(png, mode='upscale'))
    assert response.status == 400 and (await response.json())['code'] == 'upscale_unsupported'
    env.state.settings.engine_mode = 'managed'
    ensure = AsyncMock(return_value=True)
    monkeypatch.setattr(env.state.runtime, 'ensure', ensure)
    broken_crc = bytearray(png)
    broken_crc[-13] ^= 1  # CRC do bloco IDAT: Pillow.verify lança SyntaxError.
    for image, scale in [(png, 3), (b'\x89PNG\r\n\x1a\ninvalid', 2),
                         (bytes(broken_crc), 2),
                         (make_png(size=(2001, 1000)), 4),
                         (make_png(size=(1700, 1200)), 2)]:  # 8,16 MP: acima do limite do SeedVR2
        response = await env.http.post('/api/edit', data=edit_form(image, mode='upscale', scale=scale))
        assert response.status == 400
    ensure.assert_not_called()
    assert not env.comfy.prompts


async def test_upscale_cancel_interrupts_engine_without_saving(make_env, monkeypatch, png):
    env = await make_env(duration=5)
    env.state.settings.engine_mode = 'managed'
    env.state.runtime.url = env.state.settings.comfy_url
    monkeypatch.setattr(env.state.runtime, 'ensure', AsyncMock(return_value=True))
    response = await env.http.post('/api/edit', data=edit_form(png, mode='upscale'))
    async for line in response.content:
        if b'Reconstruindo' in line:
            break
    response.close()
    for _ in range(40):
        if env.comfy.interrupted:
            break
        await asyncio.sleep(.1)
    assert env.comfy.interrupted
    assert env.state.store.list()['items'] == []


@pytest.mark.parametrize('seed', [True, -1, 2**53, 'x'])
async def test_invalid_diffusion_options_fail_before_prepare(make_env, monkeypatch, png, seed):
    env = await make_env()
    env.state.settings.engine_mode = 'managed'
    ensure = AsyncMock(return_value=True)
    monkeypatch.setattr(env.state.runtime, 'ensure', ensure)
    response = await env.http.post('/api/edit', data=edit_form(png, mode='upscale', upscale_seed=seed))
    assert response.status == 400
    ensure.assert_not_called()


async def test_seed_is_recorded(make_env, monkeypatch, png):
    env = await make_env()
    env.state.settings.engine_mode = 'managed'
    env.state.runtime.url = env.state.settings.comfy_url
    monkeypatch.setattr(env.state.runtime, 'ensure', AsyncMock(return_value=True))
    events = await read_events(await env.http.post('/api/edit', data=edit_form(png, mode='upscale', upscale_seed=42)))
    assert events[-1]['type'] == 'done', events
    assert events[-1]['images'][0]['seed'] == 42
    assert next(iter(env.comfy.prompts.values()))['8']['inputs']['seed'] == 42


@pytest.fixture
def upscale_node(monkeypatch):
    torch = pytest.importorskip('torch', reason='Execute também com o Python do motor para testar os blocos.')
    comfy = ModuleType('comfy')
    management = ModuleType('comfy.model_management')
    management.unload_all_models = Mock()
    management.soft_empty_cache = Mock()
    management.throw_exception_if_processing_interrupted = Mock()
    def raise_non_oom(exc):
        if not isinstance(exc, torch.OutOfMemoryError):
            raise exc
    management.raise_non_oom = raise_non_oom
    utils = ModuleType('comfy.utils')
    utils.ProgressBar = lambda steps: SimpleNamespace(update_absolute=Mock())
    sample = ModuleType('comfy.sample')
    comfy.model_management, comfy.utils, comfy.sample = management, utils, sample
    node_module = ModuleType('nodes')
    for name, module in {'comfy': comfy, 'comfy.model_management': management, 'comfy.utils': utils,
                         'comfy.sample': sample, 'nodes': node_module}.items():
        monkeypatch.setitem(sys.modules, name, module)
    return {'torch': torch, 'mm': management}


@pytest.mark.parametrize('model', ['sinsr', 'adcsr'])
@pytest.mark.parametrize('scale', [2, 4])
async def test_selected_alternative_has_own_graph_and_metadata(make_env, monkeypatch, png, model, scale):
    env = await make_env()
    env.state.settings.engine_mode = 'managed'
    env.state.runtime.url = env.state.settings.comfy_url
    ensure = AsyncMock(return_value=True)
    monkeypatch.setattr(env.state.runtime, 'ensure', ensure)
    events = await read_events(await env.http.post('/api/edit', data=edit_form(
        png, mode='upscale', upscale_model=model, scale=scale, upscale_seed=42)))
    assert events[-1]['type'] == 'done', events
    item = events[-1]['images'][0]
    wf = ensure.call_args.args[0]
    assert wf.id == model + '-upscale'
    assert wf.requires['models'] == upscale.OPTIONS[model]['models']
    assert wf.requires['bundled_nodes'] == [{'name': 'spe_upscale_options.py'}]
    graph = next(iter(env.comfy.prompts.values()))
    assert graph['5']['class_type'] == 'SPEOneStepUpscale'
    assert graph['5']['inputs']['algorithm'] == model
    assert graph['5']['inputs']['seed'] == 42
    assert item['params'] == {'scale': scale}
    assert item['prompt'] == upscale.prompt(model)
    assert item['negative'] == ''
    assert item['upscale']['algorithm'] == model and item['upscale']['steps'] == 1
    assert item['upscale']['node_cache'] == 'disabled'
    assert item['upscale']['cpu_threads_limit'] == 4
    if model == 'sinsr':
        assert item['upscale']['attention_max_score_bytes'] == 32*1024**2
        assert item['upscale']['vae_attention'] == 'query_chunks'
        assert item['upscale']['vae_device'] == 'auto'
        assert item['upscale']['vae_precision_policy'] == 'fp16_cuda_with_fp32_fallback'
        assert '--cpu-vae' not in wf.runtime_args
        assert '--force-fp32' not in wf.runtime_args
    if model == 'adcsr':
        assert item['upscale']['color_alignment'] == 'adain_global' and item['upscale']['deterministic']
        assert item['upscale']['text_encoder'] == 'none'
        assert item['upscale']['tile_size'] == 192 and item['upscale']['noise'] == 'none'
    assert item['upscale']['downsample_from_4x'] == (scale == 2)
    assert 'denoise' not in item['upscale']
    exported = await (await env.http.get(item['url'])).read()
    assert image_metadata(exported)['upscale'] == item['upscale']
    assert await (await env.http.get(item['original_url'])).read() == png
    phases = [event.get('phase', '') for event in events if event['type'] == 'progress']
    assert 'Reconstruindo · bloco 1/1' in phases
    assert 'Finalizando · bloco 1/1' in phases
    assert 'Montando ampliação · 1/1 blocos' in phases


@pytest.mark.parametrize('model', ['unknown', [], 'sd15_tile', 'pisa'])
async def test_bad_model_options_do_not_prepare_engine(make_env, monkeypatch, png, model):
    env = await make_env()
    env.state.settings.engine_mode = 'managed'
    ensure = AsyncMock(return_value=True)
    monkeypatch.setattr(env.state.runtime, 'ensure', ensure)
    response = await env.http.post('/api/edit', data=edit_form(png, mode='upscale', upscale_model=model))
    assert response.status == 400
    ensure.assert_not_called()


async def test_size_limit_depends_on_the_model(make_env, monkeypatch):
    env = await make_env()
    env.state.settings.engine_mode = 'managed'
    env.state.runtime.url = env.state.settings.comfy_url
    monkeypatch.setattr(env.state.runtime, 'ensure', AsyncMock(return_value=True))
    big = make_png(size=(1700, 1200))  # 2×: 8,16 MP
    assert (await env.http.post('/api/edit', data=edit_form(big, mode='upscale', upscale_model='seedvr2'))).status == 400
    events = await read_events(await env.http.post('/api/edit', data=edit_form(big, mode='upscale', upscale_model='adcsr')))
    assert events[-1]['type'] == 'done', events


async def test_public_upscale_catalog_sizes_licenses_and_downloads(make_env, tmp_path):
    env = await make_env()
    env.state.settings.engine_mode = 'managed'
    env.state.runtime.source = tmp_path
    data = await (await env.http.get('/api/upscale/models')).json()
    assert data['default'] == 'seedvr2'
    models = {m['id']: m for m in data['models']}
    assert set(models) == {'seedvr2', 'adcsr', 'sinsr'}
    assert models['sinsr']['size_bytes'] == 699275569
    assert 'NC' in models['sinsr']['license']
    assert models['adcsr']['size_bytes'] == 2163260139
    assert models['seedvr2']['size_bytes'] == 3894119046 and models['seedvr2']['max_output_pixels'] == 8_000_000
    assert not any(m['downloaded'] for m in models.values())
    for m in upscale.OPTIONS['sinsr']['models']:
        target = tmp_path / 'models' / m['folder'] / m['filename']
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('wb') as file:
            file.truncate(m['size_bytes'])
    data = await (await env.http.get('/api/upscale/models')).json()
    assert next(m for m in data['models'] if m['id'] == 'sinsr')['downloaded']
    env.state.settings.engine_mode = 'external'
    data = await (await env.http.get('/api/upscale/models')).json()
    assert not any(m['downloaded'] for m in data['models'])


@pytest.fixture
def alternatives_node(upscale_node, monkeypatch):
    monkeypatch.setitem(sys.modules, 'folder_paths', ModuleType('folder_paths'))
    namespace = runpy.run_path(str(paths.PACKAGE_DIR / 'engine_nodes/spe_upscale_options.py'))
    return namespace


@pytest.mark.parametrize('scale', [2, 4])
@pytest.mark.parametrize('shape', [(1, 117, 155, 3), (1, 1, 2, 3)])
def test_sinsr_tiles_handle_odd_small_images_and_native_scale(alternatives_node, scale, shape):
    t = alternatives_node['torch']
    image = t.ones(shape) * .4
    def process(patch, noise, index, count):
        assert patch.shape[-2] % 64 == 0 and patch.shape[-1] % 64 == 0
        assert noise.shape == patch.shape
        return t.nn.functional.interpolate(patch, scale_factor=4, mode='nearest')
    output = alternatives_node['sinsr_tiles'](image, scale, 42, process, lambda: None)
    assert output.shape == (shape[0], shape[1]*scale, shape[2]*scale, 3)
    t.testing.assert_close(output, t.full_like(output, .4), atol=1e-6, rtol=1e-6)


def test_sinsr_cancel_during_final_block_does_not_return(alternatives_node):
    t = alternatives_node['torch']
    check = Mock(side_effect=[None, RuntimeError('cancelled')])
    process = Mock(side_effect=lambda patch, *args: t.nn.functional.interpolate(patch, scale_factor=4))
    with pytest.raises(RuntimeError, match='cancelled'):
        alternatives_node['sinsr_tiles'](t.ones(1, 2, 3, 3), 4, 42, process, check)
    assert process.call_count == 1


def test_sinsr_attention_matches_dense_and_limits_query_buffers(alternatives_node, monkeypatch):
    t = alternatives_node['torch']
    t.manual_seed(8)
    channels = 8
    block = SimpleNamespace(norm=t.nn.GroupNorm(2, channels),
        q=t.nn.Conv2d(channels, channels, 1), k=t.nn.Conv2d(channels, channels, 1),
        v=t.nn.Conv2d(channels, channels, 1), proj_out=t.nn.Conv2d(channels, channels, 1))
    image = t.randn(2, channels, 17, 19)
    with t.inference_mode():
        h = block.norm(image)
        q, k, v = (layer(h).flatten(2) for layer in (block.q, block.k, block.v))
        scores = t.bmm(q.transpose(1, 2), k) * channels**-.5
        expected = image + block.proj_out(t.bmm(v, scores.softmax(-1).transpose(1, 2)).reshape(image.shape))
        matmul = Mock(wraps=t.bmm)
        monkeypatch.setattr(t, 'bmm', matmul)
        result = alternatives_node['sinsr_attention'](block, image)
    t.testing.assert_close(result, expected, rtol=1e-5, atol=1e-6)
    scores = matmul.call_args_list[::2]
    assert len(scores) == 2 and len(matmul.call_args_list) == 4
    for call in scores:
        query, key = call.args
        assert query.shape[-2] <= 256
        assert key.shape[-1] == 17*19
        assert query.shape[0] * query.shape[-2] * key.shape[-1] * 4 <= 32*1024**2


def test_sinsr_attention_can_cancel_between_query_chunks(alternatives_node):
    t = alternatives_node['torch']
    identity = t.nn.Identity()
    block = SimpleNamespace(norm=identity, q=identity, k=identity, v=identity, proj_out=identity)
    mm = alternatives_node['sinsr_attention'].__globals__['mm']
    mm.throw_exception_if_processing_interrupted.side_effect = [None, RuntimeError('cancelled')]
    with t.inference_mode(), pytest.raises(RuntimeError, match='cancelled'):
        alternatives_node['sinsr_attention'](block, t.randn(1, 8, 17, 19))


def test_sinsr_quantization_matches_all_codebook_distances(alternatives_node):
    t = alternatives_node['torch']
    t.manual_seed(12)
    block = SimpleNamespace(embedding=t.nn.Embedding(8192, 3))
    image = t.randn(2, 3, 33, 33)
    flat = image.movedim(1, -1).reshape(-1, 3)
    with t.inference_mode():
        distance = flat.square().sum(1, keepdim=True) + block.embedding.weight.square().sum(1) - 2*(flat @ block.embedding.weight.T)
        expected = distance.argmin(1)
        del distance
        mm = alternatives_node['sinsr_quantize'].__globals__['mm']
        mm.throw_exception_if_processing_interrupted.reset_mock()
        quantized, _, info = alternatives_node['sinsr_quantize'](block, image)
        expected_quantized = block.embedding(expected).reshape(2, 33, 33, 3).movedim(-1, 1)
    t.testing.assert_close(info[2], expected)
    t.testing.assert_close(quantized, expected_quantized)
    assert mm.throw_exception_if_processing_interrupted.call_count == 3


def test_sinsr_gpu_oom_reduces_tiles_then_moves_vae_to_cpu(alternatives_node, monkeypatch):
    t = alternatives_node['torch']
    node = alternatives_node['SPEOneStepUpscale']()
    namespace = node.upscale.__globals__
    namespace['mm'].get_torch_device = Mock(return_value=t.device('cuda'))
    namespace['mm'].free_memory = Mock()
    model, vae = Mock(), Mock()
    model.parameters.return_value = [t.empty(1)]
    monkeypatch.setitem(namespace, 'load_sinsr', lambda: {'model': model, 'vae': vae})
    tiles = Mock(side_effect=[t.OutOfMemoryError('oom'), t.OutOfMemoryError('oom'), 'result'])
    monkeypatch.setitem(namespace, 'sinsr_tiles', tiles)
    assert node.upscale(t.ones(1, 64, 64, 3), 'sinsr', 2) == ('result',)
    assert [call.args[-1] for call in tiles.call_args_list] == [128, 64, 64]
    assert [(call.kwargs['device'].type, call.kwargs['dtype']) for call in vae.to.call_args_list] == [('cuda', t.float16), ('cpu', t.float32)]
    vae.cpu.assert_called_once()


def test_sinsr_nonfinite_half_output_retries_with_float32(alternatives_node, monkeypatch):
    t = alternatives_node['torch']
    node = alternatives_node['SPEOneStepUpscale']()
    namespace = node.upscale.__globals__
    namespace['mm'].get_torch_device = Mock(return_value=t.device('cuda'))
    namespace['mm'].free_memory = Mock()
    model, vae = Mock(), Mock()
    model.parameters.return_value = [t.empty(1)]
    monkeypatch.setitem(namespace, 'load_sinsr', lambda: {'model': model, 'vae': vae})
    tiles = Mock(side_effect=[alternatives_node['NonFiniteUpscale']('nan'), 'result'])
    monkeypatch.setitem(namespace, 'sinsr_tiles', tiles)
    assert node.upscale(t.ones(1, 64, 64, 3), 'sinsr', 2) == ('result',)
    vae.float.assert_called_once()


def test_alternative_oom_retries_smaller_blocks_and_offloads_on_failure(alternatives_node, monkeypatch):
    t = alternatives_node['torch']
    node = alternatives_node['SPEOneStepUpscale']()
    namespace = node.upscale.__globals__
    model = Mock()
    model.parameters.return_value = [t.empty(1)]
    monkeypatch.setitem(namespace, 'load_sinsr', Mock(side_effect=lambda: {'model': model, 'vae': Mock()}))
    mm = namespace['mm']
    mm.get_torch_device = Mock(return_value=t.device('cpu'))
    mm.free_memory = Mock()
    calls = Mock(side_effect=[t.OutOfMemoryError('oom'), 'result'])
    monkeypatch.setitem(namespace, 'sinsr_tiles', calls)
    assert node.upscale(t.ones(1, 64, 64, 3), 'sinsr', 4) == ('result',)
    assert [c.args[-1] for c in calls.call_args_list] == [128, 64]
    assert model.cpu.call_count == 1
    calls.side_effect = RuntimeError('cancelled')
    with pytest.raises(RuntimeError, match='cancelled'):
        node.upscale(t.ones(1, 64, 64, 3), 'sinsr', 4)
    assert model.cpu.call_count == 2
    assert mm.unload_all_models.call_count == 4


def test_alternative_restores_cpu_threads_when_loading_fails(alternatives_node, monkeypatch):
    t = alternatives_node['torch']
    node = alternatives_node['SPEOneStepUpscale']()
    namespace = node.upscale.__globals__
    namespace['mm'].get_torch_device = Mock(return_value=t.device('cpu'))
    monkeypatch.setitem(namespace, 'load_sinsr', Mock(side_effect=RuntimeError('load failed')))
    monkeypatch.setattr(t, 'get_num_threads', Mock(return_value=12))
    threads = Mock()
    monkeypatch.setattr(t, 'set_num_threads', threads)
    with pytest.raises(RuntimeError, match='load failed'):
        node.upscale(t.ones(1, 64, 64, 3), 'sinsr', 2)
    assert [c.args[0] for c in threads.call_args_list] == [4, 12]


def test_alternative_releases_local_models_before_collecting(alternatives_node, monkeypatch):
    import weakref
    t = alternatives_node['torch']
    node = alternatives_node['SPEOneStepUpscale']()
    namespace = node.upscale.__globals__
    namespace['mm'].get_torch_device = Mock(return_value=t.device('cpu'))
    namespace['mm'].free_memory = Mock()
    refs = []
    def load():
        model, vae = t.nn.Linear(2, 2), t.nn.Linear(2, 2)
        refs.extend((weakref.ref(model), weakref.ref(vae)))
        return {'model': model, 'vae': vae}
    monkeypatch.setitem(namespace, 'load_sinsr', load)
    monkeypatch.setitem(namespace, 'sinsr_tiles', Mock(return_value='result'))
    collected = []
    monkeypatch.setattr(namespace['gc'], 'collect', lambda: collected.append([ref() is None for ref in refs]))
    assert node.upscale(t.ones(1, 64, 64, 3), 'sinsr', 2) == ('result',)
    assert collected == [[True, True]]


def test_sinsr_reports_encode_inference_decode_progress(alternatives_node, monkeypatch):
    t = alternatives_node['torch']
    node = alternatives_node['SPEOneStepUpscale']()
    namespace = node.upscale.__globals__
    namespace['mm'].get_torch_device = Mock(return_value=t.device('cpu'))
    namespace['mm'].free_memory = Mock()
    class Model(t.nn.Module):
        def forward(self, x, timestep, lq):
            return x
    vae = SimpleNamespace(encode=lambda x: t.nn.functional.avg_pool2d(x, 4),
                          decode=lambda x: t.nn.functional.interpolate(x, scale_factor=4),
                          to=Mock(), cpu=Mock())
    monkeypatch.setitem(namespace, 'load_sinsr', lambda: {'model': Model(), 'vae': vae})
    progress = Mock()
    factory = Mock(return_value=progress)
    monkeypatch.setattr(namespace['comfy'].utils, 'ProgressBar', factory)
    result = node.upscale(t.ones(1, 64, 64, 3), 'sinsr', 2)[0]
    assert result.shape == (1, 128, 128, 3)
    factory.assert_called_once_with(3)
    assert [call.args for call in progress.update_absolute.call_args_list] == [(0, 3), (1, 3), (2, 3), (3, 3)]


def test_safe_checkpoint_reads_tensors_without_importing_the_authors_packages(alternatives_node, tmp_path):
    t = alternatives_node['torch']
    module = ModuleType('pytorch_lightning_fake')
    class Callback:
        pass
    Callback.__module__, Callback.__qualname__ = 'pytorch_lightning_fake', 'Callback'
    module.Callback = Callback
    sys.modules['pytorch_lightning_fake'] = module
    try:
        t.save({'state_dict': {'decoder.w': t.arange(3.)}, 'callbacks': {'x': Callback()}}, tmp_path / 'a.ckpt')
    finally:
        del sys.modules['pytorch_lightning_fake']
    loaded = alternatives_node['safe_checkpoint'](tmp_path / 'a.ckpt')
    assert loaded['state_dict']['decoder.w'].tolist() == [0., 1., 2.]
    assert 'pytorch_lightning_fake' not in sys.modules


def test_adcsr_loads_tensor_checkpoint_and_rejects_executable_pickles(alternatives_node, tmp_path):
    t = alternatives_node['torch']
    import os
    class Evil:
        def __reduce__(self):
            return (os.system, ('echo pwned > ' + str(tmp_path / 'pwned'),))
    t.save({'state_dict': {}, 'x': Evil()}, tmp_path / 'evil.ckpt')
    with pytest.raises(ValueError, match='não permitidos'):
        alternatives_node['safe_checkpoint'](tmp_path / 'evil.ckpt')
    assert not (tmp_path / 'pwned').exists()


def test_adcsr_runs_blocks_at_native_scale_then_aligns_colors_globally(alternatives_node, monkeypatch):
    t = alternatives_node['torch']
    node = alternatives_node['SPEOneStepUpscale']()
    namespace = node.upscale.__globals__
    namespace['mm'].get_torch_device = Mock(return_value=t.device('cpu'))
    namespace['mm'].free_memory = Mock()
    class Model(t.nn.Module):
        def forward(self, x):
            return t.nn.functional.interpolate(x * 0.5, scale_factor=4)  # cor deslocada: o AdaIN global corrige
    monkeypatch.setitem(namespace, 'load_adcsr', lambda: {'model': Model()})
    t.manual_seed(1)
    image = t.rand(1, 70, 90, 3)
    result = node.upscale(image, 'adcsr', 4)[0]
    assert result.shape == (1, 280, 360, 3)
    assert t.isfinite(result).all() and result.min() >= 0 and result.max() <= 1
    t.testing.assert_close(result.mean((1, 2)), image.mean((1, 2)), atol=1e-3, rtol=0)
    assert node.upscale(image, 'adcsr', 2)[0].shape == (1, 140, 180, 3)


def test_adcsr_oom_retries_with_smaller_blocks_and_offloads(alternatives_node, monkeypatch):
    t = alternatives_node['torch']
    node = alternatives_node['SPEOneStepUpscale']()
    namespace = node.upscale.__globals__
    namespace['mm'].get_torch_device = Mock(return_value=t.device('cpu'))
    namespace['mm'].free_memory = Mock()
    model = t.nn.Linear(2, 2)
    monkeypatch.setitem(namespace, 'load_adcsr', lambda: {'model': model})
    tiles = Mock(side_effect=[t.OutOfMemoryError('oom'), t.OutOfMemoryError('oom'), t.zeros(1, 8, 8, 3)])
    monkeypatch.setitem(namespace, 'sinsr_tiles', tiles)
    node.upscale(t.rand(1, 64, 64, 3), 'adcsr', 2)
    assert [call.args[-1] for call in tiles.call_args_list] == [192, 128, 64]
    tiles.side_effect = t.OutOfMemoryError('oom')
    with pytest.raises(t.OutOfMemoryError):
        node.upscale(t.rand(1, 64, 64, 3), 'adcsr', 2)
