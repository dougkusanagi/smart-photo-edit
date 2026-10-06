import asyncio
import json
import zipfile
from unittest.mock import Mock

import pytest

from smart_photo_edit import config, models, workflows, installer
from smart_photo_edit.runtime import ManagedRuntime
from smart_photo_edit.workflows import WorkflowError
from .test_server import edit_form, read_events, make_env


def builtin(wf_id='qwen21-viggle-turbo'):
    return workflows.discover()[0][wf_id]


@pytest.mark.parametrize('wf_id', ['qwen21-viggle-turbo', 'qwen21-base'])
def test_all_catalog_combinations_have_consistent_graph_and_requirements(wf_id):
    original = builtin(wf_id)
    for image_id, image in models.CATALOG['images'].items():
        if wf_id not in image['workflows']:
            continue
        for encoder_id, encoder in models.CATALOG['encoders'].items():
            for device in ['cpu', 'default']:
                wf = models.configure(original, {'image': image_id, 'text_encoder': encoder_id, 'text_device': device, 'memory': 'low'})
                graph = wf.apply(image_name='photo.png', prompt='edit', seed=1)
                for node in graph.values():
                    for value in node['inputs'].values():
                        if isinstance(value, list) and len(value) == 2:
                            assert value[0] in graph
                files = {m['filename'] for m in wf.requires['models']}
                assert image['filename'] in files and encoder['filename'] in files
                assert models.CATALOG['vae']['filename'] in files
                if wf_id == 'qwen21-viggle-turbo':
                    assert ('2' in graph) == (not image['merged'])
                    assert (models.CATALOG['lora']['filename'] in files) == (not image['merged'])
                    assert graph['3']['inputs']['model'] == (['1', 0] if image['merged'] else ['2', 0])
                clip = graph['4' if wf_id.endswith('turbo') else '2']
                assert clip['inputs']['device'] == device
                if encoder.get('format') == 'gguf':
                    assert clip['class_type'] == 'SPEQwen3VLLoader'
                    assert models.CATALOG['vision']['filename'] in files
                    assert wf.requires['bundled_nodes']
                assert bool(wf.requires['extensions']) == (image['format'] == 'gguf' or encoder.get('format') == 'gguf')
    assert original.graph['1']['class_type'] == 'UNETLoader'
    assert original.graph['4' if wf_id.endswith('turbo') else '2']['inputs']['clip_name'].endswith('int8_convrot.safetensors')


def test_incompatible_and_unknown_models_are_rejected():
    for values in [{'image': 'turbo_q4_k_m'}, {'text_encoder': 'qwen2.5'}, {'text_encoder': {}}, {'memory': 'bad'}, {'vae': 'wan'}, {'image': '../model.gguf'}]:
        with pytest.raises(WorkflowError):
            models.choices('qwen21-base', values)


def test_imported_workflow_is_not_rewritten():
    wf = builtin()
    wf.source = 'user'
    assert models.configure(wf, {'image': 'turbo_q4_k_m'}) is wf
    assert models.public(wf) is None


def test_small_encoder_keeps_matching_vision_projector_and_size():
    wf = builtin()
    spec = models.public(wf, {'image':'turbo_q4_k_m', 'text_encoder':'q3_k_m', 'text_device':'cpu'})
    encoder = next(e for e in spec['encoders'] if e['id'] == 'q3_k_m')
    assert encoder['size_bytes'] + encoder['vision_bytes'] < 6_000_000_000
    assert encoder['experimental']
    resolved = models.configure(wf, spec['values'])
    assert spec['total_bytes'] == sum(m['size_bytes'] for m in resolved.requires['models'])
    assert resolved.graph['4']['inputs']['clip_name'].startswith('spe-qwen3vl-8b/')
    assert models.CATALOG['vision']['folder'] == models.CATALOG['encoders']['q3_k_m']['folder']


async def test_selection_is_saved_and_used_by_edit_and_check(make_env, png):
    env = await make_env()
    selected = {'image':'turbo_q4_k_m', 'text_encoder':'q3_k_m', 'text_device':'cpu', 'memory':'low'}
    response = await env.http.put('/api/settings', json={'model_choices': {config.DEFAULT_WORKFLOW: selected}})
    assert response.status == 200
    assert json.loads(env.state.config_path.read_text())['model_choices'][config.DEFAULT_WORKFLOW] == selected
    listing = await (await env.http.get('/api/workflows')).json()
    wf = next(w for w in listing['workflows'] if w['active'])
    assert wf['model_options']['values'] == selected
    assert not any(m['folder'] == 'loras' for m in wf['requires']['models'])
    result = await env.http.post('/api/edit', data=edit_form(png))
    assert (await read_events(result))[-1]['type'] == 'done'
    graph = next(iter(env.comfy.prompts.values()))
    assert graph['1']['class_type'] == 'UnetLoaderGGUF'
    assert graph['4']['class_type'] == 'SPEQwen3VLLoader'
    env.comfy.available = set(n['class_type'] for n in graph.values()) - {'UnetLoaderGGUF'}
    check = await (await env.http.get('/api/workflows/qwen21-viggle-turbo/check')).json()
    assert not check['ok'] and any('UnetLoaderGGUF' in i['message'] for i in check['issues'])


async def test_invalid_selection_is_atomic_and_busy_edit_blocks_changes(make_env):
    env = await make_env()
    before = env.state.settings.to_dict()
    response = await env.http.put('/api/settings', json={'workflow':'qwen21-base','model_choices': {'qwen21-base': {'image':'turbo_q4_k_m'}}})
    assert response.status == 422 and env.state.settings.to_dict() == before
    async with env.state.edit_lock:
        response = await env.http.put('/api/settings', json={'model_choices': {config.DEFAULT_WORKFLOW: {'text_encoder':'w4a8'}}})
        assert response.status == 409
    assert env.state.settings.to_dict() == before


def test_invalid_saved_selection_falls_back_without_disabling_app():
    settings = config.Settings.from_dict({'model_choices': {config.DEFAULT_WORKFLOW: {'text_encoder':'unknown'}, 'qwen21-base': {'text_encoder':'w4a8'}}})
    assert config.DEFAULT_WORKFLOW not in settings.model_choices
    assert settings.model_choices['qwen21-base']['text_encoder'] == 'w4a8'


def test_changing_models_invalidates_running_engine(monkeypatch):
    runtime = ManagedRuntime()
    native = models.configure(builtin())
    compact = models.configure(builtin(), {'image':'turbo_q4_k_m'})
    runtime.workflow_id = native.id
    runtime.workflow_signature = models.signature(native)
    runtime.launcher = Mock(state='running')
    assert models.signature(native) != models.signature(compact)
    # Test through start in an event loop, so no real install occurs.
    async def go():
        gate = asyncio.Event()
        async def prepare(wf):
            assert wf.graph['1']['class_type'] == 'UnetLoaderGGUF'
            gate.set()
        monkeypatch.setattr(runtime, '_prepare', prepare)
        assert runtime.start(native) is None
        await runtime.start(compact)
        assert gate.is_set()
        await runtime.close()
    asyncio.run(go())


def test_extension_install_is_pinned_atomic_and_reusable(monkeypatch, tmp_path):
    wf = models.configure(builtin(), {'image':'turbo_q4_k_m'})
    calls = []
    def archive(url, path, progress):
        calls.append(url)
        with zipfile.ZipFile(path, 'w') as z:
            z.writestr('repo/__init__.py', '# plugin')
    monkeypatch.setattr(installer, 'download', archive)
    assert 'gguf>=0.13.0' in installer.install_extensions(wf, tmp_path)
    installer.install_extensions(wf, tmp_path)
    assert len(calls) == 1 and models.GGUF_COMMIT in calls[0]
    assert (tmp_path / 'custom_nodes/ComfyUI-GGUF/.spe-revision').read_text() == models.GGUF_COMMIT

@pytest.mark.parametrize('device', ['cpu', 'default'])
def test_bundled_multimodal_loader_preserves_vision_and_cpu_device(monkeypatch, device):
    import runpy
    import sys
    from types import ModuleType, SimpleNamespace
    from pathlib import Path
    state = {'model.visual.deepstack_merger_list.0.norm.weight': object()}
    extension = ModuleType('fake_gguf_extension')
    class Loader:
        def load_data(self, paths):
            assert paths == ['/models/encoder.gguf']
            return [state]
    Loader.__module__ = extension.__name__
    extension.GGMLOps = object()
    extension.GGUFModelPatcher = SimpleNamespace(clone=lambda patcher: ('gguf', patcher))
    comfy = ModuleType('comfy')
    sd = ModuleType('comfy.sd')
    sd.CLIPType = SimpleNamespace(QWEN_IMAGE='qwen_image')
    sd.load_text_encoder_state_dicts = Mock(return_value=SimpleNamespace(patcher='original'))
    management = ModuleType('comfy.model_management')
    management.text_encoder_offload_device = lambda: 'offload'
    comfy.sd, comfy.model_management = sd, management
    for name, module in {
        extension.__name__: extension, 'comfy': comfy, 'comfy.sd': sd,
        'comfy.model_management': management,
        'torch': SimpleNamespace(device=lambda value: value),
        'nodes': SimpleNamespace(NODE_CLASS_MAPPINGS={'CLIPLoaderGGUF': Loader}),
        'folder_paths': SimpleNamespace(get_full_path=lambda *_: '/models/encoder.gguf', get_folder_paths=lambda _: ['/embeddings']),
    }.items():
        monkeypatch.setitem(sys.modules, name, module)
    namespace = runpy.run_path(str(Path(models.__file__).parent / 'engine_nodes/spe_qwen_gguf.py'))
    loader = namespace['SPEQwen3VLLoader']()
    clip, = loader.load('encoder.gguf', device)
    assert clip.patcher == ('gguf', 'original')
    options = sd.load_text_encoder_state_dicts.call_args.kwargs['model_options']
    assert options['custom_operations'] is extension.GGMLOps
    assert options['initial_device'] == ('cpu' if device == 'cpu' else 'offload')
    if device == 'cpu':
        assert options['load_device'] == options['offload_device'] == 'cpu'
    state.clear()
    with pytest.raises(ValueError, match='projetor visual'):
        loader.load('encoder.gguf', device)
