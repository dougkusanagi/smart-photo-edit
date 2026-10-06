"""Catálogo fechado de pesos compatíveis com os workflows Qwen-Image 2.1."""
from __future__ import annotations

import copy
import hashlib
import json
from . import paths
from .workflows import Workflow, WorkflowError

CATALOG = json.loads((paths.PACKAGE_DIR / 'model_data/qwen21.json').read_text(encoding='utf-8'))
SUPPORTED = {'qwen21-base', 'qwen21-viggle-turbo'}
DEFAULTS = {'image': 'image_int8', 'text_encoder': 'int8', 'text_device': 'default', 'memory': 'auto'}
GGUF_COMMIT = '373048b8403a7820620065210a691263d4da0a61'
GGUF_EXTENSION = {
    'name': 'ComfyUI-GGUF', 'revision': GGUF_COMMIT,
    'url': f'https://api.github.com/repos/leejet/ComfyUI-GGUF/zipball/{GGUF_COMMIT}',
    'packages': ['gguf>=0.13.0', 'protobuf'],
}


def supported(wf: Workflow) -> bool:
    return wf.source == 'builtin' and wf.id in SUPPORTED


def choices(wf_id: str, values: dict | None = None) -> dict:
    if wf_id not in SUPPORTED:
        raise WorkflowError('Este workflow não permite trocar modelos pelo catálogo do app.')
    if values is not None and not isinstance(values, dict):
        raise WorkflowError('Configuração de modelos inválida.')
    unknown = set(values or {}) - DEFAULTS.keys()
    if unknown:
        raise WorkflowError('Opção de modelo desconhecida: ' + ', '.join(sorted(unknown)))
    out = {**DEFAULTS, **(values or {})}
    if not isinstance(out['image'], str) or out['image'] not in CATALOG['images'] or wf_id not in CATALOG['images'][out['image']]['workflows']:
        raise WorkflowError('Modelo de imagem incompatível com este workflow.')
    if not isinstance(out['text_encoder'], str) or out['text_encoder'] not in CATALOG['encoders']:
        raise WorkflowError('Use um codificador Qwen3-VL 8B do catálogo Qwen-Image 2.1.')
    if out['text_device'] not in ('default', 'cpu') or out['memory'] not in ('auto', 'low'):
        raise WorkflowError('Modo de memória ou dispositivo do codificador inválido.')
    return out


def configure(wf: Workflow, values: dict | None = None) -> Workflow:
    if not supported(wf):
        return wf
    selected = choices(wf.id, values)
    image = CATALOG['images'][selected['image']]
    encoder = CATALOG['encoders'][selected['text_encoder']]
    wf = copy.deepcopy(wf)
    image_node = wf.graph['1']
    image_node['class_type'] = 'UnetLoaderGGUF' if image['format'] == 'gguf' else 'UNETLoader'
    image_node['inputs'] = {'unet_name': image['filename']}
    if image['format'] == 'native':
        image_node['inputs']['weight_dtype'] = 'default'
    clip_id = '4' if wf.id == 'qwen21-viggle-turbo' else '2'
    wf.graph[clip_id]['inputs'].update(clip_name=encoder['filename'], device=selected['text_device'])
    required = [copy.deepcopy(image), copy.deepcopy(encoder), copy.deepcopy(CATALOG['vae'])]
    encoder_gguf = encoder.get('format') == 'gguf'
    if encoder_gguf:
        wf.graph[clip_id] = {'class_type': 'SPEQwen3VLLoader', 'inputs': {
            'clip_name': encoder['folder'].removeprefix('text_encoders/') + '/' + encoder['filename'],
            'device': selected['text_device'],
        }}
        required.append(copy.deepcopy(CATALOG['vision']))
    if wf.id == 'qwen21-viggle-turbo':
        if image['merged']:
            # Turbo já está nos pesos GGUF; outra LoRA alteraria o resultado duas vezes.
            del wf.graph['2']
            wf.graph['3']['inputs']['model'] = ['1', 0]
        else:
            required.append(copy.deepcopy(CATALOG['lora']))
        if selected['memory'] == 'low':
            wf.graph['3']['inputs']['device'] = 'cpu'
    wf.requires['models'] = required
    wf.requires['extensions'] = [copy.deepcopy(GGUF_EXTENSION)] if image['format'] == 'gguf' or encoder_gguf else []
    if encoder_gguf:
        wf.requires['bundled_nodes'] = [{'name': 'spe_qwen_gguf.py'}]
    wf.runtime_args = ['--lowvram', '--reserve-vram', '1'] if selected['memory'] == 'low' else []
    return wf


def signature(wf: Workflow) -> str:
    payload = {'id': wf.id, 'graph': wf.graph, 'requires': wf.requires, 'args': getattr(wf, 'runtime_args', [])}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def public(wf: Workflow, values: dict | None = None) -> dict | None:
    if not supported(wf):
        return None
    selected = choices(wf.id, values)
    resolved = configure(wf, selected)
    def item(key, value):
        return {'id': key, 'label': value['label'], 'size_bytes': value['size_bytes'], 'merged': value.get('merged', False), 'experimental': value.get('experimental', False), 'vision_bytes': CATALOG['vision']['size_bytes'] if value.get('format') == 'gguf' and key in CATALOG['encoders'] else 0}
    return {
        'values': selected,
        'images': [item(k, v) for k, v in CATALOG['images'].items() if wf.id in v['workflows']],
        'encoders': [item(k, v) for k, v in CATALOG['encoders'].items()],
        'vae': {'label': 'Qwen-Image 2.1 VAE BF16', 'size_bytes': CATALOG['vae']['size_bytes']},
        'total_bytes': sum(m['size_bytes'] for m in resolved.requires['models']),
        'presets': {
            'original': DEFAULTS,
            'minimum': {'image': 'turbo_q4_k_m' if wf.id == 'qwen21-viggle-turbo' else 'base_q4_k', 'text_encoder': 'q3_k_m', 'text_device': 'cpu', 'memory': 'low'},
            'compact': {'image': 'turbo_q4_k_m' if wf.id == 'qwen21-viggle-turbo' else 'base_q4_k', 'text_encoder': 'w4a8', 'text_device': 'cpu', 'memory': 'low'},
        },
        'notice': 'Tamanhos em disco. O uso de VRAM depende da resolução e do hardware; o perfil compacto não garante execução em 6 GB. CPU usa mais RAM e leva mais tempo.',
    }
