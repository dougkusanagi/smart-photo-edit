"""Catálogo fechado de pesos compatíveis com os workflows Qwen-Image 2.1."""
from __future__ import annotations

import copy
import hashlib
import json
from typing import Iterable

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
    image = CATALOG['images'][out['image']]
    encoder = CATALOG['encoders'][out['text_encoder']]
    if image.get('encoder_family') != encoder.get('family') or not encoder.get('family'):
        raise WorkflowError('O codificador não pertence à família multimodal exigida pelo modelo de imagem.')
    if image.get('family') != CATALOG['vae'].get('family') or not image.get('family'):
        raise WorkflowError('O VAE não é compatível com a família do modelo de imagem.')
    if encoder.get('format') == 'gguf' and encoder['family'] != CATALOG['vision'].get('family'):
        raise WorkflowError('O projetor visual não é compatível com o codificador GGUF.')
    if wf_id == 'qwen21-viggle-turbo' and not image['merged'] and image['family'] != CATALOG['lora'].get('family'):
        raise WorkflowError('A LoRA Turbo não é compatível com o modelo de imagem.')
    if out['text_device'] not in ('default', 'cpu') or out['memory'] not in ('auto', 'low'):
        raise WorkflowError('Modo de memória ou dispositivo do codificador inválido.')
    return out


def configure(wf: Workflow, values: dict | None = None, addons: Iterable[str] = ()) -> Workflow:
    if not supported(wf):
        if addons:
            raise WorkflowError('Este workflow não aceita LoRAs adicionais.')
        return wf
    selected = choices(wf.id, values)
    image = CATALOG['images'][selected['image']]
    encoder = CATALOG['encoders'][selected['text_encoder']]
    wf = copy.deepcopy(wf)
    wf.model_selection = dict(selected)
    image_node = wf.graph['1']
    image_node['class_type'] = 'UnetLoaderGGUF' if image['format'] == 'gguf' else 'UNETLoader'
    image_node['inputs'] = {'unet_name': image['filename']}
    if image['format'] == 'native':
        image_node['inputs']['weight_dtype'] = 'default'
    clip_id = '4' if wf.id == 'qwen21-viggle-turbo' else '2'
    wf.graph[clip_id]['inputs'].update(clip_name=encoder['filename'], device=selected['text_device'])
    required = [copy.deepcopy(image), copy.deepcopy(encoder), copy.deepcopy(CATALOG['vae'])]
    for model, role in zip(required, ('image', 'text_encoder', 'vae')):
        model['role'] = role
    encoder_gguf = encoder.get('format') == 'gguf'
    if encoder_gguf:
        wf.graph[clip_id] = {'class_type': 'SPEQwen3VLLoader', 'inputs': {
            'clip_name': encoder['folder'].removeprefix('text_encoders/') + '/' + encoder['filename'],
            'device': selected['text_device'],
        }}
        required.append({**copy.deepcopy(CATALOG['vision']), 'role': 'vision'})
    if wf.id == 'qwen21-viggle-turbo':
        if image['merged']:
            # Turbo já está nos pesos GGUF; outra LoRA alteraria o resultado duas vezes.
            del wf.graph['2']
            wf.graph['3']['inputs']['model'] = ['1', 0]
        else:
            required.append({**copy.deepcopy(CATALOG['lora']), 'role': 'turbo_lora'})
        if selected['memory'] == 'low':
            wf.graph['3']['inputs']['device'] = 'cpu'
    wf.requires['models'] = required
    wf.requires['extensions'] = [copy.deepcopy(GGUF_EXTENSION)] if image['format'] == 'gguf' or encoder_gguf else []
    wf.requires['bundled_nodes'] = [{'name': 'spe_addon_lora.py'}] + ([{'name': 'spe_qwen_gguf.py'}] if encoder_gguf else [])
    wf.runtime_args = ['--lowvram', '--reserve-vram', '1'] if selected['memory'] == 'low' else []
    # O motor só reinicia quando o essencial muda: complementos escolhidos por edição ficam fora da assinatura.
    wf.engine_signature = signature(wf)
    for addon_id in dict.fromkeys(addons):
        add_addon(wf, addon_id)
    return wf


def add_addon(wf: Workflow, addon_id: str) -> None:
    """Empilha uma LoRA do catálogo depois da cadeia de modelo, antes do cache/amostrador."""
    addon = CATALOG['addons'].get(addon_id)
    if addon is None:
        raise WorkflowError('Complemento desconhecido.')
    node_id = f'_spe_addon_{addon_id}'
    cache = [k for k, n in wf.graph.items() if n['class_type'] == 'QwenImage21Cache']
    targets = cache or [k for k, n in wf.graph.items() if n['class_type'] in ('KSampler', 'BasicGuider') and 'model' in n['inputs']]
    if len(targets) != 1:
        raise WorkflowError('Não encontrei onde aplicar a LoRA adicional neste workflow.')
    consumer = wf.graph[targets[0]]['inputs']
    wf.graph[node_id] = {'class_type': 'SPEAddonLora', 'inputs': {'model': consumer['model'], 'lora_name': addon['lora']['filename'], 'strength': addon['strength']}}
    consumer['model'] = [node_id, 0]
    wf.requires['models'] = [*wf.requires['models'], {**copy.deepcopy(addon['lora']), 'addon': addon_id, 'role': 'addon', 'label': addon['label']}]


def signature(wf: Workflow) -> str:
    if hasattr(wf, 'engine_signature'):
        return wf.engine_signature
    payload = {'id': wf.id, 'graph': wf.graph, 'requires': wf.requires, 'args': getattr(wf, 'runtime_args', [])}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


PROFILES = (
    ('original', 'Original', 'INT8 com execução automática. Padrão do app; não é uma recomendação para 6 GB.', False),
    ('compact', 'Compacto', 'GGUF Q4, codificador W4A8 na CPU, pouca VRAM e 768 px. Ainda não validado em 6 GB.', True),
    ('minimum', 'Mínimo', 'GGUF Q4 e codificador Q3 na CPU, pouca VRAM. Menor download; pode perder qualidade. Não validado em 6 GB.', True),
)


def presets(wf_id: str) -> dict:
    image = 'turbo_q4_k_m' if wf_id == 'qwen21-viggle-turbo' else 'base_q4_k'
    return {
        'original': DEFAULTS,
        'compact': {'image': image, 'text_encoder': 'w4a8', 'text_device': 'cpu', 'memory': 'low'},
        'minimum': {'image': image, 'text_encoder': 'q3_k_m', 'text_device': 'cpu', 'memory': 'low'},
    }


def model_name(m: dict) -> str:
    base = m.get('label', '').split(' · ')[0]
    return {
        'image': ('Qwen-Image 2.1 Viggle Turbo ' if m.get('merged') else 'Qwen-Image 2.1 ') + base,
        'text_encoder': base,
        'vae': 'Qwen-Image 2.1 VAE BF16',
        'vision': 'Projetor visual Qwen3-VL 8B',
        'turbo_lora': 'LoRA Viggle Turbo',
    }.get(m['role'], base or m['filename'])


def missing_bytes(models: Iterable[dict], models_dir) -> int:
    """Bytes que ainda faltam baixar; downloads parciais (.part) contam o que já chegou."""
    total = 0
    for m in models:
        target = models_dir.joinpath(m['folder'], m['filename'])
        if target.exists():
            continue
        part = target.with_name(target.name + '.part')
        total += max(0, m.get('size_bytes', 0) - (part.stat().st_size if part.exists() else 0))
    return total


def profiles(wf: Workflow, models_dir=None) -> list[dict]:
    """Perfis prontos com a lista de modelos e, se a pasta do motor for conhecida, quanto falta baixar."""
    out = []
    for key, label, summary, experimental in PROFILES:
        values = presets(wf.id)[key]
        required = configure(wf, values).requires['models']
        out.append({
            'id': key, 'label': label, 'summary': summary, 'experimental': experimental, 'values': values,
            'models': [{'label': model_name(m), 'size_bytes': m['size_bytes']} for m in required],
            'total_bytes': sum(m['size_bytes'] for m in required),
            'missing_bytes': None if models_dir is None else missing_bytes(required, models_dir),
        })
    return out


def public(wf: Workflow, values: dict | None = None, models_dir=None) -> dict | None:
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
        'compatibility': 'Catálogo restrito a Qwen-Image 2.1 + Qwen3-VL 8B. As quantizações podem ser combinadas independentemente. VAE e projetor visual são definidos pelo app; pesos Turbo com LoRA incorporada não recebem a mesma LoRA novamente. GGUF continua experimental; compatibilidade de arquitetura não comprova execução em 6 GB.',
        'total_bytes': sum(m['size_bytes'] for m in resolved.requires['models']),
        'presets': presets(wf.id),
        'profiles': profiles(wf, models_dir),
        'missing_bytes': None if models_dir is None else missing_bytes(resolved.requires['models'], models_dir),
        'addons': [{'id': k, 'label': v['label'], 'description': v['description'], 'prompt': v['prompt'], 'size_bytes': v['lora']['size_bytes'], 'source': v['source'], 'license': v['license']} for k, v in CATALOG['addons'].items()],
        'notice': 'Tamanhos em disco. O uso de VRAM depende da resolução e do hardware; o perfil compacto não garante execução em 6 GB. CPU usa mais RAM e leva mais tempo.',
    }
