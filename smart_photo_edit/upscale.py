"""Upscale generativo em um passo: SeedVR2 (nós nativos do ComfyUI), AdcSR e SinSR."""
from __future__ import annotations

import io
import json

from . import paths

from PIL import Image, ImageOps

from .workflows import Workflow, WorkflowError, parse_workflow

MODE = 'upscale'
MAX_OUTPUT_PIXELS = 32_000_000
DEFAULT_MODEL = 'seedvr2'
OPTIONS = json.loads((paths.PACKAGE_DIR / 'model_data' / 'upscale.json').read_text(encoding='utf-8'))
BASE_ARGS = ['--lowvram', '--disable-dynamic-vram', '--reserve-vram', '1', '--cache-none']


def validate_seed(value) -> int:
    if type(value) not in (int, str) or not str(value).isascii() or not str(value).isdigit():
        raise WorkflowError('Semente do upscale inválida.')
    seed = int(value)
    if not 0 <= seed <= 2**53 - 1:
        raise WorkflowError('Semente do upscale inválida.')
    return seed


def validate_scale(value) -> int:
    # Evita coerção silenciosa de 2.5, True, objetos e combinações arbitrárias.
    if type(value) not in (int, str) or str(value) not in ('2', '4'):
        raise WorkflowError('Escolha uma ampliação de 2× ou 4×.')
    return int(value)


def validate_model(value) -> str:
    if not isinstance(value, str) or value not in OPTIONS:
        raise WorkflowError('Modelo de upscale desconhecido.')
    return value


def max_pixels(model=DEFAULT_MODEL) -> int:
    return OPTIONS[model].get('max_output_pixels', MAX_OUTPUT_PIXELS)


def catalog(models_dir=None) -> list[dict]:
    return [{'id': key, 'label': option['label'], 'description': option['description'],
             'license': option['license'], 'license_url': option['license_url'],
             'size_bytes': sum(m['size_bytes'] for m in option['models']),
             'downloaded': bool(models_dir) and all((models_dir / m['folder'] / m['filename']).is_file()
                         and (models_dir / m['folder'] / m['filename']).stat().st_size == m['size_bytes']
                         for m in option['models']),
             'steps': 1, 'native_scale': option.get('native_scale'),
             'max_output_pixels': max_pixels(key)} for key, option in OPTIONS.items()]


def prompt(model: str) -> str:
    """Nenhum dos modelos usa texto: o prompt registrado é vazio."""
    return ''


def label(scale: int, model=DEFAULT_MODEL) -> str:
    return f'Upscale generativo {scale}× ({OPTIONS[model]["label"]})'


def dimensions(data: bytes, scale: int, model=DEFAULT_MODEL) -> tuple[int, int]:
    try:
        with Image.open(io.BytesIO(data)) as source:
            source.verify()
        with Image.open(io.BytesIO(data)) as source:
            width, height = source.size
            if source.getexif().get(274) in (5, 6, 7, 8):
                width, height = height, width
    except (OSError, ValueError, SyntaxError, Image.DecompressionBombError) as exc:
        raise WorkflowError('Não consegui ler a foto. Use outro PNG, JPG ou WebP.') from exc
    limit = max_pixels(model)
    if width * height * scale * scale > limit:
        raise WorkflowError(f'A saída ultrapassa {limit // 1_000_000} megapixels neste modelo. Escolha 2×, '
                            'reduza a foto antes de ampliar ou use outro modelo.')
    return width, height


def finish(original: bytes, generated: bytes, scale: int, model=DEFAULT_MODEL) -> bytes:
    """Valida resolução e preserva o alfa original e o perfil de cor, sem recriar transparência."""
    width, height = dimensions(original, scale, model)
    with Image.open(io.BytesIO(original)) as source, Image.open(io.BytesIO(generated)) as output:
        if output.size != (width * scale, height * scale):
            raise ValueError('O motor devolveu uma resolução diferente da ampliação pedida.')
        source = ImageOps.exif_transpose(source)
        photo = output.convert('RGB')
        if 'A' in source.getbands() or 'transparency' in source.info:
            photo.putalpha(source.convert('RGBA').getchannel('A').resize(photo.size, Image.Resampling.LANCZOS))
        result = io.BytesIO()
        # ResultStore.save recodifica com metadados; aqui a compressão mínima evita pagar duas vezes (~4 s a menos em 32 MP).
        photo.save(result, format='PNG', compress_level=1, **{key: source.info[key] for key in ('icc_profile', 'exif') if key in source.info})
        return result.getvalue()


def workflow(model=DEFAULT_MODEL) -> Workflow:
    validate_model(model)
    option = OPTIONS[model]
    scale = {'key': 'scale', 'label': 'Ampliação', 'type': 'choice', 'default': 2,
             'options': [{'label': '2×', 'value': 2}, {'label': '4×', 'value': 4}]}
    if model == 'seedvr2':
        # Mesmo grafo do template oficial do SeedVR2 no ComfyUI: redimensiona, restaura em um passo e corrige a cor.
        graph = {
            '1': {'class_type': 'LoadImage', 'inputs': {'image': 'example.png'}},
            '2': {'class_type': 'ImageScaleBy', 'inputs': {'image': ['1', 0], 'upscale_method': 'lanczos', 'scale_by': 2}},
            '3': {'class_type': 'SeedVR2Preprocess', 'inputs': {'resized_images': ['2', 0]}},
            '4': {'class_type': 'VAELoader', 'inputs': {'vae_name': option['models'][1]['filename']}},
            '5': {'class_type': 'UNETLoader', 'inputs': {'unet_name': option['models'][0]['filename'], 'weight_dtype': 'default'}},
            '6': {'class_type': 'VAEEncodeTiled', 'inputs': {'pixels': ['3', 0], 'vae': ['4', 0], 'tile_size': 512,
                  'overlap': 128, 'temporal_size': 4096, 'temporal_overlap': 8}},
            '7': {'class_type': 'SeedVR2Conditioning', 'inputs': {'model': ['5', 0], 'vae_conditioning': ['6', 0]}},
            '8': {'class_type': 'KSampler', 'inputs': {'model': ['5', 0], 'positive': ['7', 0], 'negative': ['7', 1],
                  'latent_image': ['6', 0], 'seed': 0, 'steps': 1, 'cfg': 1, 'sampler_name': 'euler',
                  'scheduler': 'simple', 'denoise': 1}},
            '9': {'class_type': 'VAEDecodeTiled', 'inputs': {'samples': ['8', 0], 'vae': ['4', 0], 'tile_size': 512,
                  'overlap': 128, 'temporal_size': 4096, 'temporal_overlap': 8}},
            '10': {'class_type': 'SeedVR2PostProcessing', 'inputs': {'images': ['9', 0], 'original_resized_images': ['2', 0],
                   'color_correction_method': 'lab'}},
            '11': {'class_type': 'PreviewImage', 'inputs': {'images': ['10', 0]}},
            # O prompt é só um rótulo: o nó abaixo não participa do grafo executado.
            '12': {'class_type': 'PrimitiveString', 'inputs': {'value': ''}},
        }
        bindings = {'image': ['1', 'image'], 'prompt': ['12', 'value'], 'seed': ['8', 'seed']}
        scale['bind'] = [['2', 'scale_by']]
    else:
        graph = {
            '1': {'class_type': 'LoadImage', 'inputs': {'image': 'example.png'}},
            '5': {'class_type': 'SPEOneStepUpscale', 'inputs': {'image': ['1', 0], 'algorithm': model,
                  'scale': 2, 'seed': 0, 'prompt': prompt(model)}},
            '7': {'class_type': 'PreviewImage', 'inputs': {'images': ['5', 0]}},
        }
        bindings = {'image': ['1', 'image'], 'prompt': ['5', 'prompt'], 'seed': ['5', 'seed']}
        scale['bind'] = [['5', 'scale']]
    wf = parse_workflow({
        'format': 'smart-photo-edit/workflow@1', 'name': 'Upscale generativo · ' + option['label'],
        'description': option['description'], 'requires': {'models': option['models']},
        'bindings': bindings, 'params': [scale], 'prompt': graph,
    }, wf_id=model + '-upscale', source='builtin')
    if model != 'seedvr2':
        wf.requires['bundled_nodes'] = [{'name': 'spe_upscale_options.py'}]
    for key in ('libraries', 'engine_packages'):
        wf.requires[key] = option.get(key, [])
    wf.runtime_args = runtime_args(model)
    wf.upscale_model = model
    return wf


def runtime_args(model=DEFAULT_MODEL):
    # Atenção dividida só vale para os nós próprios; o SeedVR2 usa a atenção nativa do ComfyUI.
    return list(BASE_ARGS) if model == 'seedvr2' else [*BASE_ARGS[:2], '--use-split-cross-attention', *BASE_ARGS[2:]]
