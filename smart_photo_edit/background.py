"""Modo "Remover fundo": BiRefNet nativo do ComfyUI (máscara) + alfa, sem gerar pixels nem usar o Qwen."""
from __future__ import annotations

import io

from PIL import Image, ImageChops, ImageOps

from .workflows import Workflow, parse_workflow

MODE = 'remove_background'
LABEL = 'Remover fundo (BiRefNet)'
WORKFLOW_ID = 'birefnet-remove-background'
MODEL = {
    'folder': 'background_removal', 'filename': 'birefnet.safetensors', 'size_bytes': 444473596,
    'sha256': '9ab37426bf4de0567af6b5d21b16151357149139362e6e8992021b8ce356a154',
    'url': 'https://huggingface.co/Comfy-Org/BiRefNet/resolve/25511f8787e51912e1480706b4e47b8f467fbf72/background_removal/birefnet.safetensors',
}


def apply_alpha(original: bytes, segmented: bytes) -> bytes:
    """Usa apenas o alfa do motor, preservando pixels, resolução e perfil da foto."""
    with Image.open(io.BytesIO(original)) as source, Image.open(io.BytesIO(segmented)) as output:
        if 'A' not in output.getbands():
            raise ValueError('O motor não devolveu uma máscara de transparência.')
        photo = ImageOps.exif_transpose(source).convert('RGBA')
        alpha = output.getchannel('A')
        if alpha.size != photo.size:
            alpha = alpha.resize(photo.size, Image.Resampling.LANCZOS)
        # Uma entrada já transparente não deve recuperar o fundo descartado.
        photo.putalpha(ImageChops.multiply(photo.getchannel('A'), alpha))
        result = io.BytesIO()
        photo.save(result, format='PNG', **{key: photo.info[key] for key in ('icc_profile', 'exif') if key in photo.info})
        return result.getvalue()


def workflow() -> Workflow:
    """Grafo equivalente ao template oficial "Remove Background (BiRefNet)" do ComfyUI 0.37."""
    return parse_workflow({
        'format': 'smart-photo-edit/workflow@1',
        'name': 'Remover fundo · BiRefNet',
        'description': 'Recorta o sujeito com um modelo de segmentação (BiRefNet, MIT) e devolve PNG com transparência.',
        'requires': {'models': [dict(MODEL)]},
        # O prompt é só um rótulo: o nó abaixo não participa do grafo executado.
        'bindings': {'image': ['1', 'image'], 'prompt': ['9', 'value']},
        'prompt': {
            '1': {'class_type': 'LoadImage', 'inputs': {'image': 'example.png'}},
            '2': {'class_type': 'LoadBackgroundRemovalModel', 'inputs': {'bg_removal_name': MODEL['filename']}},
            '3': {'class_type': 'RemoveBackground', 'inputs': {'bg_removal_model': ['2', 0], 'image': ['1', 0]}},
            '4': {'class_type': 'InvertMask', 'inputs': {'mask': ['3', 0]}},
            '5': {'class_type': 'JoinImageWithAlpha', 'inputs': {'image': ['1', 0], 'alpha': ['4', 0]}},
            '6': {'class_type': 'PreviewImage', 'inputs': {'images': ['5', 0]}},
            '9': {'class_type': 'PrimitiveString', 'inputs': {'value': ''}},
        },
    }, wf_id=WORKFLOW_ID, source='builtin')
