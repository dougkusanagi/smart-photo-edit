"""Carregador Qwen3-VL multimodal para o motor privado do Smart Photo Edit."""
import sys
import torch
import nodes
import folder_paths
import comfy.sd
import comfy.model_management


class SPEQwen3VLLoader:
    @classmethod
    def INPUT_TYPES(cls):
        base = nodes.NODE_CLASS_MAPPINGS['CLIPLoaderGGUF']
        return {'required': {'clip_name': (base.get_filename_list(),), 'device': (['default', 'cpu'],)}}

    RETURN_TYPES = ('CLIP',)
    FUNCTION = 'load'
    CATEGORY = 'Smart Photo Edit'

    def load(self, clip_name, device='cpu'):
        base = nodes.NODE_CLASS_MAPPINGS['CLIPLoaderGGUF']
        module = sys.modules[base.__module__]
        path = folder_paths.get_full_path('clip', clip_name)
        # O loader do fork carrega junto o mmproj e preserva o DeepStack Qwen3-VL.
        data = base().load_data([path])
        if 'model.visual.deepstack_merger_list.0.norm.weight' not in data[0]:
            raise ValueError('Codificador Qwen3-VL sem o projetor visual compatível para edição.')
        options = {'custom_operations': module.GGMLOps, 'initial_device': comfy.model_management.text_encoder_offload_device()}
        if device == 'cpu':
            options.update(load_device=torch.device('cpu'), offload_device=torch.device('cpu'), initial_device=torch.device('cpu'))
        clip = comfy.sd.load_text_encoder_state_dicts(
            clip_type=comfy.sd.CLIPType.QWEN_IMAGE, state_dicts=data,
            model_options=options, embedding_directory=folder_paths.get_folder_paths('embeddings'),
        )
        clip.patcher = module.GGUFModelPatcher.clone(clip.patcher)
        return (clip,)


NODE_CLASS_MAPPINGS = {'SPEQwen3VLLoader': SPEQwen3VLLoader}
NODE_DISPLAY_NAME_MAPPINGS = {'SPEQwen3VLLoader': 'Qwen3-VL 8B GGUF (Smart Photo Edit)'}
