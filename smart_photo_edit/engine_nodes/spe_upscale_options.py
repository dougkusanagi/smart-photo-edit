"""SinSR e AdcSR em um passo, com carregamento sob demanda.

SinSR usa VAE na GPU em FP16, com recuperação FP32/CPU. A arquitetura do autor
é instalada em namespace próprio; o passo segue p_sample_loop(one_step=True):
prior em eta_T=.99, kappa=2, t=14.

AdcSR (Apache-2.0, https://github.com/Guaishou74851/AdcSR) é o OSEDiff comprimido:
UNet SD 2.1 podado em 25% dos canais, sem codificador do VAE, sem texto e sem
embedding de tempo, e meio decodificador. Cópia adaptada de model.py/forward.py do
autor, construída em 'meta' para não inicializar GB de pesos aleatórios.
A licença da implementação AdcSR acompanha o pacote em ADCSR_LICENSE.txt.
"""
import copy
import gc
import importlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import types
from types import ModuleType

import torch
import torch.nn.functional as F
import comfy.model_management as mm
import comfy.utils
import folder_paths


class NonFiniteUpscale(ValueError):
    pass


def tile_origins(length, tile_size, overlap):
    if length <= tile_size:
        return [0]
    last = length - tile_size
    origins = list(range(0, last + 1, tile_size - overlap))
    if origins[-1] != last:
        origins.append(last)
    return origins


def feather(length, overlap, at_start, at_end):
    weight = torch.ones(length, dtype=torch.float32)
    ramp = torch.linspace(1 / (overlap + 1), overlap / (overlap + 1), overlap)
    if not at_start:
        weight[:overlap] = ramp
    if not at_end:
        weight[-overlap:] = ramp.flip(0)
    return weight


def load_weights(path):
    data = torch.load(path, map_location='cpu', weights_only=True)
    data = data.get('state_dict', data)
    if data and all(key.startswith('module.') for key in data):
        data = {key.removeprefix('module.'): value for key, value in data.items()}
    return data


def sinsr_attention(self, x):
    """Atenção original do VQ, com consultas fatiadas e todas as chaves.

    A implementação do autor materializa três matrizes HW × HW na CPU.
    Num bloco de 192 px isso pode consumir mais de 15 GB. O fatiamento
    limita cada matriz de escores a 32 MiB, mesmo no backend matemático.
    Com cabeça única de 512 canais, o SDPA usa um kernel lento; matmul +
    softmax em FP32 dá o mesmo resultado (erro ~1e-5 em FP16) até ~6x mais rápido.
    """
    h = self.norm(x)
    batch, channels, height, width = h.shape
    q, k, v = (layer(h).flatten(2).transpose(1, 2).contiguous() for layer in (self.q, self.k, self.v))
    del h
    output = torch.empty_like(q)
    keys = k.transpose(1, 2)
    scale = channels ** -0.5
    chunk = max(1, min(256, (8 * 1024**2) // (batch * height * width)))
    for start in range(0, height * width, chunk):
        mm.throw_exception_if_processing_interrupted()
        scores = torch.bmm(q[:, start:start+chunk] * scale, keys)
        output[:, start:start+chunk] = torch.bmm(torch.softmax(scores.float(), dim=-1).to(v.dtype), v)
    output = output.transpose(1, 2).reshape(batch, channels, height, width)
    return x + self.proj_out(output)


def sinsr_quantize(self, z):
    """Busca exata em todos os códigos, com distâncias FP32 em fatias de 32 MiB."""
    batch, channels, height, width = z.shape
    flat = z.movedim(1, -1).reshape(-1, channels).float()
    codebook = self.embedding.weight.float()
    code_norm = codebook.square().sum(1)
    indices = torch.empty(flat.shape[0], dtype=torch.long, device=z.device)
    chunk = max(1, (8 * 1024**2) // codebook.shape[0])
    for start in range(0, flat.shape[0], chunk):
        mm.throw_exception_if_processing_interrupted()
        patch = flat[start:start+chunk]
        distance = patch @ codebook.T
        distance.mul_(-2).add_(code_norm).add_(patch.square().sum(1, keepdim=True))
        indices[start:start+chunk] = distance.argmin(1)
    quantized = self.embedding(indices).reshape(batch, height, width, channels).movedim(-1, 1).contiguous()
    return quantized, None, (None, None, indices)


def load_sinsr():
    namespace = 'spe_sinsr'
    if namespace not in sys.modules:
        root = Path(folder_paths.base_path) / 'spe_libraries' / 'sinsr'
        if not (root / 'models' / 'unet.py').is_file():
            raise ValueError('Biblioteca SinSR não preparada. Reinicie o app e tente novamente.')
        module = ModuleType(namespace)
        module.__path__ = [str(root)]
        sys.modules[namespace] = module
    unet_module = importlib.import_module(namespace + '.models.unet')
    vae_module = importlib.import_module(namespace + '.ldm.models.autoencoder')
    # Apenas o namespace privado do SinSR; não altera outros VAEs do ComfyUI.
    attention_module = importlib.import_module(namespace + '.ldm.modules.diffusionmodules.model')
    attention_module.AttnBlock.forward = sinsr_attention
    quantize_module = importlib.import_module(namespace + '.ldm.modules.vqvae.quantize')
    quantize_module.VectorQuantizer2.forward = sinsr_quantize
    model = unet_module.UNetModelSwin(image_size=64, in_channels=6, model_channels=160,
        out_channels=3, attention_resolutions=[64, 32, 16, 8], dropout=0, channel_mult=[1, 2, 2, 4],
        num_res_blocks=[2, 2, 2, 2], conv_resample=True, dims=2, use_fp16=False,
        num_head_channels=32, use_scale_shift_norm=True, resblock_updown=False,
        swin_depth=2, swin_embed_dim=192, window_size=8, mlp_ratio=4)
    vae = vae_module.VQModelTorch(ddconfig={'double_z': False, 'z_channels': 3, 'resolution': 256,
        'in_channels': 3, 'out_ch': 3, 'ch': 128, 'ch_mult': [1, 2, 4], 'num_res_blocks': 2,
        'attn_resolutions': [], 'dropout': 0, 'padding_mode': 'zeros'}, n_embed=8192, embed_dim=3)
    root = Path(folder_paths.models_dir) / 'upscale'
    model.load_state_dict(load_weights(root / 'SinSR_v1.pth'), strict=True)
    vae.load_state_dict(load_weights(root / 'autoencoder_vq_f4.pth'), strict=True)
    return {'model': model.float().eval().requires_grad_(False),
            'vae': vae.float().eval().requires_grad_(False)}


def safe_checkpoint(path):
    """Carrega tensores com weights_only, ignorando só callbacks Lightning inertes.

    Não importa classes do checkpoint nem libera funções arbitrárias do PyTorch.
    """
    class Empty:
        def __init__(self, *args, **kwargs):
            pass

        def __setstate__(self, state):
            pass

    unknown = torch.serialization.get_unsafe_globals_in_checkpoint(path)
    if any(not name.startswith('pytorch_lightning') for name in unknown):
        raise ValueError('Checkpoint contém objetos não permitidos.')
    with torch.serialization.safe_globals([(Empty, name) for name in unknown]):
        return torch.load(path, map_location='cpu', weights_only=True)


_adc_skip = []


def _adc_unet_forward(self, x):
    global _adc_skip
    x = self.conv_in(x)
    _adc_skip = [x]
    return self.body(x)


def _adc_down_attn(self, x):
    for i in range(2):
        x = self.attentions[i](self.resnets[i](x))
        _adc_skip.append(x)
    if self.downsamplers is not None:
        x = self.downsamplers[0](x)
        _adc_skip.append(x)
    return x


def _adc_down(self, x):
    for i in range(2):
        x = self.resnets[i](x)
        _adc_skip.append(x)
    return x


def _adc_mid(self, x):
    return self.resnets[1](self.attentions[0](self.resnets[0](x)))


def _adc_up_attn(self, x):
    for i in range(3):
        x = self.attentions[i](self.resnets[i](torch.cat([x, _adc_skip.pop()], dim=1)))
    return self.upsamplers[0](x) if self.upsamplers is not None else x


def _adc_up(self, x):
    for i in range(3):
        x = self.resnets[i](torch.cat([x, _adc_skip.pop()], dim=1))
    return self.upsamplers[0](x)


def _adc_resnet(self, x_in):
    x = self.conv1(self.nonlinearity(self.norm1(x_in)))
    x = self.conv2(self.nonlinearity(self.norm2(x)))
    return x + (x_in if self.in_channels == self.out_channels else self.conv_shortcut(x_in))


def _adc_transformer(self, x_in):
    b, c, h, w = x_in.shape
    x = self.proj_in(self.norm(x_in).permute(0, 2, 3, 1).reshape(b, h * w, c).contiguous())
    for block in self.transformer_blocks:
        x = x + block.attn1(block.norm1(x))
        x = x + block.ff(block.norm3(x))
    x = self.proj_out(x).reshape(b, h, w, c).permute(0, 3, 1, 2).contiguous()
    return x + x_in


def _adc_prune(model, fraction=0.75):
    """Poda os canais como o autor: os primeiros 75% de cada convolução, linear e normalização."""
    from diffusers.models.downsampling import Downsample2D
    from diffusers.models.upsampling import Upsample2D
    nn = torch.nn

    def parent_of(name):
        parent = model
        for part in name.split('.')[:-1]:
            parent = getattr(parent, part)
        return parent, name.split('.')[-1]

    for name, module in list(model.named_modules()):
        if hasattr(module, 'pruned'):
            continue
        new = None
        if isinstance(module, nn.Conv2d):
            i, o = int(module.in_channels * fraction), int(module.out_channels * fraction)
            new = nn.Conv2d(i, o, module.kernel_size, module.stride, module.padding, module.dilation,
                            module.groups, module.bias is not None)
            new.weight.data.copy_(module.weight[:o, :i])
            if module.bias is not None:
                new.bias.data.copy_(module.bias[:o])
        elif isinstance(module, nn.Linear):
            i, o = int(module.in_features * fraction), int(module.out_features * fraction)
            new = nn.Linear(i, o, bias=module.bias is not None)
            new.weight.data.copy_(module.weight[:o, :i])
            if module.bias is not None:
                new.bias.data.copy_(module.bias[:o])
        elif isinstance(module, nn.GroupNorm):
            channels = int(module.num_channels * fraction)
            groups = next(g for g in (32, 24, 16, 12, 8, 6, 4, 2, 1) if channels % g == 0)
            new = nn.GroupNorm(groups, channels, eps=module.eps, affine=module.affine)
            new.weight.data.copy_(module.weight[:channels])
            new.bias.data.copy_(module.bias[:channels])
        elif isinstance(module, nn.LayerNorm):
            size = int(module.normalized_shape[0] * fraction)
            new = nn.LayerNorm(size, eps=module.eps, elementwise_affine=module.elementwise_affine)
            new.weight.data.copy_(module.weight[:size])
            new.bias.data.copy_(module.bias[:size])
        elif isinstance(module, (Downsample2D, Upsample2D)):
            module.channels = int(module.channels * fraction)
        if new is not None:
            parent, last = parent_of(name)
            setattr(parent, last, new)
            new.pruned = True


class AdcSRNet(torch.nn.Module):
    def __init__(self, unet, decoder):
        super().__init__()
        from diffusers.models.attention import BasicTransformerBlock
        from diffusers.models.resnet import ResnetBlock2D
        from diffusers.models.transformers.transformer_2d import Transformer2DModel
        from diffusers.models.unets.unet_2d_blocks import (
            CrossAttnDownBlock2D, CrossAttnUpBlock2D, DownBlock2D, UNetMidBlock2DCrossAttn, UpBlock2D)
        nn = torch.nn
        del unet.time_embedding
        conv_in = nn.Conv2d(16, 320, 3, padding=1)
        conv_in.weight.data = unet.conv_in.weight.data.repeat(1, 4, 1, 1)
        conv_in.bias.data = unet.conv_in.bias.data
        unet.conv_in = conv_in
        conv_out = nn.Conv2d(320, 342, 3, padding=1)
        conv_out.weight.data = unet.conv_out.weight.data.repeat(86, 1, 1, 1)[:342]
        conv_out.bias.data = unet.conv_out.bias.data.repeat(86,)[:342]
        unet.conv_out = conv_out
        for module in unet.modules():
            if isinstance(module, ResnetBlock2D):
                del module.time_emb_proj
            elif isinstance(module, BasicTransformerBlock):
                del module.attn2, module.norm2
            elif isinstance(module, (nn.Dropout, nn.SiLU)):
                module.inplace = True
        forwards = ((CrossAttnDownBlock2D, _adc_down_attn), (DownBlock2D, _adc_down),
                    (UNetMidBlock2DCrossAttn, _adc_mid), (UpBlock2D, _adc_up),
                    (CrossAttnUpBlock2D, _adc_up_attn), (ResnetBlock2D, _adc_resnet),
                    (Transformer2DModel, _adc_transformer))
        for module in unet.modules():
            for kind, function in forwards:
                if isinstance(module, kind):
                    module.forward = types.MethodType(function, module)
                    break
        unet.forward = types.MethodType(_adc_unet_forward, unet)
        _adc_prune(unet)
        unet.body = nn.Sequential(*unet.down_blocks, unet.mid_block, *unet.up_blocks,
                                  unet.conv_norm_out, unet.conv_act, unet.conv_out)
        del decoder.conv_in, decoder.up_blocks, decoder.conv_norm_out, decoder.conv_act, decoder.conv_out
        self.body = nn.Sequential(nn.PixelUnshuffle(2), unet, decoder.mid_block)

    def forward(self, x):
        return self.body(x)


def load_adcsr():
    from diffusers.models.autoencoders.vae import Decoder
    from diffusers import UNet2DConditionModel
    root = Path(folder_paths.models_dir) / 'upscale' / 'adcsr'
    decoder_args = dict(in_channels=4, out_channels=3, up_block_types=['UpDecoderBlock2D'] * 4,
                        block_out_channels=[64, 128, 256, 256], layers_per_block=2, norm_num_groups=32,
                        act_fn='silu', norm_type='group', mid_block_add_attention=True)
    # Em 'meta' nada é inicializado: os pesos entram por atribuição, sem uma segunda cópia na RAM.
    with torch.device('meta'):
        unet = UNet2DConditionModel.from_config(json.loads((root / 'unet' / 'config.json').read_text()))
        half = Decoder(**decoder_args)
        net = AdcSRNet(unet, copy.deepcopy(half))
    state = safe_checkpoint(root / 'halfDecoder.ckpt')['state_dict']
    half.load_state_dict({k.replace('decoder.', '', 1): v for k, v in state.items() if 'decoder' in k},
                         strict=True, assign=True)
    net.load_state_dict(load_weights(root / 'net_params_200.pkl'), strict=True, assign=True)
    model = torch.nn.Sequential(net, *half.up_blocks, half.conv_norm_out, half.conv_act, half.conv_out)
    return {'model': model.float().eval().requires_grad_(False)}


def align_colors(output, source):
    dims = (1, 2)
    source_std, source_mean = torch.std_mean(source, dim=dims, keepdim=True, correction=0)
    output_std, output_mean = torch.std_mean(output, dim=dims, keepdim=True, correction=0)
    return ((output - output_mean) * source_std / output_std.clamp_min(1e-6) + source_mean).clamp(0, 1)


def sinsr_tiles(image, scale, seed, process, check, tile_size=128):
    """Blocos na resolução de entrada; o SinSR é nativamente 4×, com 2× reduzido."""
    if image.shape[0] != 1 or image.shape[-1] != 3 or scale not in (2, 4):
        raise ValueError('Use uma foto RGB por vez e ampliação 2× ou 4×.')
    _, h, w, _ = image.shape
    if h * w * scale * scale > 32_000_000:
        raise ValueError('A saída precisa ter até 32 megapixels.')
    padded_h, padded_w = max(64, math.ceil(h / 64) * 64), max(64, math.ceil(w / 64) * 64)
    source = image.detach().cpu().float().movedim(-1, 1)
    source = F.pad(source, (0, padded_w-w, 0, padded_h-h), mode='replicate')
    noise = torch.randn((1, 3, padded_h, padded_w), generator=torch.Generator().manual_seed(seed))
    result = torch.zeros(1, padded_h * scale, padded_w * scale, 3)
    weights = torch.zeros(1, padded_h * scale, padded_w * scale, 1)
    ys = tile_origins(padded_h, tile_size, 32)
    xs = tile_origins(padded_w, tile_size, 32)
    count = len(ys) * len(xs)
    for iy, y in enumerate(ys):
        for ix, x in enumerate(xs):
            check()
            bottom, right = min(y+tile_size, padded_h), min(x+tile_size, padded_w)
            top_ctx, left_ctx = max(0, y-32), max(0, x-32)
            bottom_ctx, right_ctx = min(bottom+32, padded_h), min(right+32, padded_w)
            patch = source[:, :, top_ctx:bottom_ctx, left_ctx:right_ctx]
            local_noise = noise[:, :, top_ctx:bottom_ctx, left_ctx:right_ctx]
            ph, pw = patch.shape[-2:]
            pad = (0, math.ceil(pw/64)*64-pw, 0, math.ceil(ph/64)*64-ph)
            generated = process(F.pad(patch, pad, mode='replicate'),
                                F.pad(local_noise, pad, mode='replicate'), iy*len(xs)+ix, count)
            check()
            if generated.shape != (1, 3, (ph+pad[3])*4, (pw+pad[1])*4):
                raise ValueError('SinSR retornou um bloco inválido.')
            if not torch.isfinite(generated).all():
                raise NonFiniteUpscale('SinSR retornou valores não finitos.')
            generated = generated[:, :, :ph*4, :pw*4].cpu().float().clamp(0, 1)
            if scale == 2:
                generated = F.interpolate(generated, size=(ph*2, pw*2), mode='bicubic',
                                          align_corners=False, antialias=True).clamp(0, 1)
            crop = generated[:, :, (y-top_ctx)*scale:(bottom-top_ctx)*scale,
                             (x-left_ctx)*scale:(right-left_ctx)*scale].movedim(1, -1)
            fy = feather((bottom-y)*scale, 32*scale, y==0, bottom==padded_h)
            fx = feather((right-x)*scale, 32*scale, x==0, right==padded_w)
            weight = (fy[:, None]*fx[None, :])[None, :, :, None]
            result[:, y*scale:bottom*scale, x*scale:right*scale] += crop*weight
            weights[:, y*scale:bottom*scale, x*scale:right*scale] += weight
    result.div_(weights)
    return result[:, :h*scale, :w*scale].contiguous()


class SPEOneStepUpscale:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'image': ('IMAGE',), 'algorithm': (['sinsr', 'adcsr'],),
                'scale': ([2, 4],), 'seed': ('INT', {'default': 0, 'min': 0, 'max': 2**53-1}),
                'prompt': ('STRING', {'default': ''})}}
    RETURN_TYPES = ('IMAGE',)
    FUNCTION = 'upscale'
    CATEGORY = 'Smart Photo Edit'

    def upscale(self, image, algorithm, scale, seed=0, prompt=''):
        if algorithm not in ('sinsr', 'adcsr'):
            raise ValueError('Modelo de upscale desconhecido.')
        mm.unload_all_models()
        mm.soft_empty_cache()
        pipeline = None
        model = vae = process = None
        device = mm.get_torch_device()
        previous_threads = torch.get_num_threads()
        try:
            # O trabalho na CPU não deve monopolizar todos os núcleos do desktop.
            if previous_threads > 4:
                torch.set_num_threads(4)
            mm.throw_exception_if_processing_interrupted()
            pipeline = load_sinsr() if algorithm == 'sinsr' else load_adcsr()
            mm.throw_exception_if_processing_interrupted()
            model, vae = pipeline['model'], pipeline.get('vae')
            size = sum(p.numel()*p.element_size() for p in model.parameters())
            mm.free_memory(size + 1024**3, device)
            model.to(device)
            if algorithm == 'adcsr':
                return (self.run_adcsr(image, scale, model, device),)
            vae_device = device if device.type == 'cuda' else torch.device('cpu')
            vae_dtype = torch.float16 if vae_device.type == 'cuda' else torch.float32
            try:
                vae.to(device=vae_device, dtype=vae_dtype)
            except Exception as exc:
                mm.raise_non_oom(exc)
                if vae_device.type != 'cuda':
                    raise
                exc.__traceback__ = None
                vae_device, vae_dtype = torch.device('cpu'), torch.float32
                vae.to(device=vae_device, dtype=vae_dtype)
                mm.soft_empty_cache()
            core = 128
            with torch.inference_mode():
                while True:
                    progress = None
                    def process(patch, noise, index, count):
                        nonlocal progress
                        if progress is None:
                            progress = comfy.utils.ProgressBar(count * 3)
                        progress.update_absolute(index * 3, count * 3)
                        mm.throw_exception_if_processing_interrupted()
                        lq = (patch * 2 - 1).to(device=vae_device, dtype=vae_dtype)
                        latent = vae.encode(F.interpolate(lq, scale_factor=4, mode='bicubic', align_corners=False))
                        progress.update_absolute(index * 3 + 1, count * 3)
                        noisy = (latent + noise.to(device=vae_device, dtype=vae_dtype) * (2*math.sqrt(.99))) / math.sqrt(1+4*.99)
                        predicted = model(noisy.to(device=device, dtype=torch.float32), torch.tensor([14], device=device),
                                          lq=lq.to(device=device, dtype=torch.float32)).to(device=vae_device, dtype=vae_dtype)
                        progress.update_absolute(index * 3 + 2, count * 3)
                        mm.throw_exception_if_processing_interrupted()
                        output = (vae.decode(predicted).cpu() + 1) / 2
                        mm.throw_exception_if_processing_interrupted()
                        progress.update_absolute((index + 1) * 3, count * 3)
                        return output
                    try:
                        return (sinsr_tiles(image, scale, seed, process, mm.throw_exception_if_processing_interrupted, core),)
                    except Exception as exc:
                        if isinstance(exc, NonFiniteUpscale) and vae_dtype == torch.float16:
                            exc.__traceback__ = None
                            vae_dtype = torch.float32
                            vae.float()
                            mm.soft_empty_cache()
                            continue
                        mm.raise_non_oom(exc)
                        if core > 64:
                            core = 64
                        elif vae_device.type == 'cuda':
                            vae_device = torch.device('cpu')
                            vae_dtype = torch.float32
                            vae.to(device=vae_device, dtype=vae_dtype)
                        else:
                            raise
                        exc.__traceback__ = None
                        mm.soft_empty_cache()
        finally:
            if pipeline is not None:
                for key in list(pipeline):
                    pipeline[key].cpu()
                pipeline.clear()
            # Solta também as referências locais e a closure antes de coletar.
            _adc_skip.clear()
            model = vae = process = pipeline = None
            gc.collect()
            mm.unload_all_models()
            mm.soft_empty_cache()
            if previous_threads > 4:
                torch.set_num_threads(previous_threads)

    @staticmethod
    def run_adcsr(image, scale, model, device):
        """AdcSR é determinístico e nativamente 4×; 2× reduz cada bloco, como no SinSR.

        A correção de cor AdaIN do autor usa a média e o desvio da imagem inteira.
        """
        # Blocos de 192 px (256 com contexto) refazem 2,6× cada pixel, contra 4× nos de 128 px.
        core = 192
        with torch.inference_mode():
            while True:
                progress = None
                def process(patch, noise, index, count):
                    nonlocal progress
                    if progress is None:
                        progress = comfy.utils.ProgressBar(count * 3)
                    progress.update_absolute(index * 3, count * 3)
                    mm.throw_exception_if_processing_interrupted()
                    output = model((patch * 2 - 1).to(device=device, dtype=torch.float32))
                    progress.update_absolute(index * 3 + 2, count * 3)
                    output = (output.cpu() + 1) / 2
                    mm.throw_exception_if_processing_interrupted()
                    progress.update_absolute((index + 1) * 3, count * 3)
                    return output
                try:
                    result = sinsr_tiles(image, scale, 0, process, mm.throw_exception_if_processing_interrupted, core)
                    return align_colors(result, image.detach().cpu().float())
                except Exception as exc:
                    mm.raise_non_oom(exc)
                    if core <= 64:
                        raise
                    _adc_skip.clear()
                    core = 128 if core > 128 else 64
                    exc.__traceback__ = None
                    mm.soft_empty_cache()


NODE_CLASS_MAPPINGS = {'SPEOneStepUpscale': SPEOneStepUpscale}
NODE_DISPLAY_NAME_MAPPINGS = {'SPEOneStepUpscale': 'Upscale SinSR / AdcSR (Smart Photo Edit)'}
