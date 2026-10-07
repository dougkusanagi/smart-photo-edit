"""Nó do Smart Photo Edit: aplica uma LoRA adicional do Qwen-Image 2.1 em tempo de execução.

Mesmo princípio do ViggleTurboLora (y = Wx + BAx, sem fundir nos pesos int8/bf16), mas aceita LoRAs
PEFT/DiffSynth com chaves `transformer_blocks.N.….lora_A.default.weight` e pode ser empilhado depois de outra LoRA.
"""

import json

import torch
import torch.nn.functional as F

import comfy.patcher_extension
import comfy.utils
import folder_paths


class _LoraWeights:
    """Original na CPU e uma única cópia de execução por par A/B."""

    def __init__(self, a, b):
        self.original = (a.to(device="cpu"), b.to(device="cpu"))
        self.max_cached_element_size = min(a.element_size(), b.element_size())
        self.prepared = None
        self.key = None
        self.cache_cast = False

    def for_input(self, x):
        key = (x.device, x.dtype)
        if self.prepared is None or self.key != key:
            a, b = self.original
            # Cachear FP32 na GPU duplicaria a VRAM de uma LoRA BF16 na GTX 1660 Ti.
            # Nesse caso mantemos os pesos compactos e convertemos só o par em uso.
            self.cache_cast = x.device.type == "cpu" or x.element_size() <= self.max_cached_element_size
            dtypes = (x.dtype, x.dtype) if self.cache_cast else (a.dtype, b.dtype)
            self.prepared = None  # solta a cópia anterior antes de alocar outra
            self.prepared = (a.to(device=x.device, dtype=dtypes[0]), b.to(device=x.device, dtype=dtypes[1]))
            self.key = key
        if self.cache_cast:
            return self.prepared
        return tuple(w.to(dtype=x.dtype) for w in self.prepared)

    def clear(self):
        self.prepared = None
        self.key = None


class _LoraState:
    def __init__(self, lora):
        self.weights = {name: _LoraWeights(*ab) for name, ab in lora.items()}
        self.plan = None
        self.sampling_depth = 0

    def clear(self):
        self.plan = None
        for weights in self.weights.values():
            weights.clear()


def _lora_fwd(x, ab):
    a, b = ab.for_input(x)
    return F.linear(F.linear(x, a), b)


def _add_hook(mod, ab):
    return mod.register_forward_hook(lambda m, inp, out: out + _lora_fwd(inp[0], ab))


def _add_mlp_hooks(mlp, gate, up, down, hooks):
    # SwiGLU fundido: o ramo `out` roda dentro de um kernel que ignora hooks; soma-se ao final do MLP.
    h = {}

    def gate_up_hook(m, inp, out):
        h["gu"] = out + torch.cat([_lora_fwd(inp[0], gate), _lora_fwd(inp[0], up)], -1)
        return h["gu"]

    def mlp_hook(m, inp, out):
        g, u = h.pop("gu").chunk(2, -1)
        return out + _lora_fwd(F.silu(g) * u, down)

    # Registra um por vez para permitir limpeza mesmo se o segundo registro falhar.
    hooks.append(mlp.gate_up.register_forward_hook(gate_up_hook))
    hooks.append(mlp.register_forward_hook(mlp_hook))


def _hook_plan(dm, lora):
    plan = []
    for name, ab in lora.items():
        parent, _, leaf = name.rpartition(".")
        module = dm.get_submodule(parent)
        if not getattr(module, "fused", False):
            plan.append((dm.get_submodule(name), (ab,)))
        elif leaf == "out":
            plan.append((module, (lora[parent + ".gate_layer"], lora[parent + ".proj"], ab)))
    return plan


def _run_with_lora(state, executor, *args, **kwargs):
    hooks = []
    try:
        dm = executor.class_obj
        # Resolve módulos só uma vez por amostragem, após as fusões do motor.
        if state.plan is None or state.plan[0] is not dm:
            state.plan = (dm, _hook_plan(dm, state.weights))
        for module, weights in state.plan[1]:
            if len(weights) == 1:
                hooks.append(_add_hook(module, weights[0]))
            else:
                _add_mlp_hooks(module, *weights, hooks)
        return executor(*args, **kwargs)
    finally:
        # O modelo é compartilhado: hooks nunca sobrevivem à chamada, inclusive se o registro falhar.
        for hk in hooks:
            hk.remove()
        if not state.sampling_depth:
            state.clear()


def _run_sampling(state, executor, *args, **kwargs):
    state.sampling_depth += 1
    try:
        return executor(*args, **kwargs)
    finally:
        state.sampling_depth -= 1
        if not state.sampling_depth:
            # Sucesso, erro ou cancelamento: nenhuma cópia da LoRA fica presa na GPU entre edições.
            state.clear()


class SPEAddonLora:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL",),
            "lora_name": (folder_paths.get_filename_list("loras"), {"tooltip": "LoRA do Qwen-Image 2.1 (PEFT/DiffSynth)."}),
            "strength": ("FLOAT", {"default": 1.0, "min": -4.0, "max": 4.0, "step": 0.05}),
        }}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load"
    CATEGORY = "loaders"

    def load(self, model, lora_name, strength):
        if strength == 0:
            return (model,)
        sd, meta = comfy.utils.load_torch_file(folder_paths.get_full_path_or_raise("loras", lora_name), return_metadata=True)
        try:
            cfg = json.loads((meta or {}).get("lora_adapter_metadata", "{}"))
        except ValueError:
            cfg = {}
        scale = strength * cfg.get("transformer.lora_alpha", 1) / cfg.get("transformer.r", 1)
        lora = {}
        for key in sd:
            for suffix in (".lora_A.default.weight", ".lora_A.weight"):
                if key.endswith(suffix):
                    name = key.removeprefix("transformer.").removesuffix(suffix)
                    lora[name] = [sd[key], sd[key.replace("lora_A", "lora_B")] * scale]
                    break
        if not lora:
            raise ValueError("Esta LoRA não tem o formato esperado (lora_A/lora_B do Qwen-Image 2.1).")
        state = _LoraState(lora)
        m = model.clone()
        key = "spe_addon_lora:" + lora_name
        m.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, key,
                               lambda executor, *a, **kw: _run_with_lora(state, executor, *a, **kw))
        m.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.OUTER_SAMPLE, key,
                               lambda executor, *a, **kw: _run_sampling(state, executor, *a, **kw))
        return (m,)


NODE_CLASS_MAPPINGS = {"SPEAddonLora": SPEAddonLora}
NODE_DISPLAY_NAME_MAPPINGS = {"SPEAddonLora": "Smart Photo Edit LoRA adicional (Qwen-Image 2.1)"}
