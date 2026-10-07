"""Testes numéricos do nó; PyTorch é dependência do motor privado, não do servidor web."""
import runpy
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest

from smart_photo_edit import paths


@pytest.fixture
def lora_node(monkeypatch):
    torch = pytest.importorskip("torch", reason="Execute também com o Python do motor para os testes numéricos da LoRA.")
    comfy = ModuleType("comfy")
    extension = ModuleType("comfy.patcher_extension")
    extension.WrappersMP = SimpleNamespace(DIFFUSION_MODEL="diffusion", OUTER_SAMPLE="sample")
    utils = ModuleType("comfy.utils")
    utils.load_torch_file = Mock()
    comfy.patcher_extension, comfy.utils = extension, utils
    folders = SimpleNamespace(get_full_path_or_raise=lambda *_: "addon.safetensors")
    for name, module in {"comfy": comfy, "comfy.patcher_extension": extension, "comfy.utils": utils, "folder_paths": folders}.items():
        monkeypatch.setitem(sys.modules, name, module)
    namespace = runpy.run_path(str(paths.PACKAGE_DIR / "engine_nodes/spe_addon_lora.py"))
    return SimpleNamespace(utils=utils, **namespace)


class Executor:
    def __init__(self, module):
        self.class_obj = module

    def __call__(self, *args, **kwargs):
        return self.class_obj(*args, **kwargs)


def small_model(node, dtype=None):
    t = node.torch
    t.manual_seed(42)
    dtype = dtype or t.float32
    model = t.nn.Sequential(t.nn.Linear(4, 4, bias=False, dtype=dtype))
    a, b = t.randn(2, 4, dtype=t.bfloat16), t.randn(4, 2, dtype=t.bfloat16)
    state = node._LoraState({"0": (a, b)})
    x = t.randn(3, 4, dtype=dtype)
    return model, state, x


def test_sampling_reuses_weights_and_module_lookup_without_leaking_hooks(lora_node, monkeypatch):
    n = lora_node
    model, state, x = small_model(n)
    base = model(x)
    a, b = state.weights["0"].original
    expected = base + n.torch.nn.functional.linear(n.torch.nn.functional.linear(x, a.float()), b.float())
    lookups = Mock(wraps=model.get_submodule)
    monkeypatch.setattr(model, "get_submodule", lookups)

    def sample():
        first = n._run_with_lora(state, Executor(model), x)
        prepared = state.weights["0"].prepared
        second = n._run_with_lora(state, Executor(model), x)
        assert state.weights["0"].prepared is prepared
        assert lookups.call_count == 2  # parent e camada, só no primeiro passo
        assert not model[0]._forward_hooks
        n.torch.testing.assert_close(first, expected, rtol=0, atol=0)
        n.torch.testing.assert_close(second, expected, rtol=0, atol=0)

    n._run_sampling(state, sample)
    assert state.plan is None and state.weights["0"].prepared is None
    n.torch.testing.assert_close(model(x), base, rtol=0, atol=0)


@pytest.mark.parametrize("failed_stage", ["register", "forward", "cancel"])
def test_failed_sampling_always_cleans_weights_and_hooks(lora_node, monkeypatch, failed_stage):
    n = lora_node
    model, state, x = small_model(n)
    if failed_stage == "register":
        model.add_module("1", n.torch.nn.Linear(4, 4, bias=False))
        state.weights["1"] = n._LoraWeights(*state.weights["0"].original)
        monkeypatch.setattr(model[1], "register_forward_hook", Mock(side_effect=RuntimeError("registration")))

    def sample():
        executor = Executor(model)
        if failed_stage == "forward":
            class FailingExecutor:
                class_obj = model
                def __call__(self, *a, **kw):
                    model(x)  # prepara os pesos antes de falhar
                    raise RuntimeError("forward")
            executor = FailingExecutor()
        result = n._run_with_lora(state, executor, x)
        if failed_stage == "cancel":
            raise KeyboardInterrupt("cancelled")
        return result

    with pytest.raises((RuntimeError, KeyboardInterrupt)):
        n._run_sampling(state, sample)
    assert state.sampling_depth == 0 and state.plan is None
    assert all(w.prepared is None for w in state.weights.values())
    assert all(not m._forward_hooks for m in model.modules())


def test_direct_diffusion_call_also_releases_cache(lora_node):
    n = lora_node
    model, state, x = small_model(n)
    n._run_with_lora(state, Executor(model), x)
    assert state.plan is None and state.weights["0"].prepared is None
    assert not model[0]._forward_hooks


def test_stacked_loras_keep_independent_caches_and_leave_base_unchanged(lora_node):
    n = lora_node
    t = n.torch
    model, first, x = small_model(n)
    a, b = first.weights["0"].original
    second = n._LoraState({"0": (a, b * 0.5)})
    base = model(x)
    f = t.nn.functional
    expected = base + f.linear(f.linear(x, a.float()), b.float())
    expected = expected + f.linear(f.linear(x, a.float()), (b * 0.5).float())
    class StackedExecutor:
        class_obj = model
        def __call__(self, value):
            return n._run_with_lora(second, Executor(model), value)
    result = n._run_sampling(first, lambda: n._run_sampling(second, lambda: n._run_with_lora(first, StackedExecutor(), x)))
    t.testing.assert_close(result, expected, rtol=0, atol=0)
    assert first.weights["0"].prepared is second.weights["0"].prepared is None
    t.testing.assert_close(model(x), base, rtol=0, atol=0)


def test_cuda_sampling_releases_allocated_adapter_tensors(lora_node):
    n = lora_node
    t = n.torch
    if not t.cuda.is_available():
        pytest.skip("GPU CUDA necessária para medir a liberação")
    a, b = t.ones(32, 4096, dtype=t.bfloat16), t.ones(4096, 32, dtype=t.bfloat16)
    state = n._LoraState({"layer": (a, b)})
    x = t.empty(1, 4096, device="cuda", dtype=t.bfloat16)
    baseline = t.cuda.memory_allocated()
    def sample():
        state.weights["layer"].for_input(x)
        assert t.cuda.memory_allocated() > baseline
    n._run_sampling(state, sample)
    assert t.cuda.memory_allocated() == baseline
    assert state.weights["layer"].prepared is None


def test_invalid_target_does_not_leave_partial_hooks(lora_node):
    n = lora_node
    model, state, x = small_model(n)
    state.weights["missing.layer"] = n._LoraWeights(*state.weights["0"].original)
    with pytest.raises(AttributeError):
        n._run_sampling(state, lambda: n._run_with_lora(state, Executor(model), x))
    assert all(not m._forward_hooks for m in model.modules())
    assert state.plan is None


def test_dtype_changes_are_rebuilt_from_original_weights(lora_node):
    n = lora_node
    t = n.torch
    a, b = t.tensor([[1.0001, 1.0002]], dtype=t.float32), t.tensor([[1.0003]], dtype=t.float32)
    weights = n._LoraWeights(a, b)
    x = t.ones(1, 2, dtype=t.float16)
    n._lora_fwd(x, weights)
    first = weights.prepared
    n._lora_fwd(x, weights)
    assert weights.prepared is first
    exact = n._lora_fwd(x.double(), weights)
    expected = t.nn.functional.linear(t.nn.functional.linear(x.double(), a.double()), b.double())
    t.testing.assert_close(exact, expected, rtol=0, atol=0)


@pytest.mark.parametrize("dtype_name", ["float16", "bfloat16", "float32"])
def test_cuda_preserves_numerics_without_widening_resident_weights(lora_node, dtype_name):
    n = lora_node
    t = n.torch
    if not t.cuda.is_available():
        pytest.skip("GPU CUDA necessária para medir a cópia residente")
    if dtype_name == "bfloat16" and not t.cuda.is_bf16_supported():
        pytest.skip("GPU sem BF16")
    t.manual_seed(4)
    a, b = t.randn(2, 4, dtype=t.bfloat16), t.randn(4, 2, dtype=t.bfloat16)
    weights = n._LoraWeights(a, b)
    x = t.randn(3, 4, device="cuda", dtype=getattr(t, dtype_name))
    old_a, old_b = a.to(x.device), b.to(x.device)
    expected = t.nn.functional.linear(t.nn.functional.linear(x, old_a.to(x.dtype)), old_b.to(x.dtype))
    first = n._lora_fwd(x, weights)
    prepared = weights.prepared
    second = n._lora_fwd(x, weights)
    assert weights.prepared is prepared
    assert sum(w.numel() * w.element_size() for w in prepared) <= sum(w.numel() * w.element_size() for w in (a, b))
    assert all(w.device.type == "cpu" for w in weights.original)
    t.testing.assert_close(first, expected, rtol=0, atol=0)
    t.testing.assert_close(second, expected, rtol=0, atol=0)
    weights.clear()
    assert weights.prepared is None


def fused_mlp(node):
    t = node.torch
    class FusedMLP(t.nn.Module):
        fused = True
        def __init__(self):
            super().__init__()
            self.gate_up = t.nn.Linear(4, 12, bias=False)
            self.out = t.nn.Linear(6, 4, bias=False)

        def forward(self, x):
            gate, up = self.gate_up(x).chunk(2, -1)
            # Como no kernel fundido, não chama self.out.forward nem seus hooks.
            return t.nn.functional.linear(t.nn.functional.silu(gate) * up, self.out.weight)

    t.manual_seed(3)
    model = t.nn.ModuleDict({"mlp": FusedMLP()})
    pairs = {"mlp." + name: (t.randn(2, ins), t.randn(outs, 2)) for name, ins, outs in [("gate_layer", 4, 6), ("proj", 4, 6), ("out", 6, 4)]}
    return model, node._LoraState(pairs), t.randn(3, 4)


def test_fused_mlp_keeps_gate_up_and_down_updates(lora_node):
    n = lora_node
    t = n.torch
    model, state, x = fused_mlp(n)
    mlp = model["mlp"]
    f = t.nn.functional
    def delta(value, name):
        a, b = state.weights["mlp." + name].original
        return f.linear(f.linear(value, a), b)
    gu = mlp.gate_up(x) + t.cat([delta(x, "gate_layer"), delta(x, "proj")], -1)
    gate, up = gu.chunk(2, -1)
    hidden = f.silu(gate) * up
    expected = f.linear(hidden, mlp.out.weight) + delta(hidden, "out")
    class MLPExecutor:
        class_obj = model
        def __call__(self, value):
            return mlp(value)
    result = n._run_sampling(state, lambda: n._run_with_lora(state, MLPExecutor(), x))
    t.testing.assert_close(result, expected, rtol=0, atol=0)
    assert all(not m._forward_hooks for m in model.modules())


def test_partial_fused_hook_registration_is_cleaned(lora_node, monkeypatch):
    n = lora_node
    model, state, x = fused_mlp(n)
    monkeypatch.setattr(model["mlp"], "register_forward_hook", Mock(side_effect=RuntimeError("registration")))
    with pytest.raises(RuntimeError):
        n._run_sampling(state, lambda: n._run_with_lora(state, Executor(model), x))
    assert all(not m._forward_hooks for m in model.modules())


@pytest.mark.parametrize("suffix", [".lora_A.default.weight", ".lora_A.weight"])
def test_loader_preserves_scaling_and_registers_sampling_cleanup(lora_node, suffix):
    n = lora_node
    t = n.torch
    model, _, x = small_model(n)
    a, b = t.ones(2, 4), t.ones(4, 2)
    sd = {"transformer.0" + suffix: a, "transformer.0" + suffix.replace("lora_A", "lora_B"): b}
    n.utils.load_torch_file.return_value = (sd, {"lora_adapter_metadata": '{"transformer.lora_alpha": 4, "transformer.r": 2}'})
    patcher = Mock()
    clone = patcher.clone.return_value
    result, = n.SPEAddonLora().load(patcher, "addon.safetensors", 0.5)
    assert result is clone
    wrappers = {call.args[0]: call.args[2] for call in clone.add_wrapper_with_key.call_args_list}
    assert set(wrappers) == {"diffusion", "sample"}
    base = model(x)
    expected = base + t.nn.functional.linear(t.nn.functional.linear(x, a), b)
    result = wrappers["sample"](lambda: wrappers["diffusion"](Executor(model), x))
    t.testing.assert_close(result, expected, rtol=0, atol=0)
    t.testing.assert_close(model(x), base, rtol=0, atol=0)


def test_zero_strength_does_not_load_or_patch_model(lora_node):
    model = Mock()
    assert lora_node.SPEAddonLora().load(model, "addon.safetensors", 0) == (model,)
    lora_node.utils.load_torch_file.assert_not_called()
    model.clone.assert_not_called()
