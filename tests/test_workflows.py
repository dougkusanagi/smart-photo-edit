import json

import pytest

from smart_photo_edit import paths, workflows
from smart_photo_edit.workflows import WorkflowError


def builtin(wf_id="qwen21-viggle-turbo"):
    wfs, errors = workflows.discover()
    assert not errors
    return wfs[wf_id]


def test_builtins_load_and_bind():
    wfs, errors = workflows.discover()
    assert errors == {}
    assert {"qwen21-viggle-turbo", "qwen21-base"} <= set(wfs)
    wf = wfs["qwen21-viggle-turbo"]
    assert set(wf.bindings) == {"image", "prompt", "seed"}
    assert wf.supports == {"negative": False, "strength": False, "seed": True}


def test_apply_injects_values_without_mutating_original():
    wf = builtin()
    g = wf.apply(image_name="a.png [temp]", prompt="make it night", seed=7, params={"resolution": 768})
    assert g["6"]["inputs"]["image"] == "a.png [temp]"
    assert g["7"]["inputs"]["prompt"] == "make it night"
    assert g["7"]["inputs"]["resolution"] == 768
    assert g["9"]["inputs"]["noise_seed"] == 7
    assert wf.graph["7"]["inputs"]["prompt"] == ""  # original intacto


def test_param_validation_and_number_clamp():
    base = builtin("qwen21-base")
    assert base.param_values({"steps": 999})["steps"] == 40
    assert base.param_values(None)["steps"] == 25
    with pytest.raises(WorkflowError):
        base.param_values({"resolution": 12345})
    with pytest.raises(WorkflowError):
        base.param_values({"steps": "abc"})


def test_guess_bindings_on_plain_api_graph():
    graph = {
        "1": {"class_type": "LoadImage", "inputs": {"image": "x.png"}},
        "2": {"class_type": "CLIPTextEncode", "inputs": {"text": "pos", "clip": ["9", 0]}},
        "3": {"class_type": "CLIPTextEncode", "inputs": {"text": "neg", "clip": ["9", 0]}},
        "4": {"class_type": "KSampler", "inputs": {"seed": 1, "positive": ["2", 0], "negative": ["3", 0]}},
    }
    wf = workflows.parse_workflow(graph, wf_id="x", source="user")
    assert wf.bindings["image"] == [("1", "image")]
    assert wf.bindings["prompt"] == [("2", "text")]
    assert wf.bindings["negative"] == [("3", "text")]
    assert wf.bindings["seed"] == [("4", "seed")]


def test_gui_format_is_rejected_with_helpful_message():
    with pytest.raises(WorkflowError, match="API"):
        workflows.parse_workflow({"nodes": [], "links": []}, wf_id="x", source="user")


def test_missing_prompt_or_image_binding():
    with pytest.raises(WorkflowError, match="imagem"):
        workflows.parse_workflow({"1": {"class_type": "KSampler", "inputs": {"seed": 1}}}, wf_id="x", source="user")


def test_binding_to_missing_node_fails():
    data = {"bindings": {"image": ["9", "image"], "prompt": ["1", "text"]},
            "prompt": {"1": {"class_type": "X", "inputs": {"text": "", "image": "a"}}}}
    with pytest.raises(WorkflowError, match="não existe"):
        workflows.parse_workflow(data, wf_id="x", source="user")


def test_import_creates_user_file_and_never_overwrites():
    text = json.dumps({
        "1": {"class_type": "LoadImage", "inputs": {"image": "x.png"}},
        "2": {"class_type": "CLIPTextEncode", "inputs": {"text": "p"}},
    })
    a = workflows.import_text(text, "meu fluxo.json")
    b = workflows.import_text(text, "meu fluxo.json")
    assert a.id == "meu-fluxo" and b.id == "meu-fluxo-2"
    wfs, errors = workflows.discover()
    assert errors == {} and wfs["meu-fluxo"].source == "user"
    assert workflows.delete_user_workflow("meu-fluxo") and not workflows.delete_user_workflow("qwen21-base")
    assert "meu-fluxo" not in workflows.discover()[0]


def test_broken_user_file_is_reported_not_fatal():
    paths.user_workflows_dir().mkdir(parents=True)
    (paths.user_workflows_dir() / "ruim.json").write_text("{ não é json", encoding="utf-8")
    wfs, errors = workflows.discover()
    assert "qwen21-base" in wfs and "ruim.json" in errors


def test_validate_reports_missing_nodes_and_files():
    wf = builtin()
    infos = {c: {"input": {"required": {}}} for c in {n["class_type"] for n in wf.graph.values()}}
    infos["ViggleTurboLora"] = {}  # nó ausente
    infos["UNETLoader"] = {"input": {"required": {"unet_name": [["outro.safetensors"]]}}}
    infos["LoadImage"] = {"input": {"required": {"image": [["nao_conta.png"]]}}}
    issues = workflows.validate(wf, infos)
    kinds = {(i.kind, i.class_type) for i in issues}
    assert ("missing_node", "ViggleTurboLora") in kinds
    assert ("missing_file", "UNETLoader") in kinds
    assert not any(i.class_type == "LoadImage" for i in issues)  # o campo de imagem é injetado em runtime
    assert "viggle_turbo.py" in next(i for i in issues if i.kind == "missing_node").message
