import asyncio
import json

import pytest
from aiohttp import FormData
from aiohttp.test_utils import TestClient, TestServer

from smart_photo_edit import config, workflows
from smart_photo_edit.server import AppState, create_app
from smart_photo_edit.service import ResultStore

from .fake_comfy import FakeComfy


class Env:
    def __init__(self, app_client: TestClient, comfy: FakeComfy, state: AppState):
        self.http, self.comfy, self.state = app_client, comfy, state


@pytest.fixture
async def make_env(tmp_path):
    started = []

    async def factory(**comfy_kwargs) -> Env:
        comfy = FakeComfy(**comfy_kwargs)
        comfy_srv = TestServer(comfy.app)
        await comfy_srv.start_server()
        settings = config.Settings(comfy_url=f"http://127.0.0.1:{comfy_srv.port}")
        state = AppState(settings=settings, store=ResultStore(tmp_path / "results"), config_path=tmp_path / "cfg.json")
        client = TestClient(TestServer(create_app(state)))
        await client.start_server()
        started.extend([client, comfy_srv])
        return Env(client, comfy, state)

    yield factory
    for s in reversed(started):
        await s.close()


def edit_form(png: bytes, **options) -> FormData:
    form = FormData()
    form.add_field("image", png, filename="foto.png", content_type="image/png")
    form.add_field("options", json.dumps({"prompt": "deixe em preto e branco", **options}), content_type="application/json")
    return form


async def read_events(resp):
    return [json.loads(line) async for line in resp.content if line.strip()]


async def test_status_and_workflow_list(make_env):
    env = await make_env()
    st = await (await env.http.get("/api/status")).json()
    assert st["comfy"]["online"] and st["workflow"]["id"] == "qwen21-viggle-turbo"
    assert st["comfy"]["device"]["name"] == "Fake GPU"
    wf = await (await env.http.get("/api/workflows")).json()
    assert wf["active"] == "qwen21-viggle-turbo"
    ids = {w["id"]: w for w in wf["workflows"]}
    assert ids["qwen21-viggle-turbo"]["active"] and not ids["qwen21-base"]["active"]
    assert [p["key"] for p in ids["qwen21-base"]["params"]] == ["resolution", "steps"]


async def test_status_offline(make_env, tmp_path):
    state = AppState(settings=config.Settings(comfy_url="http://127.0.0.1:1"), store=ResultStore(tmp_path / "r"), config_path=tmp_path / "c.json")
    client = TestClient(TestServer(create_app(state)))
    await client.start_server()
    try:
        assert (await (await client.get("/api/status")).json())["comfy"]["online"] is False
        resp = await client.post("/api/edit", data=edit_form(b"\x89PNG\r\n\x1a\n" + b"0" * 20))
        events = await read_events(resp)
        assert events[-1]["type"] == "error" and events[-1]["code"] == "comfy_offline"
    finally:
        await client.close()


async def test_edit_end_to_end(make_env, png):
    env = await make_env()
    resp = await env.http.post("/api/edit", data=edit_form(png, seed=123, n=2))
    assert resp.status == 200
    events = await read_events(resp)
    kinds = [e["type"] for e in events]
    assert kinds[-1] == "done" and "progress" in kinds
    ps = [e["p"] for e in events if e["type"] == "progress"]
    assert ps == sorted(ps) and 0 <= ps[0] and ps[-1] <= 1
    images = events[-1]["images"]
    assert [i["seed"] for i in images] == [123, 124]
    got = await env.http.get(images[0]["url"])
    assert got.status == 200 and got.content_type == "image/png" and (await got.read()).startswith(b"\x89PNG")
    # o que chegou ao ComfyUI
    up = env.comfy.uploads[0]
    assert up["type"] == "temp" and up["bytes"] == png
    graphs = list(env.comfy.prompts.values())
    assert len(graphs) == 2
    assert graphs[0]["7"]["inputs"]["prompt"] == "deixe em preto e branco"
    assert graphs[0]["6"]["inputs"]["image"].endswith("[temp]")
    assert [g["9"]["inputs"]["noise_seed"] for g in graphs] == [123, 124]


async def test_edit_uses_saved_workflow_params(make_env, png):
    env = await make_env()
    r = await env.http.put("/api/settings", json={"workflow": "qwen21-base", "workflow_params": {"qwen21-base": {"steps": 12, "resolution": 768}}})
    assert r.status == 200
    events = await read_events(await env.http.post("/api/edit", data=edit_form(png)))
    assert events[-1]["type"] == "done"
    g = next(iter(env.comfy.prompts.values()))
    assert g["6"]["class_type"] == "KSampler" and g["6"]["inputs"]["steps"] == 12 and g["5"]["inputs"]["resolution"] == 768


async def test_settings_validation(make_env):
    env = await make_env()
    assert (await env.http.put("/api/settings", json={"workflow": "nao-existe"})).status == 404
    assert (await env.http.put("/api/settings", json={"workflow_params": {"qwen21-base": {"resolution": 5}}})).status == 422
    assert (await env.http.put("/api/settings", json={"comfy_url": "ftp://x"})).status == 400
    ok = await env.http.put("/api/settings", json={"comfy_url": "localhost:9000/"})
    assert (await ok.json())["comfy_url"] == "http://localhost:9000"
    assert json.loads(env.state.config_path.read_text())["comfy_url"] == "http://localhost:9000"


async def test_cancel_by_disconnect_interrupts_comfy(make_env, png):
    env = await make_env(duration=5)
    resp = await env.http.post("/api/edit", data=edit_form(png))
    async for line in resp.content:
        if b'"progress"' in line and b"Gerando" in line:
            break
    resp.close()
    for _ in range(40):
        if env.comfy.interrupted:
            break
        await asyncio.sleep(0.1)
    assert env.comfy.interrupted, "o ComfyUI deveria ter recebido /interrupt"


async def test_comfy_execution_error_is_friendly(make_env, png):
    env = await make_env(fail="exec")
    events = await read_events(await env.http.post("/api/edit", data=edit_form(png)))
    assert events[-1]["type"] == "error" and "memória" in events[-1]["message"]


async def test_invalid_workflow_error_names_the_node(make_env, png):
    env = await make_env(fail="invalid")
    events = await read_events(await env.http.post("/api/edit", data=edit_form(png)))
    assert events[-1]["code"] == "invalid_workflow" and "nó 7" in events[-1]["message"]


async def test_input_validation(make_env, png):
    env = await make_env()
    r = await env.http.post("/api/edit", data=edit_form(b"not an image"))
    assert r.status == 415
    events = await read_events(await env.http.post("/api/edit", data=edit_form(png, prompt="   ")))
    assert events[-1]["type"] == "error" and "Descreva" in events[-1]["message"]
    events = await read_events(await env.http.post("/api/edit", data=edit_form(png, n=9)))
    assert events[-1]["type"] == "error"


async def test_check_reports_missing_pieces(make_env):
    env = await make_env(available_classes={"UNETLoader", "CLIPLoader", "VAELoader", "LoadImage"},
                         files={("UNETLoader", "unet_name"): ["outro.safetensors"]})
    data = await (await env.http.get("/api/workflows/qwen21-viggle-turbo/check")).json()
    assert data["online"] and not data["ok"]
    assert {i["kind"] for i in data["issues"]} == {"missing_node", "missing_file"}


async def test_import_and_delete_workflow(make_env):
    env = await make_env()
    plain = {"1": {"class_type": "LoadImage", "inputs": {"image": "a.png"}}, "2": {"class_type": "CLIPTextEncode", "inputs": {"text": ""}}}
    r = await env.http.post("/api/workflows/import?filename=meu.json", data=json.dumps(plain))
    assert r.status == 201 and (await r.json())["id"] == "meu"
    bad = await env.http.post("/api/workflows/import", data=json.dumps({"nodes": [], "links": []}))
    assert bad.status == 422 and "API" in (await bad.json())["error"]
    assert (await env.http.put("/api/settings", json={"workflow": "meu"})).status == 200
    assert (await env.http.delete("/api/workflows/meu")).status == 200
    assert env.state.settings.workflow == config.DEFAULT_WORKFLOW
    assert (await env.http.delete("/api/workflows/qwen21-base")).status == 404


async def test_guard_blocks_foreign_host_and_origin(make_env):
    env = await make_env()
    assert (await env.http.get("/api/status", headers={"Host": "evil.example.com"})).status == 403
    r = await env.http.put("/api/settings", json={}, headers={"Origin": "http://evil.example.com"})
    assert r.status == 403
    assert (await env.http.get("/api/status")).status == 200


async def test_results_path_traversal_is_blocked(make_env):
    env = await make_env()
    assert (await env.http.get("/api/results/..%2Fconfig.json")).status == 404


async def test_index_served(make_env):
    env = await make_env()
    r = await env.http.get("/")
    assert r.status == 200 and "Smart Photo Edit" in await r.text()
