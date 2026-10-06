"""Servidor web local: serve a interface e expõe a API que conversa com o ComfyUI."""
from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlsplit

import aiohttp
from aiohttp import web

from . import __version__, config, paths, workflows
from .comfy import ComfyClient, ComfyError
from .launcher import ComfyLauncher
from .service import EditRequest, ResultStore, run_edit, sniff_image
from .workflows import WorkflowError

log = logging.getLogger("smart_photo_edit")
MAX_UPLOAD = 40 * 1024 * 1024
LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1"}


@dataclass
class AppState:
    settings: config.Settings
    store: ResultStore
    launcher: ComfyLauncher = field(default_factory=ComfyLauncher)
    allow_any_host: bool = False
    edit_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    config_path: Any = None

    def save(self) -> None:
        config.save(self.settings, self.config_path)


STATE_KEY = web.AppKey("state", AppState)
SESSION_KEY = web.AppKey("session", aiohttp.ClientSession)


def json_error(message: str, status: int = 400, code: str = "bad_request") -> web.Response:
    return web.json_response({"error": message, "code": code}, status=status)


@web.middleware
async def guard(request: web.Request, handler):
    """Bloqueia DNS rebinding e requisições cross-site contra a API local."""
    state: AppState = request.app[STATE_KEY]
    host = urlsplit("//" + request.headers.get("Host", "")).hostname or ""
    if not state.allow_any_host and host not in LOCAL_HOSTS:
        return json_error("Host não permitido.", 403, "forbidden")
    if request.method not in ("GET", "HEAD", "OPTIONS"):
        origin = request.headers.get("Origin")
        if origin and urlsplit(origin).netloc != request.headers.get("Host"):
            return json_error("Origem não permitida.", 403, "forbidden")
    return await handler(request)


def make_client(request: web.Request) -> ComfyClient:
    state: AppState = request.app[STATE_KEY]
    return ComfyClient(state.settings.comfy_url, request.app[SESSION_KEY])


def current_workflow(state: AppState) -> workflows.Workflow:
    wfs, _ = workflows.discover()
    wf = wfs.get(state.settings.workflow) or wfs.get(config.DEFAULT_WORKFLOW) or next(iter(wfs.values()), None)
    if wf is None:
        raise WorkflowError("Nenhum workflow disponível.")
    return wf


# ───────────────────────── rotas ─────────────────────────

async def index(request: web.Request) -> web.StreamResponse:
    return web.FileResponse(paths.WEB_DIR / "index.html", headers={"Cache-Control": "no-cache"})


async def status(request: web.Request) -> web.Response:
    state: AppState = request.app[STATE_KEY]
    stats = await make_client(request).system_stats()
    try:
        wf = current_workflow(state)
        wf_info = {"id": wf.id, "name": wf.name}
    except WorkflowError:
        wf_info = None
    device = None
    if stats and stats.get("devices"):
        d = stats["devices"][0]
        device = {"name": d.get("name"), "vram_total": d.get("vram_total"), "vram_free": d.get("vram_free")}
    return web.json_response({
        "version": __version__,
        "engine": "comfyui",
        "comfy": {
            "url": state.settings.comfy_url,
            "online": stats is not None,
            "launcher": state.launcher.state,
            "can_launch": bool(state.settings.comfy_command.strip()),
            "device": device,
        },
        "workflow": wf_info,
    })


async def list_workflows(request: web.Request) -> web.Response:
    state: AppState = request.app[STATE_KEY]
    wfs, errors = workflows.discover()
    active = current_workflow(state).id if wfs else None
    return web.json_response({
        "workflows": [w.to_public(active=(w.id == active)) for w in wfs.values()],
        "errors": errors,
        "active": active,
        "user_dir": str(paths.user_workflows_dir()),
        "values": state.settings.workflow_params,
    })


async def import_workflow(request: web.Request) -> web.Response:
    filename = request.query.get("filename", "")
    if request.content_type.startswith("multipart/"):
        reader = await request.multipart()
        field_ = await reader.next()
        if field_ is None or getattr(field_, "name", "") != "file":
            return json_error("Envie o arquivo no campo “file”.")
        filename = field_.filename or filename
        body = bytes(await field_.read(decode=False))
    else:
        body = await request.read()
    try:
        wf = workflows.import_text(body.decode("utf-8-sig"), filename)
    except UnicodeDecodeError:
        return json_error("O arquivo não está em UTF-8.")
    except WorkflowError as exc:
        return json_error(str(exc), 422, "invalid_workflow")
    return web.json_response(wf.to_public(), status=201)


async def delete_workflow(request: web.Request) -> web.Response:
    state: AppState = request.app[STATE_KEY]
    wf_id = request.match_info["id"]
    if not workflows.delete_user_workflow(wf_id):
        return json_error("Só é possível remover workflows importados.", 404, "not_found")
    if state.settings.workflow == wf_id:
        state.settings.workflow = config.DEFAULT_WORKFLOW
        state.save()
    return web.json_response({"ok": True})


async def check_workflow(request: web.Request) -> web.Response:
    wfs, _ = workflows.discover()
    wf = wfs.get(request.match_info["id"])
    if not wf:
        return json_error("Workflow não encontrado.", 404, "not_found")
    client = make_client(request)
    if await client.system_stats() is None:
        return web.json_response({"online": False, "ok": False, "issues": []})
    classes = sorted({n["class_type"] for n in wf.graph.values()})
    infos = dict(zip(classes, await asyncio.gather(*(client.object_info(c) for c in classes))))
    issues = workflows.validate(wf, infos)
    return web.json_response({"online": True, "ok": not issues, "issues": [i.to_public() for i in issues]})


async def get_settings(request: web.Request) -> web.Response:
    return web.json_response(request.app[STATE_KEY].settings.to_dict())


async def put_settings(request: web.Request) -> web.Response:
    state: AppState = request.app[STATE_KEY]
    try:
        body = await request.json()
    except ValueError:
        return json_error("JSON inválido.")
    if not isinstance(body, dict):
        return json_error("JSON inválido.")
    s = state.settings
    if "comfy_url" in body:
        url = config.normalize_url(str(body["comfy_url"]))
        if urlsplit(url).scheme not in ("http", "https") or not urlsplit(url).hostname:
            return json_error("Endereço do ComfyUI inválido (use algo como http://127.0.0.1:8188).")
        s.comfy_url = url
    for key in ("comfy_command", "comfy_cwd"):
        if key in body:
            setattr(s, key, str(body[key]).strip())
    if "comfy_autostart" in body:
        s.comfy_autostart = bool(body["comfy_autostart"])
    wfs, _ = workflows.discover()
    if "workflow" in body:
        if body["workflow"] not in wfs:
            return json_error("Workflow desconhecido.", 404, "not_found")
        s.workflow = body["workflow"]
    if "workflow_params" in body and isinstance(body["workflow_params"], dict):
        for wf_id, values in body["workflow_params"].items():
            if wf_id not in wfs or not isinstance(values, dict):
                continue
            try:
                wfs[wf_id].param_values(values)  # valida tudo antes de gravar
            except WorkflowError as exc:
                return json_error(str(exc), 422, "invalid_param")
            s.workflow_params[wf_id] = {**s.workflow_params.get(wf_id, {}), **values}
    state.save()
    return web.json_response(s.to_dict())


async def start_comfy(request: web.Request) -> web.Response:
    state: AppState = request.app[STATE_KEY]
    if not state.settings.comfy_command.strip():
        return json_error("Configure o comando para iniciar o ComfyUI nas configurações avançadas.", 400, "no_command")
    try:
        state.launcher.start(state.settings.comfy_command, state.settings.comfy_cwd)
    except ValueError as exc:
        return json_error(str(exc), 400, "launch_failed")
    return web.json_response({"ok": True, "launcher": state.launcher.state})


async def edit(request: web.Request) -> web.StreamResponse:
    state: AppState = request.app[STATE_KEY]
    reader = await request.multipart()
    image: bytes | None = None
    options: dict[str, Any] = {}
    while (part := await reader.next()) is not None:
        if part.name == "image":
            image = bytes(await part.read(decode=False))
        elif part.name == "options":
            try:
                options = json.loads((await part.read(decode=True)).decode("utf-8"))
            except ValueError:
                return json_error("Opções inválidas.")
    if not image:
        return json_error("Envie a imagem no campo “image”.")
    kind = sniff_image(image)
    if not kind:
        return json_error("Formato de imagem não suportado. Use PNG, JPG ou WebP.", 415, "unsupported_media")
    try:
        wf = current_workflow(state)
        req = EditRequest(
            image=image,
            filename=f"spe_{wf.id}.{kind[1]}",
            content_type=kind[0],
            prompt=str(options.get("prompt", "")),
            negative=str(options.get("negative", "")),
            seed=int(options["seed"]) if options.get("seed") not in (None, "") else None,
            n=int(options.get("n", 1)),
            strength=float(options["strength"]) if options.get("strength") is not None else None,
            params=state.settings.workflow_params.get(wf.id),
        )
    except (ValueError, TypeError) as exc:
        return json_error(f"Opções inválidas: {exc}")

    resp = web.StreamResponse(headers={"Content-Type": "application/x-ndjson; charset=utf-8", "Cache-Control": "no-store"})
    await resp.prepare(request)
    cancel = asyncio.Event()

    async def watch_disconnect() -> None:
        while True:
            await asyncio.sleep(0.3)
            if request.transport is None or request.transport.is_closing():
                cancel.set()
                return

    watcher = asyncio.create_task(watch_disconnect())

    async def send(ev: dict[str, Any]) -> None:
        try:
            await resp.write((json.dumps(ev, ensure_ascii=False) + "\n").encode("utf-8"))
        except (ConnectionResetError, aiohttp.ClientConnectionError):
            cancel.set()

    try:
        if state.edit_lock.locked():
            await send({"type": "progress", "p": 0, "phase": "Na fila"})
        async with state.edit_lock:  # a GPU atende uma edição por vez
            client = make_client(request)
            if await client.system_stats() is None:
                if state.settings.comfy_command.strip() and state.launcher.state != "running":
                    await send({"type": "progress", "p": 0, "phase": "Iniciando o ComfyUI"})
                    try:
                        state.launcher.start(state.settings.comfy_command, state.settings.comfy_cwd)
                    except ValueError as exc:
                        await send({"type": "error", "message": str(exc), "code": "launch_failed"})
                        return resp
                    for _ in range(240):  # até ~2 min
                        if cancel.is_set() or await client.system_stats() is not None:
                            break
                        await asyncio.sleep(0.5)
            async for ev in run_edit(client, wf, state.settings.workflow_params.get(wf.id), req, state.store, cancel):
                await send(ev)
    finally:
        watcher.cancel()
        await asyncio.gather(watcher, return_exceptions=True)
    return resp


async def get_result(request: web.Request) -> web.StreamResponse:
    p = request.app[STATE_KEY].store.path(request.match_info["rid"])
    if not p:
        return json_error("Resultado não encontrado.", 404, "not_found")
    return web.FileResponse(p, headers={"Cache-Control": "private, max-age=86400"})


# ───────────────────────── app ─────────────────────────

async def _on_startup(app: web.Application) -> None:
    app[SESSION_KEY] = aiohttp.ClientSession()
    state: AppState = app[STATE_KEY]
    if state.settings.comfy_autostart and state.settings.comfy_command.strip():
        if await ComfyClient(state.settings.comfy_url, app[SESSION_KEY]).system_stats() is None:
            try:
                state.launcher.start(state.settings.comfy_command, state.settings.comfy_cwd)
            except ValueError as exc:
                log.warning("Não foi possível iniciar o ComfyUI: %s", exc)


async def _on_cleanup(app: web.Application) -> None:
    await app[SESSION_KEY].close()
    app[STATE_KEY].launcher.stop()


def create_app(state: AppState) -> web.Application:
    app = web.Application(middlewares=[guard], client_max_size=MAX_UPLOAD)
    app[STATE_KEY] = state
    app.add_routes([
        web.get("/", index),
        web.get("/api/status", status),
        web.get("/api/workflows", list_workflows),
        web.post("/api/workflows/import", import_workflow),
        web.delete("/api/workflows/{id}", delete_workflow),
        web.get("/api/workflows/{id}/check", check_workflow),
        web.get("/api/settings", get_settings),
        web.put("/api/settings", put_settings),
        web.post("/api/comfy/start", start_comfy),
        web.post("/api/edit", edit),
        web.get("/api/results/{rid}", get_result),
    ])
    app.on_startup.append(_on_startup)
    app.on_cleanup.append(_on_cleanup)
    return app
