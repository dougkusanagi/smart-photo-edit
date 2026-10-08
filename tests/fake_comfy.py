"""ComfyUI falso, só com o necessário para testar o app de ponta a ponta."""
from __future__ import annotations

import asyncio
import io
import json
import uuid

from aiohttp import web
from PIL import Image, ImageOps

from .conftest import make_png


class FakeComfy:
    def __init__(self, *, available_classes=None, files=None, fail=False, duration=0.3):
        self.available = available_classes  # None = todos
        self.files = files or {}  # {(class, input): [opções]}
        self.fail = fail
        self.duration = duration
        self.prompts: dict[str, dict] = {}
        self.uploads: list[dict] = []
        self.interrupted: list[str] = []
        self.running: str | None = None
        self.finished: set[str] = set()
        self.app = web.Application(client_max_size=50 * 1024 * 1024)
        self.app.add_routes([
            web.get("/system_stats", self.system_stats),
            web.get("/object_info/{cls}", self.object_info),
            web.post("/upload/image", self.upload),
            web.post("/prompt", self.prompt),
            web.get("/history/{pid}", self.history),
            web.get("/view", self.view),
            web.get("/queue", self.queue),
            web.post("/queue", self.queue_post),
            web.post("/interrupt", self.interrupt),
            web.get("/ws", self.ws),
        ])
        self.sockets: list[web.WebSocketResponse] = []

    async def system_stats(self, request):
        return web.json_response({"devices": [{"name": "Fake GPU", "vram_total": 16 << 30, "vram_free": 8 << 30}]})

    async def object_info(self, request):
        cls = request.match_info["cls"]
        if self.available is not None and cls not in self.available:
            return web.json_response({})
        required = {}
        for (c, name), options in self.files.items():
            if c == cls:
                required[name] = [options]
        return web.json_response({cls: {"input": {"required": required}}})

    async def upload(self, request):
        reader = await request.multipart()
        rec: dict = {}
        while (part := await reader.next()) is not None:
            if part.name == "image":
                rec["filename"] = part.filename
                rec["bytes"] = bytes(await part.read())
            else:
                rec[part.name] = (await part.read()).decode()
        self.uploads.append(rec)
        return web.json_response({"name": rec["filename"], "subfolder": "", "type": rec.get("type", "input")})

    async def prompt(self, request):
        body = await request.json()
        if self.fail == "invalid":
            return web.json_response(
                {"error": {"message": "Prompt outputs failed validation"},
                 "node_errors": {"7": {"class_type": "TextEncodeQwenImage21", "errors": [{"message": "Bad", "details": "resolution"}]}}},
                status=400)
        pid = uuid.uuid4().hex
        self.prompts[pid] = body["prompt"]
        asyncio.get_event_loop().create_task(self._execute(pid))
        return web.json_response({"prompt_id": pid, "number": 1})

    async def _execute(self, pid):
        self.running = pid
        graph = self.prompts[pid]
        encoder = next((nid for nid, n in graph.items() if n['class_type'] == 'TextEncodeQwenImage21'), None)
        decoder = next((nid for nid, n in graph.items() if n['class_type'] == 'VAEDecode'), None)
        upscaler = next((nid for nid, n in graph.items() if n['class_type'] in ('SPEOneStepUpscale', 'KSampler')), None)
        for ws in self.sockets:  # codificador de texto também emite "progress" no ComfyUI real
            if encoder:
                await ws.send_json({"type": "executing", "data": {"prompt_id": pid, "node": encoder}})
            await ws.send_json({"type": "progress", "data": {"value": 5, "max": 12, "prompt_id": pid, "node": encoder or "7"}})
        for step in range(1, 4):
            for ws in self.sockets:
                await ws.send_json({"type": "progress", "data": {"value": step, "max": 3, "prompt_id": pid, "node": upscaler or "12"}})
            await asyncio.sleep(self.duration / 3)
            if pid in self.interrupted:
                break
        if decoder and pid not in self.interrupted:
            for ws in self.sockets:
                await ws.send_json({"type": "executing", "data": {"prompt_id": pid, "node": decoder}})
        self.running = None
        self.finished.add(pid)

    async def history(self, request):
        pid = request.match_info["pid"]
        if pid not in self.finished:
            return web.json_response({})
        if pid in self.interrupted:
            return web.json_response({pid: {"status": {"status_str": "error", "completed": False, "messages": [["execution_interrupted", {}]]}, "outputs": {}}})
        if self.fail == "exec":
            return web.json_response({pid: {"status": {"status_str": "error", "completed": False, "messages": [
                ["execution_error", {"node_type": "UNETLoader", "exception_message": "Allocation on device out of memory"}]]}, "outputs": {}}})
        return web.json_response({pid: {"status": {"status_str": "success", "completed": True, "messages": []},
                                        "outputs": {"14": {"images": [{"filename": f"{pid}.png", "subfolder": "", "type": "temp"}]}}}})

    async def view(self, request):
        graph = self.prompts.get(request.query.get('filename', '').removesuffix('.png'), {})
        upscaler = next((n for n in graph.values() if n['class_type'] in ('SPEOneStepUpscale', 'ImageScaleBy')), None)
        if upscaler:
            input_node = graph[upscaler['inputs']['image'][0]]
            filename = input_node['inputs']['image'].removesuffix(' [temp]')
            uploaded = next(u for u in reversed(self.uploads) if u['filename'] == filename)
            with Image.open(io.BytesIO(uploaded['bytes'])) as image:
                image = ImageOps.exif_transpose(image).convert('RGB')
                scale = upscaler['inputs'].get('scale', upscaler['inputs'].get('scale_by'))
                image = image.resize((image.width * scale, image.height * scale))
                buf = io.BytesIO()
                image.save(buf, 'PNG')
            return web.Response(body=buf.getvalue(), content_type='image/png')
        if any(node['class_type'] == 'RemoveBackground' for node in graph.values()):
            buf = io.BytesIO()
            Image.new('RGBA', (32, 24), (10, 200, 10, 128)).save(buf, 'PNG')
            return web.Response(body=buf.getvalue(), content_type='image/png')
        return web.Response(body=make_png((10, 200, 10), (32, 24)), content_type="image/png")

    async def queue(self, request):
        running = [[0, self.running, {}]] if self.running else []
        return web.json_response({"queue_running": running, "queue_pending": []})

    async def queue_post(self, request):
        return web.json_response({})

    async def interrupt(self, request):
        body = await request.json()
        self.interrupted.append(body.get("prompt_id") or self.running)
        return web.json_response({})

    async def ws(self, request):
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        self.sockets.append(ws)
        try:
            async for _ in ws:
                pass
        finally:
            self.sockets.remove(ws)
        return ws
