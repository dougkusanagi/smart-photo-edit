"""Cliente assíncrono da API HTTP/WebSocket do ComfyUI."""
from __future__ import annotations

import asyncio
import json
import uuid
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

import aiohttp

ProgressCb = Callable[[dict[str, Any]], Awaitable[None] | None]


class ComfyError(RuntimeError):
    """Falha ao falar com o ComfyUI ou ao executar o workflow (mensagem pronta para o usuário)."""

    def __init__(self, message: str, code: str = "comfy_error"):
        super().__init__(message)
        self.code = code


class ComfyOffline(ComfyError):
    def __init__(self, url: str):
        super().__init__(f"Não consegui conectar ao ComfyUI em {url}. Ele está rodando?", "comfy_offline")


class Cancelled(Exception):
    pass


@dataclass
class OutputImage:
    filename: str
    subfolder: str
    type: str


class ComfyClient:
    def __init__(self, base_url: str, session: aiohttp.ClientSession):
        self.base_url = base_url.rstrip("/")
        self.session = session
        self.client_id = uuid.uuid4().hex

    # ── básicos ──
    async def _get_json(self, path: str, timeout: float = 10) -> Any:
        try:
            async with self.session.get(self.base_url + path, timeout=aiohttp.ClientTimeout(total=timeout)) as r:
                if r.status >= 400:
                    raise ComfyError(f"O ComfyUI respondeu {r.status} em {path}.")
                return await r.json(content_type=None)
        except (aiohttp.ClientConnectionError, asyncio.TimeoutError):
            raise ComfyOffline(self.base_url) from None

    async def system_stats(self) -> dict[str, Any] | None:
        """None se o ComfyUI não responde."""
        try:
            return await self._get_json("/system_stats", timeout=2)
        except ComfyError:
            return None

    async def object_info(self, class_type: str) -> dict[str, Any]:
        """Definição de um nó; {} se o nó não existe."""
        data = await self._get_json(f"/object_info/{class_type}")
        return data.get(class_type, {}) if isinstance(data, dict) else {}

    # ── envio de imagem ──
    async def upload_image(self, data: bytes, filename: str, content_type: str = "image/png") -> str:
        """Envia para a pasta temporária do ComfyUI e devolve o valor para o campo `image` do LoadImage."""
        form = aiohttp.FormData()
        form.add_field("image", data, filename=filename, content_type=content_type)
        form.add_field("type", "temp")
        form.add_field("overwrite", "true")
        try:
            async with self.session.post(self.base_url + "/upload/image", data=form, timeout=aiohttp.ClientTimeout(total=60)) as r:
                if r.status >= 400:
                    raise ComfyError(f"O ComfyUI recusou a imagem ({r.status}).")
                info = await r.json(content_type=None)
        except (aiohttp.ClientConnectionError, asyncio.TimeoutError):
            raise ComfyOffline(self.base_url) from None
        name = info["name"]
        sub = info.get("subfolder") or ""
        return f"{sub}/{name} [temp]" if sub else f"{name} [temp]"

    # ── execução ──
    async def queue(self, graph: dict[str, Any]) -> str:
        payload = {"prompt": graph, "client_id": self.client_id}
        try:
            async with self.session.post(self.base_url + "/prompt", json=payload, timeout=aiohttp.ClientTimeout(total=30)) as r:
                body = await r.json(content_type=None)
                if r.status >= 400:
                    raise ComfyError(_format_queue_error(body), "invalid_workflow")
                return body["prompt_id"]
        except (aiohttp.ClientConnectionError, asyncio.TimeoutError):
            raise ComfyOffline(self.base_url) from None

    async def history(self, prompt_id: str) -> dict[str, Any] | None:
        data = await self._get_json(f"/history/{prompt_id}")
        return data.get(prompt_id) if isinstance(data, dict) else None

    async def cancel(self, prompt_id: str) -> None:
        """Cancela se estiver na fila ou rodando (nunca interrompe o trabalho de outra pessoa)."""
        try:
            q = await self._get_json("/queue", timeout=5)
            running = [item[1] for item in q.get("queue_running", [])]
            if prompt_id in running:
                async with self.session.post(
                    self.base_url + "/interrupt", json={"prompt_id": prompt_id}, timeout=aiohttp.ClientTimeout(total=5)
                ):
                    pass
            else:
                async with self.session.post(
                    self.base_url + "/queue", json={"delete": [prompt_id]}, timeout=aiohttp.ClientTimeout(total=5)
                ):
                    pass
        except ComfyError:
            pass

    async def fetch_image(self, img: OutputImage) -> bytes:
        params = {"filename": img.filename, "subfolder": img.subfolder, "type": img.type}
        try:
            async with self.session.get(self.base_url + "/view", params=params, timeout=aiohttp.ClientTimeout(total=60)) as r:
                if r.status >= 400:
                    raise ComfyError("Não consegui baixar a imagem gerada do ComfyUI.")
                return await r.read()
        except (aiohttp.ClientConnectionError, asyncio.TimeoutError):
            raise ComfyOffline(self.base_url) from None

    async def run(
        self,
        graph: dict[str, Any],
        on_progress: ProgressCb | None = None,
        cancel_event: asyncio.Event | None = None,
        poll_interval: float = 0.4,
    ) -> list[OutputImage]:
        """Enfileira o grafo e espera terminar. Progresso por WebSocket (opcional); término por /history."""
        total_nodes = max(1, len(graph))
        state = {"done": 0, "sampling": False, "p": 0.0, "phase": "Carregando modelos"}
        # Samplers contam passos e upscale conta blocos; encoders e VAE também emitem `progress`.
        samplers = {nid for nid, n in graph.items() if "Sampler" in n["class_type"] and n["class_type"] != "KSamplerSelect"}
        upscalers = {nid for nid, n in graph.items() if n['class_type'] == 'SPEOneStepUpscale'}
        seedvr = any(n['class_type'] == 'SeedVR2Conditioning' for n in graph.values())

        def node_phase(node_id: str) -> str:
            node = graph.get(node_id, {})
            if node.get('class_type') == 'SPEOneStepUpscale':
                return 'Reconstruindo detalhes por difusão'
            if node.get('class_type') in ('SeedVR2Preprocess', 'VAEEncodeTiled'):
                return 'Codificando a foto'
            if node.get('class_type') in ('SeedVR2Conditioning', 'KSampler') and seedvr:
                return 'Reconstruindo detalhes por difusão'
            if node.get('class_type') in ('VAEDecodeTiled', 'SeedVR2PostProcessing'):
                return 'Finalizando imagem'
            if node.get('class_type') == 'TextEncodeQwenImage21':
                clip = node['inputs'].get('clip', [])
                cpu = bool(clip) and graph.get(str(clip[0]), {}).get('inputs', {}).get('device') == 'cpu'
                return 'Preparando imagens e instrução' + (' na CPU' if cpu else '')
            if node.get('class_type') == 'VAEDecode':
                return 'Finalizando imagem'
            return 'Carregando modelos'

        async def emit(p: float, phase: str) -> None:
            state["p"] = max(state["p"], min(1.0, p))
            if on_progress:
                res = on_progress({"p": state["p"], "phase": phase})
                if asyncio.iscoroutine(res):
                    await res

        ws_task: asyncio.Task | None = None
        prompt_id_box: dict[str, str] = {}

        async def ws_listener() -> None:
            try:
                ws_url = self.base_url.replace("http", "ws", 1) + f"/ws?clientId={self.client_id}"
                async with self.session.ws_connect(ws_url, heartbeat=20) as ws:
                    async for msg in ws:
                        if msg.type != aiohttp.WSMsgType.TEXT:
                            continue
                        try:
                            ev = json.loads(msg.data)
                        except ValueError:
                            continue
                        data = ev.get("data") or {}
                        if data.get("prompt_id") not in (None, prompt_id_box.get("id")):
                            continue
                        kind = ev.get("type")
                        if kind in ("executing", "execution_cached"):
                            state["done"] += len(data.get("nodes", [])) if kind == "execution_cached" else (1 if data.get("node") else 0)
                            if kind == 'executing' and data.get('node'):
                                state['phase'] = node_phase(str(data['node']))
                            if state['phase'] == 'Finalizando imagem':
                                await emit(0.96, state['phase'])
                            elif not state["sampling"]:
                                await emit(0.18 * min(1.0, state["done"] / total_nodes), state['phase'])
                        elif kind == "progress":
                            value, mx = data.get("value", 0), max(1, data.get("max", 1))
                            if str(data.get('node')) in upscalers:
                                state['sampling'] = True
                                blocks = max(1, mx // 3)
                                block, step = min(blocks, value // 3 + 1), int(value) % 3
                                phase = ('Preparando', 'Reconstruindo', 'Finalizando')[step]
                                phase = f'{phase} · bloco {block}/{blocks}' if value < mx else f'Montando ampliação · {blocks}/{blocks} blocos'
                                await emit(0.18 + 0.74 * value / mx, phase)
                            elif str(data.get("node")) in samplers:
                                state["sampling"] = True
                                await emit(0.18 + 0.74 * value / mx, ('Reconstruindo detalhes' if seedvr else 'Gerando') + f" · passo {value}/{mx}")
                            elif not state["sampling"]:
                                await emit(0.18 * min(1.0, (state["done"] + value / mx) / total_nodes), node_phase(str(data.get('node'))))
            except (aiohttp.ClientError, asyncio.TimeoutError, OSError):
                return  # sem WebSocket: segue só com /history, só perde o progresso fino

        try:
            ws_task = asyncio.create_task(ws_listener())
            await asyncio.sleep(0.05)  # dá tempo de o WS conectar antes de enfileirar
            await emit(0.01, "Enviando")
            pid = await self.queue(graph)
            prompt_id_box["id"] = pid
            while True:
                if cancel_event and cancel_event.is_set():
                    await self.cancel(pid)
                    raise Cancelled()
                hist = await self.history(pid)
                if hist:
                    status = hist.get("status", {})
                    if status.get("status_str") == "error":
                        msg, code = _format_exec_error(status)
                        raise ComfyError(msg, code)
                    outputs = hist.get("outputs", {})
                    images = [
                        OutputImage(i["filename"], i.get("subfolder", ""), i.get("type", "output"))
                        for node in outputs.values()
                        for i in node.get("images", [])
                    ]
                    if images:
                        await emit(1.0, "Pronto")
                        return images
                    if status.get("completed"):
                        raise ComfyError("O workflow terminou sem gerar nenhuma imagem (falta um nó de saída?).", "no_output")
                await asyncio.sleep(poll_interval)
        except asyncio.CancelledError:
            # a requisição foi abortada (navegador fechou/recarregou): não deixa a GPU gastando à toa
            if prompt_id_box.get("id"):
                try:
                    await asyncio.wait_for(asyncio.shield(self.cancel(prompt_id_box["id"])), timeout=5)
                except (asyncio.TimeoutError, ComfyError):
                    pass
            raise
        finally:
            if ws_task:
                ws_task.cancel()
                await asyncio.gather(ws_task, return_exceptions=True)


def _format_queue_error(body: Any) -> str:
    if not isinstance(body, dict):
        return "O ComfyUI recusou o workflow."
    err = body.get("error")
    msg = err.get("message") if isinstance(err, dict) else (err if isinstance(err, str) else "")
    details = []
    for nid, ne in (body.get("node_errors") or {}).items():
        for e in ne.get("errors", []):
            details.append(f"nó {nid} ({ne.get('class_type', '?')}): {e.get('message', '')} {e.get('details', '')}".strip())
    text = msg or "O ComfyUI recusou o workflow."
    if details:
        text += " " + "; ".join(details[:3])
    return text


def _format_exec_error(status: dict[str, Any]) -> tuple[str, str]:
    """(mensagem, código) a partir do histórico de uma execução que falhou."""
    for kind, data in status.get("messages", []):
        if kind == "execution_error":
            msg = data.get("exception_message", "").strip() or "erro desconhecido"
            node = data.get("node_type") or data.get("node_id")
            low = msg.lower()
            if "out of memory" in low or "vram" in low or "allocat" in low:
                return (
                    "A GPU ficou sem memória. Feche outros programas que usam a placa de vídeo "
                    "ou escolha uma resolução menor nas configurações avançadas.",
                    "out_of_memory",
                )
            return f"Erro no nó {node}: {msg}", "execution_error"
    return "O ComfyUI falhou ao executar o workflow.", "execution_error"
