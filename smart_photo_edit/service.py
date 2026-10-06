"""Executa uma edição: envia a imagem ao ComfyUI, roda o workflow N vezes e devolve eventos de progresso."""
from __future__ import annotations

import asyncio
import secrets
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncIterator

from . import paths
from .comfy import Cancelled, ComfyClient, ComfyError
from .workflows import Workflow, WorkflowError

MAX_SEED = 2**53 - 1
MAX_VARIATIONS = 4
MAX_PROMPT = 2000


@dataclass
class EditRequest:
    image: bytes
    filename: str
    content_type: str
    prompt: str
    negative: str = ""
    seed: int | None = None
    n: int = 1
    strength: float | None = None
    params: dict[str, Any] | None = None

    def validate(self) -> None:
        if not self.prompt.strip():
            raise WorkflowError("Descreva a edição antes de gerar.")
        if len(self.prompt) > MAX_PROMPT:
            raise WorkflowError(f"O prompt passou de {MAX_PROMPT} caracteres.")
        if not 1 <= self.n <= MAX_VARIATIONS:
            raise WorkflowError(f"Escolha entre 1 e {MAX_VARIATIONS} variações.")
        if self.seed is not None and not 0 <= self.seed <= MAX_SEED:
            raise WorkflowError("Semente inválida.")


def sniff_image(data: bytes) -> tuple[str, str] | None:
    """(mime, extensão) pelos primeiros bytes; None se não for PNG/JPEG/WebP."""
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png", "png"
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg", "jpg"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp", "webp"
    return None


class ResultStore:
    """Guarda as imagens geradas em disco (e apaga as antigas)."""

    def __init__(self, folder: Path | None = None, max_age_s: float = 24 * 3600):
        self.folder = folder or paths.results_dir()
        self.max_age_s = max_age_s
        self.folder.mkdir(parents=True, exist_ok=True)
        self.purge()

    def purge(self) -> None:
        cutoff = time.time() - self.max_age_s
        for f in self.folder.glob("*"):
            try:
                if f.is_file() and f.stat().st_mtime < cutoff:
                    f.unlink()
            except OSError:
                pass

    def save(self, data: bytes) -> str:
        kind = sniff_image(data)
        ext = kind[1] if kind else "png"
        rid = f"{secrets.token_hex(12)}.{ext}"
        (self.folder / rid).write_bytes(data)
        return rid

    def path(self, rid: str) -> Path | None:
        if "/" in rid or "\\" in rid or rid.startswith("."):
            return None
        p = self.folder / rid
        return p if p.is_file() else None


async def run_edit(
    client: ComfyClient,
    wf: Workflow,
    chosen_params: dict[str, Any] | None,
    req: EditRequest,
    store: ResultStore,
    cancel_event: asyncio.Event,
) -> AsyncIterator[dict[str, Any]]:
    """Gera eventos: {"type": "progress"|"done"|"error"|"cancelled", ...}."""
    try:
        req.validate()
        wf.param_values(chosen_params)  # falha cedo se algum valor salvo ficou inválido
        yield {"type": "progress", "p": 0.0, "phase": "Enviando imagem"}
        image_name = await client.upload_image(req.image, req.filename, req.content_type)
        base_seed = req.seed if req.seed is not None else secrets.randbelow(10**8)
        results: list[dict[str, Any]] = []
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()

        for i in range(req.n):
            seed = (base_seed + i) % (MAX_SEED + 1)
            graph = wf.apply(
                image_name=image_name,
                prompt=req.prompt.strip(),
                negative=req.negative.strip(),
                seed=seed,
                strength=req.strength,
                params=chosen_params,
            )
            label = f"Variação {i + 1} de {req.n} · " if req.n > 1 else ""

            def on_progress(ev: dict[str, Any], i: int = i, label: str = label) -> None:
                queue.put_nowait({"type": "progress", "p": (i + ev["p"]) / req.n, "phase": label + ev["phase"]})

            task = asyncio.create_task(client.run(graph, on_progress, cancel_event))
            try:
                while not task.done():
                    try:
                        yield await asyncio.wait_for(queue.get(), timeout=0.25)
                    except asyncio.TimeoutError:
                        pass
                while not queue.empty():
                    yield queue.get_nowait()
                images = await task  # propaga ComfyError / Cancelled
            finally:
                if not task.done():
                    cancel_event.set()
                    task.cancel()
            data = await client.fetch_image(images[0])
            rid = store.save(data)
            results.append({"url": f"/api/results/{rid}", "seed": seed})
        yield {"type": "done", "images": results}
    except Cancelled:
        yield {"type": "cancelled"}
    except (ComfyError, WorkflowError) as exc:
        yield {"type": "error", "message": str(exc), "code": getattr(exc, "code", "invalid_request")}
