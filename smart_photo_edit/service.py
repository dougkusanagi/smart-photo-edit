"""Executa uma edição: envia a imagem ao ComfyUI, roda o workflow N vezes e devolve eventos de progresso."""
from __future__ import annotations

import asyncio
import copy
import secrets
import hashlib
import time
from dataclasses import dataclass
from typing import Any, AsyncIterator

from .history import ResultStore
from . import background, upscale
from .prompt_limits import effective_prompt, validate_prompt
from .comfy import Cancelled, ComfyClient, ComfyError
from .workflows import Workflow, WorkflowError

MAX_SEED = 2**53 - 1
MAX_VARIATIONS = 4


@dataclass
class EditRequest:
    image: bytes
    filename: str
    content_type: str
    prompt: str
    reference: bytes | None = None
    reference_filename: str = 'referencia'
    source_name: str = "Sem título"
    source_result: str | None = None
    negative: str = ""
    seed: int | None = None
    n: int = 1
    strength: float | None = None
    params: dict[str, Any] | None = None
    addons: tuple[str, ...] = ()
    mode: str = 'edit'
    prompt_prefix: str = ''  # texto de apoio das LoRAs adicionais; enviado antes do prompt do usuário

    def validate(self) -> None:
        try:
            validate_prompt(self.prompt, self.prompt_prefix)
            validate_prompt(self.negative)
        except ValueError as exc:
            raise WorkflowError(str(exc)) from exc
        if self.mode != upscale.MODE and not self.prompt.strip() and not self.prompt_prefix.strip():
            raise WorkflowError("Descreva a edição antes de gerar.")
        if self.reference is not None and not sniff_image(self.reference):
            raise WorkflowError('Referência inválida: use PNG, JPG ou WebP.')
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
        reference_name = None
        sent_prompt = effective_prompt(req.prompt, req.prompt_prefix)
        if req.reference:
            if not wf.supports['reference']:
                raise WorkflowError('Este workflow não suporta uma segunda referência.')
            mime, ext = sniff_image(req.reference)
            reference_name = await client.upload_image(req.reference, 'spe_reference.' + ext, mime)
        base_seed = req.seed if req.seed is not None else secrets.randbelow(10**8)
        results: list[dict[str, Any]] = []
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()

        for i in range(req.n):
            seed = (base_seed + i) % (MAX_SEED + 1)
            graph = wf.apply(
                image_name=image_name,
                prompt=sent_prompt,
                reference_name=reference_name,
                negative=req.negative.strip(),
                seed=seed,
                strength=req.strength,
                params=chosen_params,
            )
            label = f"Variação {i + 1} de {req.n} · " if req.n > 1 else ""

            def on_progress(ev: dict[str, Any], i: int = i, label: str = label) -> None:
                queue.put_nowait({"type": "progress", "p": (i + ev["p"]) / req.n, "phase": label + ev["phase"]})

            execution_started = time.monotonic()
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
            if req.mode == background.MODE:
                try:
                    data = await asyncio.to_thread(background.apply_alpha, req.image, data)
                except (OSError, ValueError) as exc:
                    raise ComfyError('Não consegui aplicar a transparência à foto original: ' + str(exc), 'invalid_background_result') from exc
            if req.mode == upscale.MODE:
                try:
                    data = await asyncio.to_thread(upscale.finish, req.image, data, chosen_params['scale'],
                                            getattr(wf, 'upscale_model', upscale.DEFAULT_MODEL))
                except (OSError, ValueError, SyntaxError) as exc:
                    raise ComfyError('Não consegui finalizar o upscale: ' + str(exc), 'invalid_upscale_result') from exc
                if cancel_event.is_set():
                    raise Cancelled()
            metadata = {
                'duration_seconds': round(time.monotonic() - execution_started, 3),
                'prompt': sent_prompt, 'user_prompt': req.prompt.strip(),
                'reference': {'tag': '<image2>', 'filename': req.reference_filename[:120], 'sha256': hashlib.sha256(req.reference).hexdigest()} if req.reference else None, 'negative': req.negative.strip() if wf.bindings.get('negative') else '', 'seed': seed,
                'strength': req.strength if wf.bindings.get('strength') else None,
                'workflow': {'id': wf.id, 'name': wf.name}, 'params': wf.param_values(chosen_params),
                'models': [{key: model[key] for key in ('folder', 'filename', 'label', 'format', 'sha256', 'size_bytes', 'role', 'family') if key in model} for model in wf.requires.get('models', [])],
                'model_choices': copy.deepcopy(getattr(wf, 'model_selection', {})),
                'runtime_args': list(getattr(wf, 'runtime_args', [])),
                'variation': i + 1, 'variations': req.n,
                'name': req.source_name[:60], 'source_result': req.source_result, 'mode': req.mode,
                'addons': [{'id': m['addon'], 'filename': m['filename'], 'sha256': m.get('sha256')} for m in wf.requires.get('models', []) if m.get('addon')],
            }
            if req.mode == upscale.MODE:
                model = getattr(wf, 'upscale_model', upscale.DEFAULT_MODEL)
                scale = chosen_params['scale']
                common = {'algorithm': model, 'scale': scale, 'method': 'one_step_diffusion', 'steps': 1,
                          'max_output_pixels': upscale.max_pixels(model), 'node_cache': 'disabled'}
                if model == 'seedvr2':
                    metadata['upscale'] = {**common, 'precision': 'UNet FP8 (e4m3fn) / VAE FP16',
                                           'cfg': 1, 'sampler': 'euler', 'scheduler': 'simple',
                                           'color_correction': 'lab', 'vae_tile': 512, 'vae_tile_overlap': 128,
                                           'resize': 'lanczos_before_restoration', 'native_nodes': True}
                else:
                    metadata['upscale'] = {
                        **common, 'precision': 'UNet FP32 / VAE auto' if model == 'sinsr' else 'FP32',
                        'assembly_device': 'cpu', 'noise': 'global_cpu' if model == 'sinsr' else 'none', 'native_scale': 4,
                        'downsample_from_4x': scale == 2, 'cpu_threads_limit': 4,
                        'tile_size': 128 if model == 'sinsr' else 192, 'tile_pad': 32, 'tile_overlap': 32, 'tile_units': 'input_pixels',
                        'tile_size_is_initial': True, 'oom_retry_min_tile': 64,
                    }
                    if model == 'sinsr':
                        metadata['upscale'].update(vae_attention='query_chunks', attention_max_score_bytes=32*1024**2,
                                                   vae_device='auto', vae_device_policy='cuda_then_smaller_tiles_then_cpu',
                                                   vae_precision_policy='fp16_cuda_with_fp32_fallback')
                    else:
                        metadata['upscale'].update(color_alignment='adain_global', deterministic=True,
                                                   text_encoder='none', vae_device='same_as_model')
            try:
                rid = await asyncio.to_thread(store.save, data, metadata, req.image)
            except (OSError, ValueError) as exc:
                raise ComfyError('Não consegui salvar a edição no histórico: ' + str(exc), 'result_save_failed') from exc
            results.append(await asyncio.to_thread(store.record, rid))
        yield {"type": "done", "images": results}
    except Cancelled:
        yield {"type": "cancelled"}
    except (ComfyError, WorkflowError) as exc:
        yield {"type": "error", "message": str(exc), "code": getattr(exc, "code", "invalid_request")}
