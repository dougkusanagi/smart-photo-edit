"""Linha de comando: `python -m smart_photo_edit [serve|setup|check]`."""
from __future__ import annotations

import argparse
import asyncio
import sys
import threading
import webbrowser
from pathlib import Path

from aiohttp import web

from . import __version__, config, paths, workflows, models
from .installer import install_requirements
from .server import AppState, create_app
from .service import ResultStore


def _serve(args: argparse.Namespace) -> int:
    settings = config.load()
    if args.comfy_url:
        settings.comfy_url = config.normalize_url(args.comfy_url)
        settings.engine_mode = "external"
    wildcard = args.host in ("0.0.0.0", "::")
    state = AppState(settings=settings, store=ResultStore(), allow_any_host=wildcard)
    url = f"http://{'localhost' if wildcard else args.host}:{args.port}/"
    engine = "local gerenciado pelo app" if settings.engine_mode == "managed" else settings.comfy_url
    print(f"Smart Photo Edit {__version__}\n  Interface: {url}\n  Motor:     {engine}\n  Dados:     {paths.user_data_dir()}")
    if wildcard:
        print("  ⚠ Escutando em todas as interfaces: qualquer pessoa na rede poderá usar sua GPU.")
    if not args.no_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        web.run_app(create_app(state), host=args.host, port=args.port, print=None)
    except OSError as exc:
        print(f"\nNão consegui abrir {args.host}:{args.port} ({exc.strerror}). Outra instância já está rodando? Tente --port {args.port + 1}.", file=sys.stderr)
        return 1
    return 0


def _setup(args: argparse.Namespace) -> int:
    wfs, _ = workflows.discover()
    wf = wfs.get(args.workflow)
    if not wf:
        print(f"Workflow desconhecido: {args.workflow}. Disponíveis: {', '.join(wfs)}", file=sys.stderr)
        return 2
    wf = models.configure(wf, config.load().model_choices.get(wf.id))
    if not args.comfyui_dir:
        from .runtime import ManagedRuntime
        async def prepare():
            runtime = ManagedRuntime()
            try:
                await runtime.start(wf)
            finally:
                await runtime.close()
        asyncio.run(prepare())
        print("Motor local e modelos preparados.")
        return 0
    comfy_dir = Path(args.comfyui_dir).expanduser()
    if not (comfy_dir / "custom_nodes").is_dir():
        print(f"Não parece uma pasta do ComfyUI (sem custom_nodes/): {comfy_dir}", file=sys.stderr)
        return 2
    models = Path(args.models_dir).expanduser() if args.models_dir else None
    print(f"Instalando requisitos de “{wf.name}”…")
    for note in install_requirements(wf, comfy_dir, models):
        print("  •", note)
    print("Pronto.")
    return 0


def _check(args: argparse.Namespace) -> int:
    from .comfy import ComfyClient
    import aiohttp

    settings = config.load()
    wfs, _ = workflows.discover()
    wf = wfs.get(args.workflow or settings.workflow)
    if not wf:
        print("Workflow desconhecido.", file=sys.stderr)
        return 2

    wf = models.configure(wf, settings.model_choices.get(wf.id))

    async def go() -> int:
        async with aiohttp.ClientSession() as session:
            from .runtime import ManagedRuntime
            runtime = ManagedRuntime()
            if settings.engine_mode == "managed" and not args.comfy_url:
                async def emit(ev):
                    print(ev["phase"])
                try:
                    await runtime.ensure(wf, session, asyncio.Event(), emit)
                    return await validate(ComfyClient(runtime.url, session))
                finally:
                    await runtime.close()
            return await validate(ComfyClient(args.comfy_url or settings.comfy_url, session))

    async def validate(client):
        if await client.system_stats() is None:
            print(f"ComfyUI offline em {client.base_url}")
            return 1
        classes = sorted({n["class_type"] for n in wf.graph.values()})
        infos = dict(zip(classes, await asyncio.gather(*(client.object_info(c) for c in classes))))
        issues = workflows.validate(wf, infos)
        for i in issues:
            print("✗", i.message)
        print("✓ Tudo certo." if not issues else f"{len(issues)} problema(s).")
        return 1 if issues else 0

    return asyncio.run(go())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="smart-photo-edit", description="Editor de imagens com IA (ComfyUI).")
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="cmd")

    def add_serve(p: argparse.ArgumentParser) -> None:
        p.add_argument("--host", default="127.0.0.1")
        p.add_argument("--port", type=int, default=8765)
        p.add_argument("--comfy-url", help="Ativa modo externo opcional neste endereço")
        p.add_argument("--no-browser", action="store_true")

    add_serve(sub.add_parser("serve", help="Inicia o app (padrão)"))
    add_serve(parser)
    ps = sub.add_parser("setup", help="Prepara o motor local e os modelos de um workflow")
    ps.add_argument("--comfyui-dir", help="Pasta do ComfyUI (a que contém custom_nodes/)")
    ps.add_argument("--models-dir", help="Pasta de modelos, se não for <comfyui-dir>/models")
    ps.add_argument("--workflow", default=config.DEFAULT_WORKFLOW)
    pc = sub.add_parser("check", help="Prepara e verifica o motor do workflow")
    pc.add_argument("--workflow")
    pc.add_argument("--comfy-url")

    args = parser.parse_args(argv)
    if args.cmd == "setup" and args.models_dir and not args.comfyui_dir:
        parser.error("--models-dir requer --comfyui-dir no modo externo.")
    handler = {"setup": _setup, "check": _check}.get(args.cmd, _serve)
    try:
        return handler(args)
    except (RuntimeError, OSError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130
