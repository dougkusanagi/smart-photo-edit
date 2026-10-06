"""Linha de comando: `python -m smart_photo_edit [serve|setup|check]`."""
from __future__ import annotations

import argparse
import asyncio
import sys
import threading
import webbrowser
from pathlib import Path

from aiohttp import web

from . import __version__, config, paths, workflows
from .installer import install_requirements
from .server import AppState, create_app
from .service import ResultStore


def _serve(args: argparse.Namespace) -> int:
    settings = config.load()
    if args.comfy_url:
        settings.comfy_url = config.normalize_url(args.comfy_url)
    wildcard = args.host in ("0.0.0.0", "::")
    state = AppState(settings=settings, store=ResultStore(), allow_any_host=wildcard)
    url = f"http://{'localhost' if wildcard else args.host}:{args.port}/"
    print(f"Smart Photo Edit {__version__}\n  Interface: {url}\n  ComfyUI:   {settings.comfy_url}\n  Dados:     {paths.user_data_dir()}")
    if wildcard:
        print("  ⚠ Escutando em todas as interfaces: qualquer pessoa na rede poderá usar sua GPU.")
    if not args.no_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    web.run_app(create_app(state), host=args.host, port=args.port, print=None)
    return 0


def _setup(args: argparse.Namespace) -> int:
    wfs, _ = workflows.discover()
    wf = wfs.get(args.workflow)
    if not wf:
        print(f"Workflow desconhecido: {args.workflow}. Disponíveis: {', '.join(wfs)}", file=sys.stderr)
        return 2
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

    async def go() -> int:
        async with aiohttp.ClientSession() as session:
            client = ComfyClient(args.comfy_url or settings.comfy_url, session)
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
        p.add_argument("--comfy-url", help="Endereço do ComfyUI (padrão: configuração salva)")
        p.add_argument("--no-browser", action="store_true")

    add_serve(sub.add_parser("serve", help="Inicia o app (padrão)"))
    add_serve(parser)
    ps = sub.add_parser("setup", help="Baixa os nós customizados e modelos de um workflow")
    ps.add_argument("--comfyui-dir", required=True, help="Pasta do ComfyUI (a que contém custom_nodes/)")
    ps.add_argument("--models-dir", help="Pasta de modelos, se não for <comfyui-dir>/models")
    ps.add_argument("--workflow", default=config.DEFAULT_WORKFLOW)
    pc = sub.add_parser("check", help="Verifica se o ComfyUI tem tudo que o workflow precisa")
    pc.add_argument("--workflow")
    pc.add_argument("--comfy-url")

    args = parser.parse_args(argv)
    handler = {"setup": _setup, "check": _check}.get(args.cmd, _serve)
    try:
        return handler(args)
    except KeyboardInterrupt:
        return 130
