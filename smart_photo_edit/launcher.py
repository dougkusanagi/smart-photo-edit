"""Inicia (e encerra) o ComfyUI como subprocesso quando há um comando configurado."""
from __future__ import annotations

import os
import shlex
import subprocess
import sys
from pathlib import Path

from . import paths


def split_command(command: str, posix: bool | None = None) -> list[str]:
    """Divide a linha de comando respeitando aspas, em Linux e Windows (caminhos com \\)."""
    if posix is None:
        posix = os.name != "nt"
    parts = shlex.split(command, posix=posix)
    if not posix:  # no modo Windows o shlex mantém as aspas nos argumentos
        parts = [p[1:-1] if len(p) > 1 and p[0] == p[-1] and p[0] in "\"'" else p for p in parts]
    return parts


class ComfyLauncher:
    def __init__(self) -> None:
        self.proc: subprocess.Popen | None = None
        self._log = None

    @property
    def state(self) -> str:
        if self.proc is None:
            return "stopped"
        return "running" if self.proc.poll() is None else "exited"

    def start(self, command: str, cwd: str = "") -> None:
        if self.state == "running":
            return
        argv = split_command(command)
        if not argv:
            raise ValueError("Nenhum comando configurado para iniciar o ComfyUI.")
        workdir = Path(cwd).expanduser() if cwd else None
        if workdir and not workdir.is_dir():
            raise ValueError(f"A pasta de trabalho não existe: {workdir}")
        log = paths.log_file()
        log.parent.mkdir(parents=True, exist_ok=True)
        self._log = open(log, "ab")
        kwargs: dict = {}
        if sys.platform == "win32":
            kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP | getattr(subprocess, "CREATE_NO_WINDOW", 0)
        else:
            kwargs["start_new_session"] = True
        try:
            self.proc = subprocess.Popen(argv, cwd=workdir, stdout=self._log, stderr=subprocess.STDOUT, **kwargs)
        except OSError as exc:
            self._close_log()
            raise ValueError(f"Não consegui executar o comando: {exc}") from None

    def stop(self) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self.proc = None
        self._close_log()

    def _close_log(self) -> None:
        if self._log:
            self._log.close()
            self._log = None
