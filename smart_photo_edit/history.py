"""Resultados persistentes com metadados PNG e índice local recuperável."""
from __future__ import annotations

import io
import json
import re
import secrets
import time
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image
from PIL.PngImagePlugin import PngInfo

from . import paths


def image_metadata(data: bytes) -> dict | None:
    """Lê apenas os metadados do app, sem importar a imagem para o histórico."""
    with Image.open(io.BytesIO(data)) as image:
        try:
            record = json.loads(image.info.get('SmartPhotoEdit', '{}'))
        except ValueError:
            return None
    if not isinstance(record, dict) or not isinstance(record.get('prompt'), str):
        return None
    return {key: record[key] for key in ('prompt', 'seed', 'name', 'workflow', 'params', 'created_at', 'duration_seconds', 'addons', 'user_prompt', 'mode', 'models', 'model_choices', 'runtime_args', 'reference', 'negative', 'strength', 'variation', 'variations') if key in record}


class ResultStore:
    def __init__(self, folder: Path | None = None, max_age_s: float | None = None):
        self.folder = folder or paths.results_dir()
        self.max_age_s = max_age_s
        self.folder.mkdir(parents=True, exist_ok=True)
        if max_age_s is not None:
            self.purge()

    def purge(self):
        if self.max_age_s is None:
            return
        cutoff = time.time() - self.max_age_s
        for f in self.folder.iterdir():
            if f.is_file() and f.stat().st_mtime < cutoff:
                f.unlink(missing_ok=True)

    def path(self, rid: str) -> Path | None:
        if not re.fullmatch(r'[a-f0-9]{24}\.(png|jpg|webp)', rid):
            return None
        p = self.folder / rid
        return p if p.is_file() else None

    def save(self, data: bytes, metadata: dict | None = None, original: bytes | None = None) -> str:
        rid = secrets.token_hex(12) + '.png'
        record = {**(metadata or {}), 'id': rid, 'created_at': datetime.now(timezone.utc).isoformat()}
        with Image.open(io.BytesIO(data)) as image:
            image.load()
            record.update(width=image.width, height=image.height)
            info = PngInfo()
            for key, value in image.info.items():
                if isinstance(value, str) and key not in ('SmartPhotoEdit', 'Description'):
                    info.add_itxt('comfyui_prompt' if key == 'prompt' else key, value)
            info.add_itxt('prompt', record.get('prompt', ''))
            info.add_itxt('Description', record.get('prompt', ''))
            info.add_itxt('SmartPhotoEdit', json.dumps(record, ensure_ascii=False))
            target = self.folder / rid
            temporary = target.with_suffix('.part')
            image.save(temporary, format='PNG', pnginfo=info, **{key: image.info[key] for key in ('icc_profile', 'exif') if key in image.info})
            temporary.replace(target)
            # O PNG é a fonte de verdade; o JSON pode ser reconstruído se a escrita falhar.
            if original:
                (self.folder / (rid + '.orig')).write_bytes(original)
            self._write_record(rid, record)
        return rid

    def _write_record(self, rid, record):
        target = self.folder / (rid + '.json')
        temporary = target.with_suffix('.part')
        temporary.write_text(json.dumps(record, ensure_ascii=False), encoding='utf-8')
        temporary.replace(target)

    def record(self, rid):
        target = self.path(rid)
        if target is None:
            return None
        try:
            record = json.loads((self.folder / (rid + '.json')).read_text(encoding='utf-8'))
            if not isinstance(record, dict):
                raise ValueError('Índice inválido')
        except (OSError, ValueError):
            with Image.open(target) as image:
                try:
                    record = json.loads(image.info.get('SmartPhotoEdit', '{}'))
                    if not isinstance(record, dict):
                        record = {}
                except ValueError:
                    record = {}
                record.update(width=image.width, height=image.height)
            record.setdefault('prompt', '')
            record.setdefault('created_at', datetime.fromtimestamp(target.stat().st_mtime, timezone.utc).isoformat())
            self._write_record(rid, record)
        extra = {'original_url': f'/api/history/{rid}/original'} if (self.folder / (rid + '.orig')).is_file() else {}
        return {**record, 'id': rid, 'url': f'/api/results/{rid}', 'thumbnail_url': f'/api/history/{rid}/thumbnail', **extra}

    def original(self, rid):
        if self.path(rid) is None:
            return None
        target = self.folder / (rid + '.orig')
        return target if target.is_file() else None

    def list(self, offset=0, limit=30):
        files = sorted((p for p in self.folder.iterdir() if re.fullmatch(r'[a-f0-9]{24}\.(png|jpg|webp)', p.name)), key=lambda p: (p.stat().st_mtime_ns, p.name), reverse=True)
        items = []
        for p in files[offset:offset + limit]:
            try:
                record = self.record(p.name)
                if record:
                    items.append(record)
            except (OSError, ValueError):
                continue
        return {'items': items, 'has_more': offset + limit < len(files), 'next_offset': offset + limit}

    def thumbnail(self, rid):
        source = self.path(rid)
        if source is None:
            return None
        target = self.folder / (rid + '.thumb.webp')
        if not target.exists():
            with Image.open(source) as image:
                image.thumbnail((320, 320))
                image.convert('RGBA').save(target, format='WEBP', quality=80)
        return target

    def delete(self, rid):
        target = self.path(rid)
        if target is None:
            return False
        target.unlink()
        for suffix in ('.json', '.thumb.webp', '.orig'):
            (self.folder / (rid + suffix)).unlink(missing_ok=True)
        return True
