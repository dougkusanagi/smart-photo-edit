import io

import pytest
from PIL import Image

from smart_photo_edit import paths


@pytest.fixture(autouse=True)
def isolated_home(tmp_path, monkeypatch):
    monkeypatch.setenv("SPE_HOME", str(tmp_path / "spe-home"))
    monkeypatch.delenv("SPE_COMFY_URL", raising=False)
    return tmp_path / "spe-home"


def make_png(color=(200, 80, 40), size=(64, 48)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, "PNG")
    return buf.getvalue()


@pytest.fixture
def png() -> bytes:
    return make_png()
