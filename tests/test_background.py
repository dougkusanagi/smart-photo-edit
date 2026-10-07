import io

import pytest
from PIL import Image, ImageOps

from smart_photo_edit.background import apply_alpha


def encoded(image, **kwargs):
    buf = io.BytesIO()
    image.save(buf, format='PNG', **kwargs)
    return buf.getvalue()


def test_background_keeps_original_detail_and_color_profile():
    source = Image.new('RGB', (257, 193))
    source.putdata([(i % 256, (i * 13) % 256, (i * 71) % 256) for i in range(257 * 193)])
    mask = Image.new('RGBA', (32, 24), (0, 0, 0, 128))
    result = Image.open(io.BytesIO(apply_alpha(encoded(source, icc_profile=b'test-profile'), encoded(mask))))
    assert result.size == source.size
    assert result.convert('RGB').tobytes() == source.tobytes()
    assert result.getchannel('A').getextrema() == (128, 128)
    assert result.info['icc_profile'] == b'test-profile'


def test_background_keeps_existing_transparency():
    source = Image.new('RGBA', (3, 1), (11, 22, 33, 255))
    source.putalpha(Image.frombytes('L', (3, 1), bytes([0, 128, 255])))
    result = Image.open(io.BytesIO(apply_alpha(encoded(source), encoded(Image.new('RGBA', (3, 1), (0, 0, 0, 128))))))
    assert result.convert('RGB').tobytes() == source.convert('RGB').tobytes()
    assert result.getchannel('A').tobytes() == bytes([0, 64, 128])


def test_background_respects_exif_orientation():
    source = Image.new('RGB', (3, 2))
    source.putdata([(i * 30, 0, 0) for i in range(6)])
    exif = source.getexif()
    exif[274] = 6
    original = encoded(source, exif=exif)
    result = Image.open(io.BytesIO(apply_alpha(original, encoded(Image.new('RGBA', (2, 3), (0, 0, 0, 255))))))
    expected = ImageOps.exif_transpose(Image.open(io.BytesIO(original)))
    assert result.size == (2, 3)
    assert result.convert('RGB').tobytes() == expected.tobytes()
    assert result.getexif().get(274, 1) == 1


def test_background_rejects_result_without_alpha():
    with pytest.raises(ValueError, match='transparência'):
        apply_alpha(encoded(Image.new('RGB', (2, 2))), encoded(Image.new('RGB', (2, 2))))
