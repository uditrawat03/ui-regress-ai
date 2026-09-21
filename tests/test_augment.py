import numpy as np
from PIL import Image

from uiregress.data.augment import add_subtle_pixel_noise


def test_subtle_noise_is_seeded_and_bounded(tmp_path) -> None:
    source = tmp_path / "source.png"
    first = tmp_path / "first.png"
    second = tmp_path / "second.png"
    Image.new("RGB", (12, 12), color=(120, 130, 140)).save(source)

    add_subtle_pixel_noise(source, first, seed=7, amplitude=1)
    add_subtle_pixel_noise(source, second, seed=7, amplitude=1)

    original = np.asarray(Image.open(source), dtype=np.int16)
    first_pixels = np.asarray(Image.open(first), dtype=np.int16)
    second_pixels = np.asarray(Image.open(second), dtype=np.int16)

    assert np.array_equal(first_pixels, second_pixels)
    assert np.abs(first_pixels - original).max() <= 1
    assert not np.array_equal(first_pixels, original)
