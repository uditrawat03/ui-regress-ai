from __future__ import annotations

from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image


def _load_rgb_pixels(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.int16)


def _save_rgb_pixels(pixels: np.ndarray, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.clip(pixels, 0, 255).astype(np.uint8), mode="RGB").save(
        destination,
        format="PNG",
    )


def add_subtle_pixel_noise(
    source: str | Path,
    destination: str | Path,
    *,
    seed: int,
    amplitude: int = 1,
) -> None:
    """Create a pixel-level negative pair without changing UI semantics."""
    if amplitude < 0 or amplitude > 4:
        raise ValueError("amplitude must be between 0 and 4")

    source_path = Path(source)
    destination_path = Path(destination)
    pixels = _load_rgb_pixels(source_path)

    rng = np.random.default_rng(seed)
    noise = rng.integers(-amplitude, amplitude + 1, size=pixels.shape, dtype=np.int16)
    _save_rgb_pixels(pixels + noise, destination_path)


def add_rendering_variation(
    source: str | Path,
    destination: str | Path,
    *,
    seed: int,
    mode: str,
    amount: int,
) -> None:
    """Create a deterministic hard negative that changes pixels, not UI semantics."""
    source_path = Path(source)
    destination_path = Path(destination)
    pixels = _load_rgb_pixels(source_path)

    if mode == "pixel_noise":
        if not 1 <= amount <= 24:
            raise ValueError("pixel_noise amount must be between 1 and 24")
        rng = np.random.default_rng(seed)
        noise = rng.integers(-amount, amount + 1, size=pixels.shape, dtype=np.int16)
        _save_rgb_pixels(pixels + noise, destination_path)
        return

    if mode == "brightness_shift":
        if not -24 <= amount <= 24 or amount == 0:
            raise ValueError("brightness_shift amount must be between -24 and 24, excluding 0")
        _save_rgb_pixels(pixels + amount, destination_path)
        return

    if mode == "translate_1px":
        if amount not in {-1, 1}:
            raise ValueError("translate_1px amount must be -1 or 1")
        translated = pixels.copy()
        if amount > 0:
            translated[:, 1:] = pixels[:, :-1]
            translated[:, :1] = pixels[:, :1]
        else:
            translated[:, :-1] = pixels[:, 1:]
            translated[:, -1:] = pixels[:, -1:]
        _save_rgb_pixels(translated, destination_path)
        return

    if mode == "jpeg_roundtrip":
        if not 80 <= amount <= 100:
            raise ValueError("jpeg_roundtrip amount must be a JPEG quality from 80 to 100")
        buffer = BytesIO()
        Image.fromarray(pixels.astype(np.uint8), mode="RGB").save(
            buffer,
            format="JPEG",
            quality=amount,
            optimize=False,
        )
        buffer.seek(0)
        with Image.open(buffer) as compressed:
            roundtrip = np.asarray(compressed.convert("RGB"), dtype=np.int16)
        _save_rgb_pixels(roundtrip, destination_path)
        return

    raise ValueError(
        "Unknown rendering variation mode. Expected one of: "
        "pixel_noise, brightness_shift, translate_1px, jpeg_roundtrip"
    )
