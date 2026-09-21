from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


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
    destination_path.parent.mkdir(parents=True, exist_ok=True)

    with Image.open(source_path) as image:
        pixels = np.asarray(image.convert("RGB"), dtype=np.int16)

    rng = np.random.default_rng(seed)
    noise = rng.integers(-amplitude, amplitude + 1, size=pixels.shape, dtype=np.int16)
    augmented = np.clip(pixels + noise, 0, 255).astype(np.uint8)
    Image.fromarray(augmented, mode="RGB").save(destination_path, format="PNG")
