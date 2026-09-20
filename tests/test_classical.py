from pathlib import Path

import pytest
from PIL import Image

from uiregress.inference import compare_images


def _write_image(path: Path, size: tuple[int, int], color: tuple[int, int, int]) -> None:
    Image.new("RGB", size, color).save(path)


def test_identical_images_are_not_different(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.png"
    current = tmp_path / "current.png"
    _write_image(baseline, (32, 32), (255, 255, 255))
    _write_image(current, (32, 32), (255, 255, 255))

    result = compare_images(baseline, current, device="cpu")

    assert result.different is False
    assert result.changed_area_ratio == 0.0
    assert result.mean_absolute_error == 0.0
    assert result.ssim == pytest.approx(1.0, abs=1e-6)


def test_changed_region_is_detected(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.png"
    current = tmp_path / "current.png"
    heatmap = tmp_path / "diff.png"

    base = Image.new("RGB", (32, 32), (255, 255, 255))
    changed = base.copy()
    for x in range(8, 24):
        for y in range(8, 24):
            changed.putpixel((x, y), (0, 0, 0))

    base.save(baseline)
    changed.save(current)

    result = compare_images(
        baseline,
        current,
        device="cpu",
        heatmap_path=heatmap,
    )

    assert result.different is True
    assert result.changed_area_ratio > 0.20
    assert result.ssim < 0.99
    assert heatmap.is_file()


def test_dimension_mismatch_is_rejected(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.png"
    current = tmp_path / "current.png"
    _write_image(baseline, (32, 32), (255, 255, 255))
    _write_image(current, (64, 32), (255, 255, 255))

    with pytest.raises(ValueError, match="dimensions must match"):
        compare_images(baseline, current, device="cpu")
