from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, UnidentifiedImageError

from uiregress.device import resolve_device


@dataclass(frozen=True, slots=True)
class ClassicalComparisonResult:
    baseline: str
    current: str
    device: str
    width: int
    height: int
    mean_absolute_error: float
    changed_area_ratio: float
    ssim: float
    different: bool
    changed_pixel_threshold: float
    fail_changed_ratio: float
    fail_ssim_below: float
    heatmap: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _validate_unit_interval(name: str, value: float) -> None:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1")


def _load_rgb(path: Path) -> torch.Tensor:
    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {path}")

    try:
        with Image.open(path) as image:
            array = np.array(image.convert("RGB"), dtype=np.float32, copy=True)
    except UnidentifiedImageError as exc:
        raise ValueError(f"Unsupported or corrupted image: {path}") from exc

    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0) / 255.0


def _to_grayscale(image: torch.Tensor) -> torch.Tensor:
    weights = image.new_tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
    return (image * weights).sum(dim=1, keepdim=True)


def structural_similarity(
    baseline: torch.Tensor,
    current: torch.Tensor,
    *,
    window_size: int = 11,
) -> float:
    """Return a deterministic local SSIM-style score in the range [-1, 1]."""
    baseline_gray = _to_grayscale(baseline)
    current_gray = _to_grayscale(current)

    height, width = baseline_gray.shape[-2:]
    kernel = min(window_size, height, width)
    if kernel % 2 == 0:
        kernel -= 1
    kernel = max(kernel, 1)
    padding = kernel // 2

    def local_mean(tensor: torch.Tensor) -> torch.Tensor:
        return F.avg_pool2d(
            tensor,
            kernel_size=kernel,
            stride=1,
            padding=padding,
            count_include_pad=False,
        )

    mu_x = local_mean(baseline_gray)
    mu_y = local_mean(current_gray)

    sigma_x = (local_mean(baseline_gray.square()) - mu_x.square()).clamp_min(0.0)
    sigma_y = (local_mean(current_gray.square()) - mu_y.square()).clamp_min(0.0)
    sigma_xy = local_mean(baseline_gray * current_gray) - (mu_x * mu_y)

    c1 = 0.01**2
    c2 = 0.03**2

    numerator = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
    denominator = (mu_x.square() + mu_y.square() + c1) * (sigma_x + sigma_y + c2)

    score = numerator / denominator.clamp_min(torch.finfo(denominator.dtype).eps)
    return float(score.mean().clamp(-1.0, 1.0).item())


def _write_heatmap(pixel_difference: torch.Tensor, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    heatmap = (pixel_difference.squeeze(0).clamp(0.0, 1.0) * 255.0).to(torch.uint8)
    Image.fromarray(heatmap.cpu().numpy(), mode="L").save(output)


def compare_images(
    baseline_path: str | Path,
    current_path: str | Path,
    *,
    device: str = "auto",
    changed_pixel_threshold: float = 0.05,
    fail_changed_ratio: float = 0.01,
    fail_ssim_below: float = 0.99,
    heatmap_path: str | Path | None = None,
) -> ClassicalComparisonResult:
    """Compare two screenshots using deterministic non-ML metrics."""
    _validate_unit_interval("changed_pixel_threshold", changed_pixel_threshold)
    _validate_unit_interval("fail_changed_ratio", fail_changed_ratio)
    _validate_unit_interval("fail_ssim_below", fail_ssim_below)

    baseline_file = Path(baseline_path)
    current_file = Path(current_path)

    baseline = _load_rgb(baseline_file)
    current = _load_rgb(current_file)

    if baseline.shape != current.shape:
        baseline_size = (baseline.shape[-1], baseline.shape[-2])
        current_size = (current.shape[-1], current.shape[-2])
        raise ValueError(
            "Screenshot dimensions must match: "
            f"baseline={baseline_size[0]}x{baseline_size[1]}, "
            f"current={current_size[0]}x{current_size[1]}"
        )

    device_info = resolve_device(device)
    torch_device = torch.device(device_info.selected)
    baseline = baseline.to(torch_device)
    current = current.to(torch_device)

    absolute_difference = (baseline - current).abs()
    pixel_difference = absolute_difference.mean(dim=1)

    mean_absolute_error = float(absolute_difference.mean().item())
    changed_area_ratio = float(
        (pixel_difference >= changed_pixel_threshold).to(torch.float32).mean().item()
    )
    ssim = structural_similarity(baseline, current)

    different = changed_area_ratio >= fail_changed_ratio or ssim <= fail_ssim_below

    heatmap_value: str | None = None
    if heatmap_path is not None:
        heatmap_file = Path(heatmap_path)
        _write_heatmap(pixel_difference, heatmap_file)
        heatmap_value = str(heatmap_file)

    _, _, height, width = baseline.shape

    return ClassicalComparisonResult(
        baseline=str(baseline_file),
        current=str(current_file),
        device=device_info.selected,
        width=width,
        height=height,
        mean_absolute_error=mean_absolute_error,
        changed_area_ratio=changed_area_ratio,
        ssim=ssim,
        different=different,
        changed_pixel_threshold=changed_pixel_threshold,
        fail_changed_ratio=fail_changed_ratio,
        fail_ssim_below=fail_ssim_below,
        heatmap=heatmap_value,
    )
