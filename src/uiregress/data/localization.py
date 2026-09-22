from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image, ImageDraw

from uiregress.data.schema import BoundingBox

LOCALIZATION_TARGET_VERSION = "normalized-xyxy-mask-v1"
DEFAULT_LOCALIZATION_SIZE = 56


@dataclass(frozen=True, slots=True)
class NormalizedBox:
    """Viewport-normalized XYXY box with coordinates clamped to [0, 1]."""

    x1: float
    y1: float
    x2: float
    y2: float

    def to_tensor(self) -> torch.Tensor:
        return torch.tensor([self.x1, self.y1, self.x2, self.y2], dtype=torch.float32)

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class LocalizationTarget:
    valid: bool
    box: NormalizedBox | None
    mask: torch.Tensor


def _viewport_dimensions(viewport: dict[str, Any]) -> tuple[float, float]:
    try:
        width = float(viewport["width"])
        height = float(viewport["height"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("viewport must contain positive width and height values") from exc
    if width <= 0 or height <= 0:
        raise ValueError("viewport width and height must be positive")
    return width, height


def normalize_region(
    region: BoundingBox | dict[str, float] | None,
    viewport: dict[str, Any],
) -> NormalizedBox | None:
    """Convert a pixel-space region to a clamped normalized XYXY box."""
    if region is None:
        return None
    if isinstance(region, dict):
        region = BoundingBox.from_mapping(region)
    if region is None:
        return None

    viewport_width, viewport_height = _viewport_dimensions(viewport)
    left = min(max(region.x, 0.0), viewport_width)
    top = min(max(region.y, 0.0), viewport_height)
    right = min(max(region.x + region.width, 0.0), viewport_width)
    bottom = min(max(region.y + region.height, 0.0), viewport_height)

    if right <= left or bottom <= top:
        return None

    return NormalizedBox(
        x1=left / viewport_width,
        y1=top / viewport_height,
        x2=right / viewport_width,
        y2=bottom / viewport_height,
    )


def box_to_mask(
    box: NormalizedBox | None,
    *,
    size: int = DEFAULT_LOCALIZATION_SIZE,
) -> torch.Tensor:
    """Rasterize a normalized box into a 1xHxW binary localization mask."""
    if size <= 0:
        raise ValueError("localization mask size must be greater than zero")

    mask = torch.zeros((1, size, size), dtype=torch.float32)
    if box is None:
        return mask

    x1 = max(0, min(size - 1, math.floor(box.x1 * size)))
    y1 = max(0, min(size - 1, math.floor(box.y1 * size)))
    x2 = max(x1 + 1, min(size, math.ceil(box.x2 * size)))
    y2 = max(y1 + 1, min(size, math.ceil(box.y2 * size)))
    mask[:, y1:y2, x1:x2] = 1.0
    return mask


def build_localization_target(
    record: dict[str, Any],
    *,
    mask_size: int = DEFAULT_LOCALIZATION_SIZE,
) -> LocalizationTarget:
    """Build localization supervision directly from a manifest record."""
    region = record.get("region")
    viewport = record.get("viewport")
    if region is None or not isinstance(viewport, dict):
        return LocalizationTarget(valid=False, box=None, mask=box_to_mask(None, size=mask_size))

    box = normalize_region(region, viewport)
    if box is None:
        return LocalizationTarget(valid=False, box=None, mask=box_to_mask(None, size=mask_size))
    return LocalizationTarget(valid=True, box=box, mask=box_to_mask(box, size=mask_size))


def _coerce_box(value: NormalizedBox | Sequence[float] | torch.Tensor) -> NormalizedBox:
    if isinstance(value, NormalizedBox):
        return value
    if isinstance(value, torch.Tensor):
        values = value.detach().cpu().flatten().tolist()
    else:
        values = list(value)
    if len(values) != 4:
        raise ValueError("localization boxes must contain four XYXY coordinates")
    x1, y1, x2, y2 = (float(item) for item in values)
    return NormalizedBox(x1=x1, y1=y1, x2=x2, y2=y2)


def box_iou(
    predicted: NormalizedBox | Sequence[float] | torch.Tensor,
    target: NormalizedBox | Sequence[float] | torch.Tensor,
) -> float:
    """Intersection-over-union for normalized XYXY boxes."""
    predicted_box = _coerce_box(predicted)
    target_box = _coerce_box(target)

    intersection_width = max(
        0.0,
        min(predicted_box.x2, target_box.x2) - max(predicted_box.x1, target_box.x1),
    )
    intersection_height = max(
        0.0,
        min(predicted_box.y2, target_box.y2) - max(predicted_box.y1, target_box.y1),
    )
    intersection = intersection_width * intersection_height

    predicted_area = max(0.0, predicted_box.x2 - predicted_box.x1) * max(
        0.0, predicted_box.y2 - predicted_box.y1
    )
    target_area = max(0.0, target_box.x2 - target_box.x1) * max(
        0.0, target_box.y2 - target_box.y1
    )
    union = predicted_area + target_area - intersection
    if union <= 0.0:
        return 0.0
    return intersection / union


def center_hit(
    predicted: NormalizedBox | Sequence[float] | torch.Tensor,
    target: NormalizedBox | Sequence[float] | torch.Tensor,
) -> bool:
    """Return whether the predicted box center lies inside the ground-truth box."""
    predicted_box = _coerce_box(predicted)
    target_box = _coerce_box(target)
    center_x = (predicted_box.x1 + predicted_box.x2) / 2.0
    center_y = (predicted_box.y1 + predicted_box.y2) / 2.0
    return (
        target_box.x1 <= center_x <= target_box.x2
        and target_box.y1 <= center_y <= target_box.y2
    )


def localization_metrics(
    predicted_boxes: Sequence[NormalizedBox | Sequence[float] | torch.Tensor],
    target_boxes: Sequence[NormalizedBox | Sequence[float] | torch.Tensor],
) -> dict[str, float | int]:
    """Aggregate box IoU and center-hit metrics for matched predictions/targets."""
    if len(predicted_boxes) != len(target_boxes):
        raise ValueError("predicted_boxes and target_boxes must have the same length")
    if not predicted_boxes:
        return {"samples": 0, "mean_iou": 0.0, "center_hit_rate": 0.0}

    ious = [
        box_iou(predicted, target)
        for predicted, target in zip(predicted_boxes, target_boxes, strict=True)
    ]
    hits = [
        center_hit(predicted, target)
        for predicted, target in zip(predicted_boxes, target_boxes, strict=True)
    ]
    return {
        "samples": len(ious),
        "mean_iou": sum(ious) / len(ious),
        "center_hit_rate": sum(hits) / len(hits),
    }


def _read_manifest(dataset_root: Path) -> list[dict[str, Any]]:
    manifest_path = dataset_root / "manifest.jsonl"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Dataset manifest not found: {manifest_path}")
    records = [
        json.loads(line)
        for line in manifest_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not records:
        raise ValueError(f"Dataset manifest is empty: {manifest_path}")
    return records


def _pixel_box(box: NormalizedBox, width: int, height: int) -> tuple[int, int, int, int]:
    return (
        round(box.x1 * width),
        round(box.y1 * height),
        round(box.x2 * width),
        round(box.y2 * height),
    )


def export_localization_targets(
    dataset_root: str | Path,
    output_dir: str | Path,
    *,
    split: str = "train",
    limit: int = 12,
    mask_size: int = DEFAULT_LOCALIZATION_SIZE,
) -> dict[str, Any]:
    """Export ground-truth overlays and masks for manual localization inspection."""
    if split not in {"train", "validation", "test"}:
        raise ValueError("split must be one of: train, validation, test")
    if limit <= 0:
        raise ValueError("limit must be greater than zero")
    if mask_size <= 0:
        raise ValueError("mask_size must be greater than zero")

    root = Path(dataset_root)
    output = Path(output_dir)
    overlays_dir = output / "overlays"
    masks_dir = output / "masks"
    overlays_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    records = [record for record in _read_manifest(root) if record.get("split") == split]
    target_records = [record for record in records if build_localization_target(record).valid]
    selected = target_records[:limit]
    exported: list[dict[str, Any]] = []

    for record in selected:
        target = build_localization_target(record, mask_size=mask_size)
        if target.box is None:
            continue
        sample_id = str(record["sample_id"])
        current_path = root / str(record["current"])
        if not current_path.is_file():
            raise FileNotFoundError(f"Dataset image not found: {current_path}")

        with Image.open(current_path) as source:
            image = source.convert("RGB")
        draw = ImageDraw.Draw(image)
        draw.rectangle(_pixel_box(target.box, image.width, image.height), outline="red", width=3)
        draw.text((8, 8), str(record.get("label", "regression")), fill="red")

        overlay_relative = Path("overlays") / f"{sample_id}.png"
        mask_relative = Path("masks") / f"{sample_id}.png"
        image.save(output / overlay_relative, format="PNG")

        mask_image = Image.fromarray(
            (target.mask.squeeze(0).mul(255).to(torch.uint8).numpy()),
            mode="L",
        )
        mask_image.save(output / mask_relative, format="PNG")

        exported.append(
            {
                "sample_id": sample_id,
                "label": str(record.get("label", "")),
                "box": target.box.to_dict(),
                "overlay": overlay_relative.as_posix(),
                "mask": mask_relative.as_posix(),
            }
        )

    summary = {
        "target_version": LOCALIZATION_TARGET_VERSION,
        "dataset": str(root),
        "split": split,
        "split_samples": len(records),
        "localizable_samples": len(target_records),
        "exported_samples": len(exported),
        "mask_size": mask_size,
        "output": str(output.resolve()),
        "samples": exported,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary
