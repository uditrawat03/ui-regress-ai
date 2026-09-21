from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
NO_REGRESSION_LABEL = "no_regression"
PREPROCESSING_VERSION = "imagenet-resize-v1"


def _read_manifest(dataset_root: Path) -> list[dict[str, Any]]:
    manifest_path = dataset_root / "manifest.jsonl"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Dataset manifest not found: {manifest_path}")

    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        manifest_path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Invalid JSON in {manifest_path} at line {line_number}."
            ) from exc
        records.append(record)

    if not records:
        raise ValueError(f"Dataset manifest is empty: {manifest_path}")
    return records


def build_label_mapping(dataset_root: str | Path) -> dict[str, int]:
    root = Path(dataset_root)
    labels = {str(record["label"]) for record in _read_manifest(root)}
    ordered = sorted(labels)
    if NO_REGRESSION_LABEL in labels:
        ordered.remove(NO_REGRESSION_LABEL)
        ordered.insert(0, NO_REGRESSION_LABEL)
    return {label: index for index, label in enumerate(ordered)}


def load_dataset_metadata(dataset_root: str | Path) -> dict[str, Any]:
    metadata_path = Path(dataset_root) / "dataset.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Dataset metadata not found: {metadata_path}")
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def build_image_transform(input_size: int = 224) -> transforms.Compose:
    if input_size <= 0:
        raise ValueError("input_size must be greater than zero")
    return transforms.Compose(
        [
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


class PairedScreenshotDataset(Dataset[dict[str, Any]]):
    """Training dataset backed by a generated UIRegressAI manifest."""

    def __init__(
        self,
        dataset_root: str | Path,
        *,
        split: str,
        label_to_index: dict[str, int] | None = None,
        input_size: int = 224,
    ) -> None:
        if split not in {"train", "validation", "test"}:
            raise ValueError("split must be one of: train, validation, test")

        self.root = Path(dataset_root)
        self.split = split
        self.label_to_index = label_to_index or build_label_mapping(self.root)
        self.transform = build_image_transform(input_size)
        self.records = [
            record for record in _read_manifest(self.root) if record.get("split") == split
        ]

        if not self.records:
            raise ValueError(f"Dataset split '{split}' contains no samples.")

        unknown = {
            str(record["label"])
            for record in self.records
            if str(record["label"]) not in self.label_to_index
        }
        if unknown:
            raise ValueError(f"Label mapping is missing labels: {sorted(unknown)}")

    def __len__(self) -> int:
        return len(self.records)

    def _load_image(self, relative_path: str) -> torch.Tensor:
        path = self.root / relative_path
        if not path.is_file():
            raise FileNotFoundError(f"Dataset image not found: {path}")
        with Image.open(path) as image:
            return self.transform(image.convert("RGB"))

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        label = str(record["label"])
        return {
            "baseline": self._load_image(str(record["baseline"])),
            "current": self._load_image(str(record["current"])),
            "binary_target": torch.tensor(
                float(label != NO_REGRESSION_LABEL),
                dtype=torch.float32,
            ),
            "class_target": torch.tensor(
                self.label_to_index[label],
                dtype=torch.long,
            ),
            "sample_id": str(record["sample_id"]),
            "label": label,
        }
