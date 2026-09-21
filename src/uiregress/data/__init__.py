"""Synthetic dataset generation and manifest utilities."""

from uiregress.data.generator import generate_dataset
from uiregress.data.pairs import (
    NO_REGRESSION_LABEL,
    PREPROCESSING_VERSION,
    PairedScreenshotDataset,
    build_image_transform,
    build_label_mapping,
    load_dataset_metadata,
)
from uiregress.data.schema import BoundingBox, DatasetSummary, MutationSpec, SampleManifest

__all__ = [
    "NO_REGRESSION_LABEL",
    "PREPROCESSING_VERSION",
    "BoundingBox",
    "DatasetSummary",
    "MutationSpec",
    "PairedScreenshotDataset",
    "SampleManifest",
    "build_image_transform",
    "build_label_mapping",
    "generate_dataset",
    "load_dataset_metadata",
]
