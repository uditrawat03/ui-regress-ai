"""Synthetic dataset generation and manifest utilities."""

from uiregress.data.generator import generate_dataset
from uiregress.data.schema import BoundingBox, DatasetSummary, MutationSpec, SampleManifest

__all__ = [
    "BoundingBox",
    "DatasetSummary",
    "MutationSpec",
    "SampleManifest",
    "generate_dataset",
]
