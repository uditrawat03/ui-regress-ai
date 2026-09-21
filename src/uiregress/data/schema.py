from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

DatasetSplit = Literal["train", "validation", "test"]


@dataclass(frozen=True, slots=True)
class BoundingBox:
    x: float
    y: float
    width: float
    height: float

    @classmethod
    def from_mapping(cls, value: dict[str, float] | None) -> BoundingBox | None:
        if value is None:
            return None
        return cls(
            x=float(value["x"]),
            y=float(value["y"]),
            width=float(value["width"]),
            height=float(value["height"]),
        )

    def union(self, other: BoundingBox | None) -> BoundingBox:
        if other is None:
            return self
        left = min(self.x, other.x)
        top = min(self.y, other.y)
        right = max(self.x + self.width, other.x + other.width)
        bottom = max(self.y + self.height, other.y + other.height)
        return BoundingBox(x=left, y=top, width=right - left, height=bottom - top)


@dataclass(frozen=True, slots=True)
class MutationSpec:
    operator: str
    label: str
    selector: str | None
    parameters: dict[str, Any]


@dataclass(frozen=True, slots=True)
class SampleManifest:
    sample_id: str
    fixture: str
    split: DatasetSplit
    seed: int
    baseline: str
    current: str
    label: str
    viewport: dict[str, int]
    region: BoundingBox | None
    mutation: MutationSpec
    browser: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class DatasetSummary:
    version: str
    seed: int
    total_samples: int
    fixtures: int
    samples_per_fixture: int
    manifest: str
    splits: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
