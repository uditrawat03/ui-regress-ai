from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class RegressionResult:
    regression: bool
    label: str
    confidence: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
