from dataclasses import dataclass
from typing import Literal

DeviceName = Literal["auto", "cpu", "cuda"]


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    device: DeviceName = "auto"
    seed: int = 42
