from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class DeviceInfo:
    requested: str
    selected: str
    cuda_available: bool
    cuda_device_count: int
    cuda_device_name: str | None


def resolve_device(requested: str = "auto") -> DeviceInfo:
    if requested not in {"auto", "cpu", "cuda"}:
        raise ValueError("device must be one of: auto, cpu, cuda")

    cuda_available = torch.cuda.is_available()

    if requested == "cuda" and not cuda_available:
        raise RuntimeError("CUDA was requested, but PyTorch reports that CUDA is unavailable.")

    selected = "cuda" if requested == "cuda" or (requested == "auto" and cuda_available) else "cpu"
    count = torch.cuda.device_count() if cuda_available else 0
    name = torch.cuda.get_device_name(0) if cuda_available and count else None

    return DeviceInfo(
        requested=requested,
        selected=selected,
        cuda_available=cuda_available,
        cuda_device_count=count,
        cuda_device_name=name,
    )
