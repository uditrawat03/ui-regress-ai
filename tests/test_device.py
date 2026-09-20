import pytest

from uiregress.device import resolve_device


def test_cpu_is_always_selectable() -> None:
    info = resolve_device("cpu")
    assert info.selected == "cpu"


def test_invalid_device_is_rejected() -> None:
    with pytest.raises(ValueError):
        resolve_device("tpu")
