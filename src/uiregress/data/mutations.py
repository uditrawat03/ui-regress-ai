from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Callable

from uiregress.data.schema import MutationSpec

ParameterFactory = Callable[[random.Random], dict[str, Any]]


@dataclass(frozen=True, slots=True)
class MutationTemplate:
    operator: str
    label: str
    parameters: ParameterFactory


def _no_parameters(_: random.Random) -> dict[str, Any]:
    return {}


def _translate_parameters(rng: random.Random) -> dict[str, Any]:
    return {
        "dx": rng.choice([-48, -32, 32, 48]),
        "dy": rng.choice([-24, 0, 24]),
    }


def _width_parameters(rng: random.Random) -> dict[str, Any]:
    return {"scale": rng.choice([1.25, 1.4, 1.6])}


def _clip_parameters(rng: random.Random) -> dict[str, Any]:
    return {"height_ratio": rng.choice([0.35, 0.5, 0.65])}


def _style_parameters(rng: random.Random) -> dict[str, Any]:
    return {
        "background": rng.choice(["#fca5a5", "#fde68a", "#c4b5fd"]),
        "color": rng.choice(["#111827", "#7f1d1d", "#312e81"]),
    }


MUTATION_TEMPLATES: tuple[MutationTemplate, ...] = (
    MutationTemplate("hide_element", "element_missing", _no_parameters),
    MutationTemplate("translate_element", "layout_shift", _translate_parameters),
    MutationTemplate("increase_width", "incorrect_size", _width_parameters),
    MutationTemplate("overflow_hidden", "text_clipping", _clip_parameters),
    MutationTemplate("change_style_token", "unexpected_style", _style_parameters),
)


def choose_mutation(rng: random.Random, selectors: list[str]) -> MutationSpec:
    if not selectors:
        raise ValueError("Fixture does not contain any data-uiregress-target elements.")

    template = rng.choice(MUTATION_TEMPLATES)
    selector = rng.choice(selectors)
    return MutationSpec(
        operator=template.operator,
        label=template.label,
        selector=selector,
        parameters=template.parameters(rng),
    )


def no_regression_mutation() -> MutationSpec:
    return MutationSpec(
        operator="subtle_pixel_noise",
        label="no_regression",
        selector=None,
        parameters={"amplitude": 1},
    )
