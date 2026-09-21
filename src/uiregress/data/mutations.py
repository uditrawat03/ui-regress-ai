from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

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
REGRESSION_LABELS: tuple[str, ...] = tuple(template.label for template in MUTATION_TEMPLATES)
ALL_LABELS: tuple[str, ...] = ("no_regression", *REGRESSION_LABELS)
_TEMPLATE_BY_LABEL = {template.label: template for template in MUTATION_TEMPLATES}


def _mutation_from_template(
    template: MutationTemplate,
    rng: random.Random,
    selectors: list[str],
) -> MutationSpec:
    selector = rng.choice(selectors)
    return MutationSpec(
        operator=template.operator,
        label=template.label,
        selector=selector,
        parameters=template.parameters(rng),
    )


def choose_mutation(rng: random.Random, selectors: list[str]) -> MutationSpec:
    if not selectors:
        raise ValueError("Fixture does not contain any data-uiregress-target elements.")

    template = rng.choice(MUTATION_TEMPLATES)
    return _mutation_from_template(template, rng, selectors)


def choose_mutation_for_label(
    rng: random.Random,
    selectors: list[str],
    label: str,
) -> MutationSpec:
    """Choose a seeded mutation for a specific semantic regression label."""
    if not selectors:
        raise ValueError("Fixture does not contain any data-uiregress-target elements.")

    try:
        template = _TEMPLATE_BY_LABEL[label]
    except KeyError as exc:
        raise ValueError(
            f"Unknown regression label: {label}. Expected one of: {', '.join(REGRESSION_LABELS)}"
        ) from exc
    return _mutation_from_template(template, rng, selectors)


def no_regression_mutation() -> MutationSpec:
    return MutationSpec(
        operator="subtle_pixel_noise",
        label="no_regression",
        selector=None,
        parameters={"amplitude": 1},
    )
