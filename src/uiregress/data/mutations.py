from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from uiregress.data.schema import MutationSpec

ParameterFactory = Callable[[random.Random], dict[str, Any]]

STANDARD_PROFILE = "standard"
SEMANTIC_CHALLENGE_PROFILE = "semantic-challenge"
GENERATION_PROFILES = (STANDARD_PROFILE, SEMANTIC_CHALLENGE_PROFILE)


@dataclass(frozen=True, slots=True)
class MutationTemplate:
    operator: str
    label: str
    parameters: ParameterFactory
    challenge_parameters: ParameterFactory


def _no_parameters(_: random.Random) -> dict[str, Any]:
    return {}


def _translate_parameters(rng: random.Random) -> dict[str, Any]:
    return {
        "dx": rng.choice([-48, -32, 32, 48]),
        "dy": rng.choice([-24, 0, 24]),
    }


def _challenge_translate_parameters(rng: random.Random) -> dict[str, Any]:
    return {
        "dx": rng.choice([-4, -2, 2, 4]),
        "dy": rng.choice([-4, -2, 0, 2, 4]),
    }


def _width_parameters(rng: random.Random) -> dict[str, Any]:
    return {"scale": rng.choice([1.25, 1.4, 1.6])}


def _challenge_width_parameters(rng: random.Random) -> dict[str, Any]:
    return {"scale": rng.choice([1.03, 1.05, 1.08])}


def _clip_parameters(rng: random.Random) -> dict[str, Any]:
    return {"height_ratio": rng.choice([0.35, 0.5, 0.65])}


def _challenge_clip_parameters(rng: random.Random) -> dict[str, Any]:
    return {"height_ratio": rng.choice([0.86, 0.9, 0.94])}


def _style_parameters(rng: random.Random) -> dict[str, Any]:
    return {
        "background": rng.choice(["#fca5a5", "#fde68a", "#c4b5fd"]),
        "color": rng.choice(["#111827", "#7f1d1d", "#312e81"]),
    }


def _challenge_style_parameters(rng: random.Random) -> dict[str, Any]:
    return {
        "background": rng.choice(["#f3f4f6", "#eef2ff", "#fef3c7"]),
        "color": rng.choice(["#374151", "#4b5563", "#4338ca"]),
    }


MUTATION_TEMPLATES: tuple[MutationTemplate, ...] = (
    MutationTemplate(
        "hide_element",
        "element_missing",
        _no_parameters,
        _no_parameters,
    ),
    MutationTemplate(
        "translate_element",
        "layout_shift",
        _translate_parameters,
        _challenge_translate_parameters,
    ),
    MutationTemplate(
        "increase_width",
        "incorrect_size",
        _width_parameters,
        _challenge_width_parameters,
    ),
    MutationTemplate(
        "overflow_hidden",
        "text_clipping",
        _clip_parameters,
        _challenge_clip_parameters,
    ),
    MutationTemplate(
        "change_style_token",
        "unexpected_style",
        _style_parameters,
        _challenge_style_parameters,
    ),
)
REGRESSION_LABELS: tuple[str, ...] = tuple(template.label for template in MUTATION_TEMPLATES)
ALL_LABELS: tuple[str, ...] = ("no_regression", *REGRESSION_LABELS)
_TEMPLATE_BY_LABEL = {template.label: template for template in MUTATION_TEMPLATES}


def validate_generation_profile(profile: str) -> str:
    if profile not in GENERATION_PROFILES:
        raise ValueError(
            f"Unknown generation profile: {profile}. "
            f"Expected one of: {', '.join(GENERATION_PROFILES)}"
        )
    return profile


def _mutation_from_template(
    template: MutationTemplate,
    rng: random.Random,
    selectors: list[str],
    *,
    profile: str,
) -> MutationSpec:
    validate_generation_profile(profile)
    selector = rng.choice(selectors)
    parameter_factory = (
        template.challenge_parameters
        if profile == SEMANTIC_CHALLENGE_PROFILE
        else template.parameters
    )
    parameters = parameter_factory(rng)
    parameters["profile"] = profile
    return MutationSpec(
        operator=template.operator,
        label=template.label,
        selector=selector,
        parameters=parameters,
    )


def choose_mutation(
    rng: random.Random,
    selectors: list[str],
    *,
    profile: str = STANDARD_PROFILE,
) -> MutationSpec:
    if not selectors:
        raise ValueError("Fixture does not contain any data-uiregress-target elements.")

    template = rng.choice(MUTATION_TEMPLATES)
    return _mutation_from_template(template, rng, selectors, profile=profile)


def choose_mutation_for_label(
    rng: random.Random,
    selectors: list[str],
    label: str,
    *,
    profile: str = STANDARD_PROFILE,
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
    return _mutation_from_template(template, rng, selectors, profile=profile)


def no_regression_mutation(
    rng: random.Random | None = None,
    *,
    profile: str = STANDARD_PROFILE,
) -> MutationSpec:
    validate_generation_profile(profile)
    if profile == STANDARD_PROFILE:
        return MutationSpec(
            operator="subtle_pixel_noise",
            label="no_regression",
            selector=None,
            parameters={"amplitude": 1, "profile": profile},
        )

    challenge_rng = rng or random.Random(0)
    mode = challenge_rng.choice(
        ["pixel_noise", "brightness_shift", "translate_1px", "jpeg_roundtrip"]
    )
    if mode == "pixel_noise":
        amount = challenge_rng.choice([14, 18, 22])
    elif mode == "brightness_shift":
        amount = challenge_rng.choice([-16, -14, 14, 16])
    elif mode == "translate_1px":
        amount = challenge_rng.choice([-1, 1])
    else:
        amount = challenge_rng.choice([88, 90, 92])

    return MutationSpec(
        operator="rendering_variation",
        label="no_regression",
        selector=None,
        parameters={
            "mode": mode,
            "amount": amount,
            "profile": profile,
        },
    )
