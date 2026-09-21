import json
import random
from pathlib import Path
from typing import Self

import torch
from PIL import Image

from uiregress.data.augment import add_rendering_variation
from uiregress.data.generator import generate_dataset
from uiregress.data.mutations import (
    SEMANTIC_CHALLENGE_PROFILE,
    choose_mutation_for_label,
    no_regression_mutation,
)
from uiregress.data.renderer import RenderedPair
from uiregress.data.schema import MutationSpec
from uiregress.models import SiameseRegressionClassifier
from uiregress.training import _regression_label_mapping


class ChallengeFakeRenderer:
    def __init__(self, *, width: int, height: int) -> None:
        self.width = width
        self.height = height

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *_: object) -> None:
        return None

    async def render_pair(
        self,
        fixture: Path,
        baseline_path: Path,
        current_path: Path,
        **kwargs: object,
    ) -> RenderedPair:
        baseline_path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (self.width, self.height), "white").save(baseline_path)
        Image.new("RGB", (self.width, self.height), "white").save(current_path)
        no_regression = bool(kwargs["no_regression"])
        regression_label = kwargs.get("regression_label")
        label = "no_regression" if no_regression else str(regression_label or "layout_shift")
        return RenderedPair(
            label=label,
            mutation=MutationSpec(
                operator="fake",
                label=label,
                selector=None,
                parameters={"profile": kwargs["profile"]},
            ),
            region=None,
            browser={
                "engine": "chromium",
                "browser_version": "test",
                "playwright_version": "test",
            },
        )


def test_challenge_mutations_use_subtle_layout_parameters() -> None:
    mutation = choose_mutation_for_label(
        random.Random(42),
        ['[data-uiregress-target="card"]'],
        "layout_shift",
        profile=SEMANTIC_CHALLENGE_PROFILE,
    )

    assert abs(int(mutation.parameters["dx"])) <= 4
    assert abs(int(mutation.parameters["dy"])) <= 4
    assert mutation.parameters["profile"] == SEMANTIC_CHALLENGE_PROFILE


def test_challenge_negative_is_rendering_variation() -> None:
    mutation = no_regression_mutation(
        random.Random(42),
        profile=SEMANTIC_CHALLENGE_PROFILE,
    )

    assert mutation.operator == "rendering_variation"
    assert mutation.label == "no_regression"
    assert mutation.parameters["mode"] in {
        "pixel_noise",
        "brightness_shift",
        "translate_1px",
        "jpeg_roundtrip",
    }


def test_rendering_variation_is_deterministic(tmp_path) -> None:
    source = tmp_path / "source.png"
    first = tmp_path / "first.png"
    second = tmp_path / "second.png"
    Image.new("RGB", (32, 24), "#808080").save(source)

    add_rendering_variation(
        source,
        first,
        seed=42,
        mode="pixel_noise",
        amount=18,
    )
    add_rendering_variation(
        source,
        second,
        seed=42,
        mode="pixel_noise",
        amount=18,
    )

    assert first.read_bytes() == second.read_bytes()
    assert first.read_bytes() != source.read_bytes()


def test_regression_mapping_excludes_no_regression() -> None:
    mapping = _regression_label_mapping(
        {
            "no_regression": 0,
            "element_missing": 1,
            "incorrect_size": 2,
            "layout_shift": 3,
            "text_clipping": 4,
            "unexpected_style": 5,
        }
    )

    assert mapping == {
        "element_missing": 0,
        "incorrect_size": 1,
        "layout_shift": 2,
        "text_clipping": 3,
        "unexpected_style": 4,
    }


def test_model_type_head_has_five_logits() -> None:
    model = SiameseRegressionClassifier(num_classes=5, pretrained=False)
    model.eval()

    with torch.inference_mode():
        output = model(
            torch.randn(2, 3, 64, 64),
            torch.randn(2, 3, 64, 64),
        )

    assert output.binary_logits.shape == (2,)
    assert output.class_logits.shape == (2, 5)


def test_generator_records_semantic_challenge_profile(tmp_path, monkeypatch) -> None:
    fixtures = tmp_path / "fixtures"
    fixtures.mkdir()
    for name in ("dashboard", "checkout", "profile"):
        (fixtures / f"{name}.html").write_text("<html></html>", encoding="utf-8")

    monkeypatch.setattr(
        "uiregress.data.generator.PlaywrightRenderer",
        ChallengeFakeRenderer,
    )
    output = tmp_path / "dataset"

    summary = generate_dataset(
        fixtures,
        output,
        version="synthetic-v0.3-test",
        samples_per_class=1,
        seed=42,
        width=64,
        height=48,
        profile=SEMANTIC_CHALLENGE_PROFILE,
    )

    assert summary.generation_profile == SEMANTIC_CHALLENGE_PROFILE
    metadata = json.loads((output / "dataset.json").read_text(encoding="utf-8"))
    assert metadata["generation_profile"] == SEMANTIC_CHALLENGE_PROFILE
