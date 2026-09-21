from __future__ import annotations

import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Self

from PIL import Image

from uiregress.data.generator import generate_dataset
from uiregress.data.mutations import ALL_LABELS, choose_mutation_for_label
from uiregress.data.renderer import RenderedPair
from uiregress.data.schema import BoundingBox, MutationSpec

_OPERATOR_BY_LABEL = {
    "element_missing": "hide_element",
    "incorrect_size": "increase_width",
    "layout_shift": "translate_element",
    "text_clipping": "overflow_hidden",
    "unexpected_style": "change_style_token",
}


class BalancedFakeRenderer:
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
        mutation = MutationSpec(
            operator=("subtle_pixel_noise" if no_regression else _OPERATOR_BY_LABEL[label]),
            label=label,
            selector=None if no_regression else '[data-uiregress-target="card"]',
            parameters={"amplitude": 1} if no_regression else {},
        )
        return RenderedPair(
            label=label,
            mutation=mutation,
            region=None if no_regression else BoundingBox(x=10, y=10, width=100, height=50),
            browser={
                "engine": "chromium",
                "browser_version": "test",
                "playwright_version": "test",
            },
        )


def test_balanced_generation_has_equal_classes_and_fixture_level_splits(
    tmp_path, monkeypatch
) -> None:
    fixtures = tmp_path / "fixtures"
    fixtures.mkdir()
    for index in range(18):
        (fixtures / f"fixture-{index:02d}.html").write_text("<html></html>", encoding="utf-8")

    monkeypatch.setattr("uiregress.data.generator.PlaywrightRenderer", BalancedFakeRenderer)
    output = tmp_path / "dataset"
    summary = generate_dataset(
        fixtures,
        output,
        version="synthetic-balanced-test",
        samples_per_class=2,
        seed=42,
        width=64,
        height=48,
    )

    assert summary.generation_mode == "balanced"
    assert summary.samples_per_class == 2
    assert summary.samples_per_fixture == len(ALL_LABELS) * 2
    assert summary.total_samples == 18 * len(ALL_LABELS) * 2
    assert summary.splits == {"train": 144, "validation": 36, "test": 36}
    assert summary.class_distribution == {label: 36 for label in ALL_LABELS}
    assert summary.split_class_distribution["train"] == {label: 24 for label in ALL_LABELS}
    assert summary.split_class_distribution["validation"] == {label: 6 for label in ALL_LABELS}
    assert summary.split_class_distribution["test"] == {label: 6 for label in ALL_LABELS}

    rows = [json.loads(line) for line in (output / "manifest.jsonl").read_text().splitlines()]
    assert Counter(row["label"] for row in rows) == Counter({label: 36 for label in ALL_LABELS})
    fixture_splits: dict[str, set[str]] = {}
    for row in rows:
        fixture_splits.setdefault(row["fixture"], set()).add(row["split"])
    assert all(len(splits) == 1 for splits in fixture_splits.values())

    metadata = json.loads((output / "dataset.json").read_text())
    assert metadata["generation_mode"] == "balanced"
    assert metadata["samples_per_class"] == 2
    assert metadata["class_distribution"] == {label: 36 for label in ALL_LABELS}


def test_forced_mutation_uses_requested_label() -> None:
    selectors = ['[data-uiregress-target="card"]']
    for label in ALL_LABELS[1:]:
        mutation = choose_mutation_for_label(random.Random(42), selectors, label)
        assert mutation.label == label
        assert mutation.selector == selectors[0]


def test_fixture_corpus_has_diverse_targets() -> None:
    fixture_dir = Path(__file__).resolve().parents[1] / "fixtures"
    fixtures = sorted(fixture_dir.glob("*.html"))
    assert len(fixtures) >= 18

    target_pattern = re.compile(r'data-uiregress-target="([^"]+)"')
    for fixture in fixtures:
        targets = target_pattern.findall(fixture.read_text(encoding="utf-8"))
        assert len(set(targets)) >= 5, f"{fixture.name} needs at least five unique targets"
