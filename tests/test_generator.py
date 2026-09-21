import json
from pathlib import Path
from typing import Self

from PIL import Image

from uiregress.data.generator import generate_dataset
from uiregress.data.renderer import RenderedPair
from uiregress.data.schema import BoundingBox, MutationSpec


class FakeRenderer:
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
        mutation = MutationSpec(
            operator="translate_element",
            label="layout_shift",
            selector='[data-uiregress-target="card"]',
            parameters={"dx": 32, "dy": 0},
        )
        return RenderedPair(
            label="layout_shift",
            mutation=mutation,
            region=BoundingBox(x=10, y=10, width=100, height=50),
            browser={
                "engine": "chromium",
                "browser_version": "test",
                "playwright_version": "test",
            },
        )


def test_generator_writes_versioned_manifest_and_fixture_level_splits(
    tmp_path, monkeypatch
) -> None:
    fixtures = tmp_path / "fixtures"
    fixtures.mkdir()
    for name in ("dashboard", "checkout", "profile"):
        (fixtures / f"{name}.html").write_text("<html></html>", encoding="utf-8")

    monkeypatch.setattr("uiregress.data.generator.PlaywrightRenderer", FakeRenderer)
    output = tmp_path / "dataset"

    summary = generate_dataset(
        fixtures,
        output,
        version="synthetic-test",
        samples_per_fixture=2,
        seed=42,
        width=64,
        height=48,
    )

    assert summary.total_samples == 6
    assert sum(summary.splits.values()) == 6

    lines = [json.loads(line) for line in (output / "manifest.jsonl").read_text().splitlines()]
    fixture_splits: dict[str, set[str]] = {}
    for sample in lines:
        fixture_splits.setdefault(sample["fixture"], set()).add(sample["split"])

    assert all(len(splits) == 1 for splits in fixture_splits.values())
    assert json.loads((output / "dataset.json").read_text())["version"] == "synthetic-test"
