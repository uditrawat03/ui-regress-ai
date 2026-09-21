from uiregress.data.schema import BoundingBox, MutationSpec, SampleManifest


def test_bounding_box_union_contains_before_and_after() -> None:
    before = BoundingBox(x=10, y=20, width=100, height=50)
    after = BoundingBox(x=40, y=5, width=120, height=80)

    region = before.union(after)

    assert region == BoundingBox(x=10, y=5, width=150, height=80)


def test_sample_manifest_serializes_nested_dataclasses() -> None:
    sample = SampleManifest(
        sample_id="fixture-0001",
        fixture="fixture.html",
        split="train",
        seed=42,
        baseline="images/baseline.png",
        current="images/current.png",
        label="layout_shift",
        viewport={"width": 1280, "height": 720},
        region=BoundingBox(x=1, y=2, width=3, height=4),
        mutation=MutationSpec(
            operator="translate_element",
            label="layout_shift",
            selector='[data-uiregress-target="card"]',
            parameters={"dx": 32, "dy": 0},
        ),
        browser={"engine": "chromium", "browser_version": "1", "playwright_version": "1"},
    )

    payload = sample.to_dict()

    assert payload["region"]["width"] == 3
    assert payload["mutation"]["operator"] == "translate_element"
