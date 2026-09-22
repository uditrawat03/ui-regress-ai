import json

from PIL import Image

from uiregress.data.localization import (
    NormalizedBox,
    box_iou,
    box_to_mask,
    build_localization_target,
    center_hit,
    export_localization_targets,
    localization_metrics,
    normalize_region,
)


def test_normalize_region_clamps_to_viewport() -> None:
    box = normalize_region(
        {"x": -10.0, "y": 10.0, "width": 60.0, "height": 40.0},
        {"width": 100, "height": 100},
    )

    assert box == NormalizedBox(x1=0.0, y1=0.1, x2=0.5, y2=0.5)


def test_box_to_mask_rasterizes_target() -> None:
    mask = box_to_mask(NormalizedBox(0.25, 0.25, 0.5, 0.5), size=8)

    assert mask.shape == (1, 8, 8)
    assert mask.sum().item() == 4.0


def test_missing_region_builds_invalid_zero_target() -> None:
    target = build_localization_target(
        {"viewport": {"width": 100, "height": 50}, "region": None},
        mask_size=16,
    )

    assert target.valid is False
    assert target.box is None
    assert target.mask.shape == (1, 16, 16)
    assert target.mask.sum().item() == 0.0


def test_localization_metrics_report_iou_and_center_hit() -> None:
    target = NormalizedBox(0.2, 0.2, 0.6, 0.6)
    exact = NormalizedBox(0.2, 0.2, 0.6, 0.6)
    miss = NormalizedBox(0.7, 0.7, 0.9, 0.9)

    assert box_iou(exact, target) == 1.0
    assert box_iou(miss, target) == 0.0
    assert center_hit(exact, target) is True
    assert center_hit(miss, target) is False

    metrics = localization_metrics([exact, miss], [target, target])
    assert metrics == {"samples": 2, "mean_iou": 0.5, "center_hit_rate": 0.5}


def test_export_localization_targets_writes_overlay_mask_and_summary(tmp_path) -> None:
    dataset = tmp_path / "dataset"
    images = dataset / "images"
    images.mkdir(parents=True)
    current = images / "sample-current.png"
    baseline = images / "sample-baseline.png"
    Image.new("RGB", (100, 50), "white").save(current)
    Image.new("RGB", (100, 50), "white").save(baseline)

    record = {
        "sample_id": "sample",
        "fixture": "fixture.html",
        "split": "train",
        "baseline": "images/sample-baseline.png",
        "current": "images/sample-current.png",
        "label": "layout_shift",
        "viewport": {"width": 100, "height": 50},
        "region": {"x": 20.0, "y": 10.0, "width": 40.0, "height": 20.0},
    }
    (dataset / "manifest.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")

    output = tmp_path / "localization"
    summary = export_localization_targets(dataset, output, split="train", limit=1, mask_size=16)

    assert summary["exported_samples"] == 1
    assert summary["localizable_samples"] == 1
    assert (output / "summary.json").is_file()
    assert (output / "overlays" / "sample.png").is_file()
    assert (output / "masks" / "sample.png").is_file()
