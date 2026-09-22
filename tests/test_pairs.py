import json

from PIL import Image

from uiregress.data.pairs import PairedScreenshotDataset, build_label_mapping


def _write_sample(
    root,
    *,
    sample_id: str,
    label: str,
    split: str,
    region: dict[str, float] | None = None,
) -> None:
    images = root / "images"
    images.mkdir(exist_ok=True)
    baseline = images / f"{sample_id}-baseline.png"
    current = images / f"{sample_id}-current.png"
    Image.new("RGB", (32, 24), "white").save(baseline)
    Image.new("RGB", (32, 24), "black").save(current)
    record = {
        "sample_id": sample_id,
        "baseline": baseline.relative_to(root).as_posix(),
        "current": current.relative_to(root).as_posix(),
        "label": label,
        "split": split,
        "viewport": {"width": 32, "height": 24},
        "region": region,
    }
    with (root / "manifest.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")


def test_label_mapping_is_stable_and_no_regression_is_zero(tmp_path) -> None:
    _write_sample(tmp_path, sample_id="a", label="layout_shift", split="train")
    _write_sample(tmp_path, sample_id="b", label="no_regression", split="validation")
    _write_sample(tmp_path, sample_id="c", label="element_missing", split="test")

    assert build_label_mapping(tmp_path) == {
        "no_regression": 0,
        "element_missing": 1,
        "layout_shift": 2,
    }


def test_paired_dataset_returns_training_tensors(tmp_path) -> None:
    _write_sample(tmp_path, sample_id="a", label="no_regression", split="train")
    _write_sample(tmp_path, sample_id="b", label="layout_shift", split="train")

    mapping = build_label_mapping(tmp_path)
    dataset = PairedScreenshotDataset(
        tmp_path,
        split="train",
        label_to_index=mapping,
        input_size=64,
    )

    first = dataset[0]
    second = dataset[1]

    assert first["baseline"].shape == (3, 64, 64)
    assert first["current"].shape == (3, 64, 64)
    assert first["binary_target"].item() == 0.0
    assert second["binary_target"].item() == 1.0
    assert second["class_target"].item() == mapping["layout_shift"]
    assert first["localization_valid"].item() is False
    assert first["localization_box"].shape == (4,)
    assert first["localization_mask"].shape == (1, 56, 56)
    assert first["localization_mask"].sum().item() == 0.0


def test_paired_dataset_exposes_localization_target_for_regression(tmp_path) -> None:
    _write_sample(
        tmp_path,
        sample_id="localized",
        label="layout_shift",
        split="train",
        region={"x": 8.0, "y": 6.0, "width": 8.0, "height": 6.0},
    )

    dataset = PairedScreenshotDataset(tmp_path, split="train", input_size=64)
    sample = dataset[0]

    assert sample["localization_valid"].item() is True
    assert sample["localization_box"].tolist() == [0.25, 0.25, 0.5, 0.5]
    assert sample["localization_mask"].sum().item() > 0.0
