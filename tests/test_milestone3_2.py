from pathlib import Path

from uiregress.training import (
    _calibrate_threshold,
    _multiclass_checkpoint_path,
    _multiclass_metrics,
    _operational_selection_key,
)


def test_threshold_calibration_prefers_low_fpr_while_meeting_recall() -> None:
    targets = [0, 0, 0, 1, 1, 1, 1]
    scores = [0.05, 0.10, 0.35, 0.40, 0.70, 0.80, 0.90]

    calibration = _calibrate_threshold(targets, scores, minimum_recall=0.75)

    assert calibration["constraint_met"] is True
    assert calibration["metrics"]["recall"] >= 0.75
    assert calibration["metrics"]["false_positive_rate"] == 0.0
    assert calibration["threshold"] == 0.40


def test_operational_selection_prioritizes_lower_fpr_over_macro_f1() -> None:
    low_fpr = {
        "constraint_met": True,
        "metrics": {
            "recall": 0.91,
            "false_positive_rate": 0.10,
            "f1": 0.90,
        },
    }
    high_fpr = {
        "constraint_met": True,
        "metrics": {
            "recall": 1.0,
            "false_positive_rate": 0.80,
            "f1": 0.92,
        },
    }

    assert _operational_selection_key(low_fpr, macro_f1=0.45) > _operational_selection_key(
        high_fpr,
        macro_f1=0.70,
    )


def test_multiclass_metrics_include_confusion_matrix() -> None:
    metrics = _multiclass_metrics(
        [0, 0, 1, 1, 2],
        [0, 1, 1, 2, 2],
        index_to_label={0: "no_regression", 1: "layout_shift", 2: "unexpected_style"},
    )

    assert metrics["confusion_matrix"] == {
        "labels": ["no_regression", "layout_shift", "unexpected_style"],
        "matrix": [
            [1, 1, 0],
            [0, 1, 1],
            [0, 0, 1],
        ],
    }


def test_multiclass_checkpoint_is_sibling_of_operational_checkpoint() -> None:
    path = Path("artifacts/checkpoints/milestone3-v0.2-best.pt")

    assert _multiclass_checkpoint_path(path) == Path(
        "artifacts/checkpoints/milestone3-v0.2-best.multiclass.pt"
    )
