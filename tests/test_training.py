from uiregress.training import _binary_metrics, _multiclass_metrics


def test_binary_metrics_include_false_positive_rate() -> None:
    metrics = _binary_metrics([0, 0, 1, 1], [0, 1, 1, 1])

    assert metrics["accuracy"] == 0.75
    assert metrics["recall"] == 1.0
    assert metrics["false_positive_rate"] == 0.5


def test_multiclass_metrics_report_per_class_and_macro_f1() -> None:
    metrics = _multiclass_metrics(
        [0, 1, 1, 2],
        [0, 1, 2, 2],
        index_to_label={0: "no_regression", 1: "layout_shift", 2: "element_missing"},
    )

    assert metrics["accuracy"] == 0.75
    assert set(metrics["per_class"]) == {
        "no_regression",
        "layout_shift",
        "element_missing",
    }
    assert 0.0 <= metrics["macro_f1"] <= 1.0
