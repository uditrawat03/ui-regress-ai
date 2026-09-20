from uiregress.inference import RegressionResult


def test_result_serializes() -> None:
    result = RegressionResult(regression=True, label="element_overlap", confidence=0.9)
    assert result.to_dict()["label"] == "element_overlap"
