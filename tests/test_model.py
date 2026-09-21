import torch

from uiregress.models import SiameseRegressionClassifier


def test_siamese_model_has_binary_and_multiclass_heads() -> None:
    model = SiameseRegressionClassifier(num_classes=6, pretrained=False)
    model.eval()

    baseline = torch.randn(2, 3, 64, 64)
    current = torch.randn(2, 3, 64, 64)

    with torch.inference_mode():
        output = model(baseline, current)

    assert output.binary_logits.shape == (2,)
    assert output.class_logits.shape == (2, 6)
