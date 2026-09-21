from __future__ import annotations

from typer.testing import CliRunner

from uiregress.cli import app
from uiregress.data.schema import DatasetSummary

runner = CliRunner()


def test_generate_dataset_command_invokes_generator(monkeypatch, tmp_path) -> None:
    fixtures = tmp_path / "fixtures"
    fixtures.mkdir()
    output = tmp_path / "generated"
    called: dict[str, object] = {}

    def fake_generate_dataset(fixtures_dir, output_dir, **kwargs):
        called["fixtures_dir"] = fixtures_dir
        called["output_dir"] = output_dir
        called.update(kwargs)
        return DatasetSummary(
            version="test-v0.1",
            seed=42,
            total_samples=3,
            fixtures=3,
            samples_per_fixture=1,
            manifest="manifest.jsonl",
            splits={"train": 1, "validation": 1, "test": 1},
        )

    monkeypatch.setattr(
        "uiregress.data.generator.generate_dataset",
        fake_generate_dataset,
    )

    result = runner.invoke(
        app,
        [
            "generate-dataset",
            "--fixtures-dir",
            str(fixtures),
            "--output",
            str(output),
            "--version",
            "test-v0.1",
            "--samples-per-fixture",
            "1",
            "--samples-per-class",
            "2",
            "--seed",
            "42",
            "--overwrite",
        ],
    )

    assert result.exit_code == 0, result.output
    assert called["fixtures_dir"] == str(fixtures)
    assert called["output_dir"] == str(output)
    assert called["version"] == "test-v0.1"
    assert called["samples_per_class"] == 2
    assert '"total_samples": 3' in result.output
    assert '"output":' in result.output


def test_train_command_invokes_training(monkeypatch, tmp_path) -> None:
    dataset = tmp_path / "dataset"
    checkpoint = tmp_path / "best.pt"
    dataset.mkdir()
    called: dict[str, object] = {}

    def fake_train_model(config, *, progress=None):
        called["config"] = config
        if progress is not None:
            progress(
                {
                    "epoch": 1,
                    "train": {"loss": 1.0},
                    "validation": {
                        "loss": 0.8,
                        "multiclass": {"macro_f1": 0.5},
                    },
                }
            )
        return {
            "checkpoint": str(checkpoint),
            "device": "cpu",
            "amp_enabled": False,
            "classes": {"no_regression": 0, "layout_shift": 1},
            "best_epoch": 1,
            "best_validation_macro_f1": 0.5,
            "history": [],
            "history_file": str(checkpoint.with_suffix(".history.json")),
        }

    monkeypatch.setattr("uiregress.training.train_model", fake_train_model)

    result = runner.invoke(
        app,
        [
            "train",
            "--dataset",
            str(dataset),
            "--checkpoint",
            str(checkpoint),
            "--epochs",
            "1",
            "--batch-size",
            "2",
            "--device",
            "cpu",
            "--no-pretrained",
        ],
    )

    assert result.exit_code == 0, result.output
    config = called["config"]
    assert config.dataset_root == str(dataset)
    assert config.checkpoint_path == str(checkpoint)
    assert config.epochs == 1
    assert config.batch_size == 2
    assert config.device == "cpu"
    assert config.pretrained is False
    assert '"best_epoch": 1' in result.output
