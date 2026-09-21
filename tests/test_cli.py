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
            "--seed",
            "42",
            "--overwrite",
        ],
    )

    assert result.exit_code == 0, result.output
    assert called["fixtures_dir"] == str(fixtures)
    assert called["output_dir"] == str(output)
    assert called["version"] == "test-v0.1"
    assert '"total_samples": 3' in result.output
    assert '"output":' in result.output
