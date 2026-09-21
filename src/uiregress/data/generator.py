from __future__ import annotations

import json
import random
import shutil
from pathlib import Path

from uiregress.data.renderer import PlaywrightRenderer
from uiregress.data.schema import DatasetSummary, SampleManifest
from uiregress.data.splitting import split_fixtures


def _prepare_output(output: Path, *, overwrite: bool) -> None:
    if output.exists() and any(output.iterdir()):
        if not overwrite:
            raise ValueError(
                f"Output directory is not empty: {output}. Pass --overwrite to replace it."
            )
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)


def generate_dataset(
    fixtures_dir: str | Path,
    output_dir: str | Path,
    *,
    version: str = "synthetic-v0.1",
    samples_per_fixture: int = 8,
    seed: int = 42,
    width: int = 1280,
    height: int = 720,
    no_regression_fraction: float = 0.25,
    overwrite: bool = False,
) -> DatasetSummary:
    if samples_per_fixture <= 0:
        raise ValueError("samples_per_fixture must be greater than zero")
    if not 0.0 <= no_regression_fraction <= 1.0:
        raise ValueError("no_regression_fraction must be between 0 and 1")
    if not version.strip():
        raise ValueError("version must not be empty")

    fixtures_path = Path(fixtures_dir)
    output_path = Path(output_dir)
    if not fixtures_path.is_dir():
        raise FileNotFoundError(f"Fixtures directory not found: {fixtures_path}")

    fixtures = sorted(fixtures_path.glob("*.html"))
    if not fixtures:
        raise ValueError(f"No HTML fixtures found in: {fixtures_path}")

    _prepare_output(output_path, overwrite=overwrite)
    images_dir = output_path / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    fixture_splits = split_fixtures((fixture.stem for fixture in fixtures), seed=seed)
    master_rng = random.Random(seed)
    manifest_path = output_path / "manifest.jsonl"
    split_counts = {"train": 0, "validation": 0, "test": 0}
    total_samples = 0

    with PlaywrightRenderer(width=width, height=height) as renderer, manifest_path.open(
        "w", encoding="utf-8"
    ) as manifest_file:
        for fixture in fixtures:
            split = fixture_splits[fixture.stem]
            for index in range(samples_per_fixture):
                sample_seed = master_rng.randrange(0, 2**31)
                sample_rng = random.Random(sample_seed)
                no_regression = sample_rng.random() < no_regression_fraction
                sample_id = f"{fixture.stem}-{index:04d}-{sample_seed:010d}"

                baseline_relative = Path("images") / f"{sample_id}-baseline.png"
                current_relative = Path("images") / f"{sample_id}-current.png"
                rendered = renderer.render_pair(
                    fixture,
                    output_path / baseline_relative,
                    output_path / current_relative,
                    rng=sample_rng,
                    sample_seed=sample_seed,
                    no_regression=no_regression,
                )

                sample = SampleManifest(
                    sample_id=sample_id,
                    fixture=fixture.name,
                    split=split,
                    seed=sample_seed,
                    baseline=baseline_relative.as_posix(),
                    current=current_relative.as_posix(),
                    label=rendered.label,
                    viewport={"width": width, "height": height},
                    region=rendered.region,
                    mutation=rendered.mutation,
                    browser=rendered.browser,
                )
                manifest_file.write(json.dumps(sample.to_dict(), sort_keys=True) + "\n")
                split_counts[split] += 1
                total_samples += 1

    summary = DatasetSummary(
        version=version,
        seed=seed,
        total_samples=total_samples,
        fixtures=len(fixtures),
        samples_per_fixture=samples_per_fixture,
        manifest="manifest.jsonl",
        splits=split_counts,
    )
    (output_path / "dataset.json").write_text(
        json.dumps(summary.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary
