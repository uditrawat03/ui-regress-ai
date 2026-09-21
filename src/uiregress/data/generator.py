from __future__ import annotations

import asyncio
import json
import random
import shutil
from collections.abc import Iterator
from pathlib import Path

from uiregress.data.mutations import ALL_LABELS
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


def _sample_plan(
    *,
    master_rng: random.Random,
    samples_per_fixture: int,
    samples_per_class: int | None,
) -> Iterator[tuple[int, int, bool, str | None, str | None]]:
    """Yield sample index, seed, no-regression flag, forced label, expected label."""
    if samples_per_class is None:
        for index in range(samples_per_fixture):
            sample_seed = master_rng.randrange(0, 2**31)
            yield index, sample_seed, False, None, None
        return

    index = 0
    for label in ALL_LABELS:
        for _ in range(samples_per_class):
            sample_seed = master_rng.randrange(0, 2**31)
            no_regression = label == "no_regression"
            forced_label = None if no_regression else label
            yield index, sample_seed, no_regression, forced_label, label
            index += 1


async def _generate_dataset_async(
    fixtures_dir: str | Path,
    output_dir: str | Path,
    *,
    version: str,
    samples_per_fixture: int,
    samples_per_class: int | None,
    seed: int,
    width: int,
    height: int,
    no_regression_fraction: float,
    overwrite: bool,
) -> DatasetSummary:
    if samples_per_fixture <= 0:
        raise ValueError("samples_per_fixture must be greater than zero")
    if samples_per_class is not None and samples_per_class <= 0:
        raise ValueError("samples_per_class must be greater than zero")
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
    class_distribution = {label: 0 for label in ALL_LABELS}
    split_class_distribution = {
        split: {label: 0 for label in ALL_LABELS}
        for split in ("train", "validation", "test")
    }
    total_samples = 0
    generation_mode = "balanced" if samples_per_class is not None else "random"
    effective_samples_per_fixture = (
        len(ALL_LABELS) * samples_per_class
        if samples_per_class is not None
        else samples_per_fixture
    )

    async with PlaywrightRenderer(width=width, height=height) as renderer:
        with manifest_path.open("w", encoding="utf-8") as manifest_file:
            for fixture in fixtures:
                split = fixture_splits[fixture.stem]
                plan = _sample_plan(
                    master_rng=master_rng,
                    samples_per_fixture=samples_per_fixture,
                    samples_per_class=samples_per_class,
                )
                for index, sample_seed, balanced_negative, forced_label, expected_label in plan:
                    sample_rng = random.Random(sample_seed)
                    no_regression = (
                        balanced_negative
                        if samples_per_class is not None
                        else sample_rng.random() < no_regression_fraction
                    )
                    label_token = expected_label or "random"
                    sample_id = (
                        f"{fixture.stem}-{label_token}-{index:04d}-{sample_seed:010d}"
                        if samples_per_class is not None
                        else f"{fixture.stem}-{index:04d}-{sample_seed:010d}"
                    )

                    baseline_relative = Path("images") / f"{sample_id}-baseline.png"
                    current_relative = Path("images") / f"{sample_id}-current.png"
                    rendered = await renderer.render_pair(
                        fixture,
                        output_path / baseline_relative,
                        output_path / current_relative,
                        rng=sample_rng,
                        sample_seed=sample_seed,
                        no_regression=no_regression,
                        regression_label=forced_label,
                    )
                    if expected_label is not None and rendered.label != expected_label:
                        raise RuntimeError(
                            "Balanced generation produced the wrong label: "
                            f"expected {expected_label}, got {rendered.label}."
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
                    manifest_file.flush()
                    split_counts[split] += 1
                    class_distribution.setdefault(rendered.label, 0)
                    class_distribution[rendered.label] += 1
                    split_class_distribution[split].setdefault(rendered.label, 0)
                    split_class_distribution[split][rendered.label] += 1
                    total_samples += 1

    summary = DatasetSummary(
        version=version,
        seed=seed,
        total_samples=total_samples,
        fixtures=len(fixtures),
        samples_per_fixture=effective_samples_per_fixture,
        manifest="manifest.jsonl",
        splits=split_counts,
        generation_mode=generation_mode,
        samples_per_class=samples_per_class,
        class_distribution=class_distribution,
        split_class_distribution=split_class_distribution,
    )
    (output_path / "dataset.json").write_text(
        json.dumps(summary.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def generate_dataset(
    fixtures_dir: str | Path,
    output_dir: str | Path,
    *,
    version: str = "synthetic-v0.1",
    samples_per_fixture: int = 8,
    samples_per_class: int | None = None,
    seed: int = 42,
    width: int = 1280,
    height: int = 720,
    no_regression_fraction: float = 0.25,
    overwrite: bool = False,
) -> DatasetSummary:
    """Generate a dataset using Playwright's async API.

    ``samples_per_class`` enables balanced generation and guarantees the same
    number of examples for every semantic label on every fixture. When omitted,
    the original random sampling behavior is preserved.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        pass
    else:
        raise RuntimeError(
            "generate_dataset() cannot run inside an active asyncio event loop. "
            "Call _generate_dataset_async() from async code instead."
        )

    return asyncio.run(
        _generate_dataset_async(
            fixtures_dir,
            output_dir,
            version=version,
            samples_per_fixture=samples_per_fixture,
            samples_per_class=samples_per_class,
            seed=seed,
            width=width,
            height=height,
            no_regression_fraction=no_regression_fraction,
            overwrite=overwrite,
        )
    )
