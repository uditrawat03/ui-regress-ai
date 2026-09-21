from __future__ import annotations

import json
from dataclasses import asdict

import torch
import typer

from uiregress import __version__
from uiregress.device import resolve_device
from uiregress.inference import compare_images

app = typer.Typer(help="Semantic visual regression testing with PyTorch.")


@app.command()
def info() -> None:
    """Print package and PyTorch information."""
    payload = {
        "uiregress_version": __version__,
        "torch_version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
    }
    typer.echo(json.dumps(payload, indent=2))


@app.command()
def device(
    requested: str = typer.Option("auto", "--device", help="auto, cpu, or cuda"),
) -> None:
    """Show which compute device UIRegressAI would use."""
    try:
        info = resolve_device(requested)
    except (ValueError, RuntimeError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=2) from exc

    typer.echo(json.dumps(asdict(info), indent=2))


@app.command()
def compare(
    baseline: str,
    current: str,
    device: str = typer.Option("auto", "--device", help="auto, cpu, or cuda"),
    changed_pixel_threshold: float = typer.Option(
        0.05,
        "--changed-pixel-threshold",
        help="Per-pixel mean RGB difference required to count as changed.",
    ),
    fail_changed_ratio: float = typer.Option(
        0.01,
        "--fail-changed-ratio",
        help="Fail when at least this fraction of pixels changed.",
    ),
    fail_ssim_below: float = typer.Option(
        0.99,
        "--fail-ssim-below",
        help="Fail when structural similarity is at or below this threshold.",
    ),
    heatmap: str | None = typer.Option(
        None,
        "--heatmap",
        help="Optional output path for a grayscale difference heatmap.",
    ),
) -> None:
    """Compare two screenshots with deterministic non-ML metrics."""
    try:
        result = compare_images(
            baseline,
            current,
            device=device,
            changed_pixel_threshold=changed_pixel_threshold,
            fail_changed_ratio=fail_changed_ratio,
            fail_ssim_below=fail_ssim_below,
            heatmap_path=heatmap,
        )
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=2) from exc

    typer.echo(json.dumps(result.to_dict(), indent=2))
    if result.different:
        raise typer.Exit(code=1)


@app.command("generate-dataset")
def generate_dataset_command(
    fixtures_dir: str = typer.Option(
        "fixtures", "--fixtures-dir", help="Directory containing reusable HTML fixtures."
    ),
    output: str = typer.Option(
        "data/generated/synthetic-v0.1", "--output", help="Dataset output directory."
    ),
    version: str = typer.Option("synthetic-v0.1", "--version"),
    samples_per_fixture: int = typer.Option(8, "--samples-per-fixture", min=1),
    seed: int = typer.Option(42, "--seed"),
    width: int = typer.Option(1280, "--width", min=1),
    height: int = typer.Option(720, "--height", min=1),
    no_regression_fraction: float = typer.Option(
        0.25,
        "--no-regression-fraction",
        min=0.0,
        max=1.0,
        help="Fraction of generated pairs that contain only tolerated pixel noise.",
    ),
    overwrite: bool = typer.Option(False, "--overwrite"),
) -> None:
    """Generate a versioned paired screenshot dataset with known regressions."""
    try:
        from uiregress.data.generator import generate_dataset

        summary = generate_dataset(
            fixtures_dir,
            output,
            version=version,
            samples_per_fixture=samples_per_fixture,
            seed=seed,
            width=width,
            height=height,
            no_regression_fraction=no_regression_fraction,
            overwrite=overwrite,
        )
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=2) from exc

    typer.echo(json.dumps(summary.to_dict(), indent=2))


if __name__ == "__main__":
    app()
