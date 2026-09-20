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


if __name__ == "__main__":
    app()
