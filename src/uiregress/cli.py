from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

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
    samples_per_class: int | None = typer.Option(
        None,
        "--samples-per-class",
        min=1,
        help=(
            "Generate an equal number of samples for every semantic label on every "
            "fixture. Overrides random class sampling when set."
        ),
    ),
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
    fixtures_path = Path(fixtures_dir).resolve()
    output_path = Path(output).resolve()
    typer.echo(
        f"Generating dataset from {fixtures_path} -> {output_path}",
        err=True,
    )

    try:
        from uiregress.data.generator import generate_dataset

        summary = generate_dataset(
            fixtures_dir,
            output,
            version=version,
            samples_per_fixture=samples_per_fixture,
            samples_per_class=samples_per_class,
            seed=seed,
            width=width,
            height=height,
            no_regression_fraction=no_regression_fraction,
            overwrite=overwrite,
        )
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=2) from exc

    payload = summary.to_dict()
    payload["output"] = str(output_path)
    typer.echo(json.dumps(payload, indent=2))


@app.command("train")
def train_command(
    dataset: str = typer.Option(
        "data/generated/synthetic-v0.1",
        "--dataset",
        help="Generated dataset root containing dataset.json and manifest.jsonl.",
    ),
    checkpoint: str = typer.Option(
        "artifacts/checkpoints/milestone3-best.pt",
        "--checkpoint",
        help="Path for the best validation checkpoint.",
    ),
    epochs: int = typer.Option(5, "--epochs", min=1),
    batch_size: int = typer.Option(8, "--batch-size", min=1),
    learning_rate: float = typer.Option(1e-4, "--learning-rate", min=1e-9),
    input_size: int = typer.Option(224, "--input-size", min=32),
    num_workers: int = typer.Option(0, "--num-workers", min=0),
    seed: int = typer.Option(42, "--seed"),
    device: str = typer.Option("auto", "--device", help="auto, cpu, or cuda"),
    amp: bool = typer.Option(True, "--amp/--no-amp", help="Use CUDA automatic mixed precision."),
    pretrained: bool = typer.Option(
        True,
        "--pretrained/--no-pretrained",
        help="Initialize the shared ResNet-18 encoder with ImageNet weights.",
    ),
) -> None:
    """Train the first dual-head Siamese visual-regression model."""
    try:
        from uiregress.training import TrainingConfig, train_model

        config = TrainingConfig(
            dataset_root=dataset,
            checkpoint_path=checkpoint,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            input_size=input_size,
            num_workers=num_workers,
            seed=seed,
            device=device,
            amp=amp,
            pretrained=pretrained,
        )

        def report_epoch(payload: dict[str, object]) -> None:
            validation = payload["validation"]
            if not isinstance(validation, dict):
                return
            multiclass = validation.get("multiclass", {})
            macro_f1 = multiclass.get("macro_f1") if isinstance(multiclass, dict) else None
            typer.echo(
                f"epoch={payload['epoch']} "
                f"train_loss={payload['train']['loss']:.4f} "
                f"val_loss={validation['loss']:.4f} "
                f"val_macro_f1={float(macro_f1):.4f}",
                err=True,
            )

        summary = train_model(config, progress=report_epoch)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=2) from exc

    typer.echo(json.dumps(summary, indent=2))


def main() -> None:
    """Run the UIRegressAI command-line application."""
    app()


if __name__ == "__main__":
    main()
