from __future__ import annotations

import json
from dataclasses import asdict

import torch
import typer

from uiregress import __version__
from uiregress.device import resolve_device

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
) -> None:
    """Placeholder for Milestone 1 screenshot comparison."""
    typer.echo(
        json.dumps(
            {
                "status": "not_implemented",
                "baseline": baseline,
                "current": current,
                "next_milestone": "Implement deterministic classical comparison before ML inference.",
            },
            indent=2,
        )
    )
    raise typer.Exit(code=3)


if __name__ == "__main__":
    app()
