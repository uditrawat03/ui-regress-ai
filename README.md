# UIRegressAI

GPU-powered semantic visual regression testing with PyTorch.

Traditional screenshot tests often fail because pixels changed, not because the UI is actually broken. UIRegressAI aims to distinguish harmless rendering noise from meaningful regressions such as clipped text, missing elements, overlap, layout shifts, and responsive failures.

## What we are building

Given a baseline screenshot and a current screenshot, UIRegressAI will eventually return:

- whether a meaningful regression is present
- the regression type
- a confidence score
- the affected region as a heatmap or bounding box
- machine-readable JSON for CI
- a human-readable explanation for pull requests

Example target output:

```text
Regression: detected
Type: element_overlap
Confidence: 0.96
Region: x=612 y=433 width=280 height=94
Severity: high
Message: Checkout action overlaps the payment form.
```

## Why PyTorch + GPU

This project is not adding GPU support as decoration. The intended system compares batches of high-resolution screenshots using learned visual features. Training the model, running large-scale feature extraction, generating heatmaps, and processing CI screenshot suites are workloads that benefit directly from CUDA acceleration.

## Core idea

```text
Baseline screenshot ──► shared vision encoder ─┐
                                               ├─► feature difference ─► heads
Current screenshot  ──► shared vision encoder ─┘

Heads:
  • regression / no-regression
  • regression class
  • affected-region localization
  • confidence
```

A later hybrid inference pipeline will combine:

```text
pixel / SSIM signal
        +
PyTorch semantic signal
        +
optional DOM metadata
        ↓
final regression decision
```

## Initial regression classes

1. `no_regression`
2. `layout_shift`
3. `element_missing`
4. `element_overlap`
5. `text_clipping`
6. `incorrect_size`
7. `unexpected_style`
8. `image_missing`
9. `responsive_layout_failure`

## Repository status

The repository currently contains the project specification, architecture, roadmap, dataset plan, evaluation plan, GPU strategy, and starter Python package. The ML model is intentionally not presented as completed.

## Project structure

```text
ui-regress-ai/
├── docs/
│   ├── ARCHITECTURE.md
│   ├── DATASET.md
│   ├── EVALUATION.md
│   ├── GPU_STRATEGY.md
│   ├── MILESTONES.md
│   ├── PRODUCT_SPEC.md
│   └── ROADMAP.md
├── src/uiregress/
│   ├── data/
│   ├── inference/
│   ├── models/
│   ├── cli.py
│   ├── config.py
│   └── device.py
├── tests/
├── scripts/
├── .github/workflows/
├── pyproject.toml
├── Dockerfile
└── README.md
```

## Local setup

Target path on Windows:

```powershell
C:\working\Projects\Python\ui-regress-ai
```

This project uses `uv` for dependency and environment management:

```powershell
cd C:\working\Projects\Python\ui-regress-ai
uv sync --all-extras
```

Check available acceleration:

```powershell
uv run uiregress device
```

The repository pins PyTorch to the CUDA 13.2 wheel index on Windows/Linux. A working NVIDIA setup should report:

```text
"selected": "cuda"
"cuda_available": true
```

Verify the complete environment with:

```powershell
uv run python scripts\check_environment.py
```

## CLI

Runtime information:

```powershell
uv run uiregress device
uv run uiregress info
```

Compare two screenshots:

```powershell
uv run uiregress compare baseline.png current.png
```

Write a visual difference heatmap:

```powershell
uv run uiregress compare baseline.png current.png --heatmap artifacts\diff.png
```

Exit code `0` means the baseline thresholds did not detect a meaningful difference. Exit code `1` means the comparison crossed the configured changed-area or SSIM threshold. Exit code `2` means the input or runtime configuration was invalid.

## Development principles

- Measure semantic regressions, not arbitrary pixel changes.
- Keep a deterministic classical baseline before adding neural complexity.
- Synthetic training data must remain reproducible and labeled by construction.
- Separate model confidence from final CI policy.
- Benchmark CPU and GPU paths independently.
- Never claim production readiness from synthetic-only evaluation.
- Store enough metadata to reproduce every training and evaluation run.

## Definition of MVP

The first useful version is complete when it can:

- generate paired screenshots containing known UI mutations
- train a PyTorch baseline on those pairs
- run inference on CUDA when available
- distinguish `no_regression` from at least four real regression categories
- localize the changed region at a useful coarse resolution
- expose comparison through a CLI
- emit deterministic JSON suitable for CI
- report precision, recall, F1, confusion matrix, and false-positive rate

See [docs/MILESTONES.md](docs/MILESTONES.md) for the build order.

## Non-goals for the first release

The first release will not try to:

- understand arbitrary UX quality
- replace accessibility testing
- replace browser functional tests
- auto-approve production deployments
- use an LLM to invent explanations without grounded visual evidence
- promise cross-browser perfection before benchmark data supports it

## License

No license has been selected yet. Add one before accepting external contributions or publishing a release.
