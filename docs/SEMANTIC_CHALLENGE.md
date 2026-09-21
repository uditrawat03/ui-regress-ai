# Milestone 3.3 — Semantic Challenge Dataset

Milestone 3.2 showed that `changed_area_ratio` could solve `synthetic-v0.2` almost perfectly.
That means the dataset was still dominated by pixel magnitude instead of semantic meaning.

Milestone 3.3 changes both the dataset and the regression-type objective.

## Conditional type head

The binary head answers:

```text
regression / no regression
```

The type head is trained only when the sample is a regression and predicts one of:

```text
element_missing
incorrect_size
layout_shift
text_clipping
unexpected_style
```

`no_regression` is not a type-head class. Checkpoints record the five-class mapping separately from
the six-label dataset mapping.

## Semantic-challenge profile

Generate v0.3 with:

```powershell
uv run uiregress generate-dataset `
  --fixtures-dir .\fixtures `
  --output .\data\generated\synthetic-v0.3 `
  --version synthetic-v0.3 `
  --samples-per-class 10 `
  --profile semantic-challenge `
  --seed 42 `
  --overwrite
```

The profile keeps fixture-level train/validation/test isolation and balanced labels, but changes how
pairs are rendered.

Regression samples prefer smaller mutation targets and use subtler parameter ranges:

- layout shifts use only 2–4 pixel displacement
- incorrect-size changes use roughly 3–8 percent width changes
- clipping removes only a small portion of the target
- style-token changes use less dramatic colors
- missing-element cases prefer smaller eligible elements

No-regression samples use deterministic hard-negative raster variations that change pixels without
changing page semantics:

- stronger pixel noise
- small global brightness drift
- one-pixel raster offset
- mild JPEG recompression artifacts

These negatives intentionally overlap the pixel-distance range of subtle regressions. Classical
thresholds must therefore trade recall against false positives.

## Train and benchmark

```powershell
uv run uiregress train `
  --dataset .\data\generated\synthetic-v0.3 `
  --checkpoint .\artifacts\checkpoints\milestone3-v0.3-best.pt `
  --epochs 8 `
  --batch-size 8 `
  --device cuda `
  --minimum-binary-recall 0.90
```

Keep the classical benchmark enabled. Threshold calibration remains validation-only and the test split
is still evaluated only after checkpoint and threshold selection.

Milestone 3 closes only if the learned detector adds measurable value over at least one meaningful
classical baseline on the frozen semantic-challenge test split.
