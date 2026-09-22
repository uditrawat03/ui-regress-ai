# Localization Targets

Milestone 4.1 turns the bounding regions already stored in generated manifests into deterministic
training and evaluation targets. Existing datasets do not need to be regenerated.

## Target contract

Every `PairedScreenshotDataset` sample now contains:

- `localization_valid`: boolean tensor indicating whether a usable region exists
- `localization_box`: normalized `[x1, y1, x2, y2]` tensor in viewport coordinates
- `localization_mask`: `1 x 56 x 56` binary mask by default

The target representation is versioned as:

```text
normalized-xyxy-mask-v1
```

Regression regions are clamped to the screenshot viewport before normalization. This keeps targets
valid when a translated or resized element extends partly outside the captured image.

For `no_regression` samples, or legacy manifest records without a region:

```text
localization_valid = false
localization_box   = [0, 0, 0, 0]
localization_mask  = all zeros
```

The mask size can be changed when constructing `PairedScreenshotDataset` with
`localization_size=<size>`.

## Inspect targets visually

Use the existing `synthetic-v0.3` dataset:

```powershell
uv run uiregress localization-targets `
  --dataset .\data\generated\synthetic-v0.3 `
  --split validation `
  --output .\artifacts\localization-targets\validation `
  --limit 12 `
  --mask-size 56
```

The command writes:

```text
artifacts/localization-targets/validation/
├── summary.json
├── overlays/
│   └── <sample-id>.png
└── masks/
    └── <sample-id>.png
```

`overlays/` shows the manifest region on the current screenshot. `masks/` contains the coarse
binary target that can be consumed by a future localization head.

## Metric infrastructure

Milestone 4.1 provides two box-level metrics:

- mean intersection-over-union (IoU)
- center-point hit rate

The center-point metric counts a hit when the center of a predicted box falls inside the
corresponding ground-truth box.

These utilities validate the target/evaluation contract now. Milestone 4.2 will add model-produced
localization predictions; only then should held-out localization IoU be reported as a model metric.

## Why no dataset regeneration is required

The synthetic generator already stores each regression's affected `region` and screenshot
`viewport` in `manifest.jsonl`. Milestone 4.1 derives normalized boxes and masks from those values
at load time, preserving the frozen v0.3 benchmark images and split assignments.
