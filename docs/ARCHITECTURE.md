# Architecture

## High-level pipeline

```text
                         ┌────────────────────┐
                         │ baseline screenshot│
                         └─────────┬──────────┘
                                   │
                                   ▼
                          image normalization
                                   │
                     ┌─────────────┴─────────────┐
                     │                           │
                     ▼                           ▼
              classical diff              shared encoder
                     │                           │
                     │                    baseline features
                     │                           │
┌────────────────────┐                           │
│ current screenshot │────────► normalization ──►shared encoder
└────────────────────┘                           │
                                         current features
                                                │
                                                ▼
                                      feature comparison
                                                │
                  ┌─────────────────────────────┼──────────────────────────┐
                  ▼                             ▼                          ▼
          binary regression head       class prediction head      localization head
                  │                             │                          │
                  └─────────────────────────────┼──────────────────────────┘
                                                ▼
                                         decision engine
                                                │
                                      JSON / CLI / CI report
```

## Layer 1: Input pipeline

Responsibilities:

- decode PNG/JPEG/WebP
- normalize color mode
- validate image sizes
- resize or tile according to inference policy
- preserve original coordinates for region mapping

The preprocessing configuration must be versioned because changes can alter model output.

## Layer 2: Classical baseline

Before deep learning, implement measurable baselines:

- absolute RGB difference
- thresholded changed-area ratio
- SSIM-style similarity
- optional edge-map comparison

Reasons:

- provides a sanity check
- catches data-pipeline bugs
- creates a benchmark the learned model must beat
- allows hybrid decision rules

## Layer 3: Shared PyTorch encoder

Initial candidate:

- ConvNeXt Tiny or ResNet-50 transfer learning

Later candidates:

- ViT backbone
- DINO-style pretrained features
- multi-scale encoder

The first model should be intentionally simple enough to debug.

## Layer 4: Pair comparison

For baseline feature tensor `F_b` and current feature tensor `F_c`, candidate comparison signals are:

```text
abs(F_b - F_c)
F_b * F_c
concat(F_b, F_c)
cosine distance
```

The first implementation should prefer simple absolute difference plus concatenation.

## Layer 5: Prediction heads

### Binary head

Answers whether a meaningful regression exists.

### Multi-class head

Predicts regression type.

### Localization head

Produces a coarse heatmap or bounding box identifying the affected region.

The localization task can start as coarse segmentation. Precise object detection is not required for the MVP.

## Layer 6: Decision engine

The neural model should not directly decide whether CI fails.

The decision engine combines:

- classifier confidence
- localization confidence
- changed-area ratio
- optional DOM evidence
- repository policy thresholds

Example policy:

```yaml
fail_when:
  confidence_gte: 0.90
  classes:
    - element_overlap
    - text_clipping
    - element_missing
```

## Layer 7: Interfaces

### CLI

Primary MVP interface.

### Python package

Allows direct integration into test systems.

### REST API

Later milestone, likely FastAPI.

### GitHub Action

Later milestone after CLI output stabilizes.

## Model artifact contract

Every checkpoint should store or reference:

- architecture name
- class mapping
- preprocessing version
- input resolution
- training dataset version
- git commit
- PyTorch version
- CUDA version where applicable
- validation metrics

## Configuration

Configuration should be explicit and serializable. Avoid hidden global settings.

Planned config groups:

```text
training
model
dataset
preprocessing
inference
evaluation
ci_policy
```

## Error handling

Inference should fail clearly for:

- missing files
- corrupted image
- unsupported format
- invalid checkpoint
- incompatible model metadata
- mismatched preprocessing version
- CUDA requested but unavailable

Automatic CPU fallback should only occur when the caller permits it.
