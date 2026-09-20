# Product Specification

## Problem

Visual regression tools commonly compare screenshots at the pixel level. This creates two recurring problems:

1. harmless rendering differences can fail a test
2. meaningful UI failures can be hard to classify and explain

Examples of meaningful failures include:

- a button overlapping another control
- text clipped by a container
- an image disappearing
- an element moving enough to break hierarchy
- a mobile layout overflowing the viewport
- an incorrect width or height causing downstream layout damage

## Objective

Build an open-source PyTorch system that compares two rendered UI screenshots and estimates whether the change is a meaningful regression.

The system should output structured evidence rather than a single similarity number.

## Primary users

- frontend engineers
- QA engineers
- design-system teams
- platform teams running CI
- open-source maintainers

## Input

Required:

- baseline screenshot
- current screenshot

Optional later inputs:

- browser and viewport metadata
- DOM bounding boxes
- accessibility tree snapshot
- changed source files

## Output contract

```json
{
  "regression": true,
  "class": "element_overlap",
  "confidence": 0.962,
  "severity": "high",
  "regions": [
    {
      "x": 612,
      "y": 433,
      "width": 280,
      "height": 94,
      "score": 0.91
    }
  ],
  "metrics": {
    "pixel_change_ratio": 0.041,
    "semantic_distance": 0.73
  }
}
```

## Functional requirements

### FR-1 Pair validation

The system validates dimensions, image mode, corrupted files, and unsupported inputs before inference.

### FR-2 Deterministic baseline

A classical image-difference baseline must be implemented before the learned model. It provides a benchmark and fallback.

### FR-3 Semantic classifier

The PyTorch model classifies meaningful regression categories.

### FR-4 Localization

The system highlights the image area responsible for the prediction.

### FR-5 GPU acceleration

Training and inference use CUDA automatically when available, with an explicit CPU fallback.

### FR-6 CLI

The core system works without a server.

### FR-7 Machine-readable output

CI integrations consume a stable JSON schema.

### FR-8 Reproducibility

Training runs store configuration, seed, dataset version, model version, metrics, and checkpoint metadata.

## Quality requirements

The project will optimize for false positives as a first-class metric. A visual regression tool that frequently blocks correct pull requests will not be trusted even if aggregate accuracy is high.

We will measure:

- per-class precision
- per-class recall
- macro F1
- false-positive rate on no-regression pairs
- localization IoU
- calibration error
- CPU latency
- GPU latency
- peak GPU memory

## Success criteria for v0.1

v0.1 is successful when:

- dataset generation is reproducible
- at least 5 regression labels are supported including `no_regression`
- GPU training works from a clean setup
- inference works on CPU and CUDA
- evaluation scripts produce a versioned report
- CLI returns stable JSON
- synthetic holdout results are clearly separated from real-world benchmark results

## Risks

### Synthetic-to-real gap

Synthetic CSS mutations may be easier to detect than real regressions. Mitigation: maintain a real-world benchmark and never report synthetic scores as real-world performance.

### Screenshot shortcuts

The model may learn generator artifacts rather than UI defects. Mitigation: multiple mutation strategies, renderer diversity, randomized content, and out-of-generator validation.

### Browser rendering noise

Different fonts, subpixel rendering, and device scale factors can produce irrelevant differences. Mitigation: augmentations, tolerance baselines, and environment-aware evaluation.

### Class imbalance

Some defects are easier to generate than others. Mitigation: explicit class distributions and per-class metrics.

### Confidence misuse

A model confidence score is not severity. The CI policy layer must remain separate from model output.
