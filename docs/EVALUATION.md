# Evaluation Plan

## Main rule

Accuracy alone is not enough for visual regression testing.

A system that flags harmless changes frequently will be ignored or disabled.

## Classification metrics

Report:

- precision per class
- recall per class
- F1 per class
- macro F1
- confusion matrix
- binary regression ROC/PR curves where useful

## False-positive metrics

The primary operational metric is the false-positive rate on `no_regression` pairs.

Also report false-positive rate grouped by:

- browser
- resolution
- rendering environment
- font configuration
- compression/noise augmentation

## Localization metrics

For samples with ground-truth regions:

- intersection-over-union (IoU)
- center-point hit rate
- recall at coarse heatmap threshold

The MVP does not need pixel-perfect segmentation.

## Calibration

Confidence should be evaluated rather than assumed.

Track:

- expected calibration error
- reliability curve
- confidence distribution per class

## Performance benchmarks

Measure separately on CPU and CUDA:

- single-pair latency
- batch latency
- images/second
- peak GPU memory
- model load time

Suggested benchmark resolutions:

```text
1280x720
1440x900
1920x1080
```

## Baselines

Every neural experiment should compare against at least:

1. absolute pixel difference
2. changed-area threshold
3. structural similarity metric

The learned model needs to solve cases the classical baselines cannot handle reliably.

## Dataset reporting

Every evaluation report must state:

- dataset version
- split
- number of examples
- class distribution
- renderer/browser
- checkpoint identifier
- preprocessing configuration

## Release gate

Do not call a model release-ready based only on synthetic holdout metrics.

A release candidate should also pass:

- frozen real-world benchmark
- cross-browser noise benchmark
- CPU fallback tests
- GPU inference tests
- CLI schema compatibility tests
