# Milestone 3.2 — Operational Evaluation and Classical Benchmark

Milestone 3.2 separates "best semantic classifier" from "best CI detector" and evaluates both without using the test split for model or threshold selection.

## Checkpoint policy

The path passed to `--checkpoint` is the operational checkpoint. It is selected on the validation split using this policy:

1. require binary regression recall to meet `minimum_binary_recall` (default `0.90`)
2. among eligible epochs, minimize false-positive rate
3. break ties with multi-class macro F1
4. break remaining ties with binary F1

A second checkpoint is written beside it with `.multiclass.pt` appended to the stem. That checkpoint tracks the highest validation macro F1 regardless of binary false-positive rate.

Example:

```text
artifacts/checkpoints/milestone3-v0.2-best.pt
artifacts/checkpoints/milestone3-v0.2-best.multiclass.pt
```

## Threshold calibration

The binary decision threshold is calibrated independently for every epoch using validation probabilities only. The chosen threshold is stored in the checkpoint under `decision.binary_threshold`.

The test split is never used to select an epoch or threshold.

## Test evaluation

After training finishes, UIRegressAI reloads the selected operational checkpoint and evaluates it once on the untouched test split.

The report includes:

- binary accuracy, precision, recall, F1, and false-positive rate
- multi-class accuracy and macro F1
- per-class precision, recall, F1, and support
- labeled confusion matrix

## Classical benchmark

The same validation/test split policy is applied to three non-ML scores:

- mean absolute RGB error
- changed-area ratio
- SSIM distance (`1 - SSIM`)

Each classical score gets its threshold from validation using the same minimum-recall constraint. Its frozen threshold is then evaluated on the test split.

## Artifacts

A training run now produces:

```text
milestone3-v0.2-best.pt                 # operational checkpoint
milestone3-v0.2-best.multiclass.pt      # best macro-F1 checkpoint
milestone3-v0.2-best.history.json       # epoch history
milestone3-v0.2-best.benchmark.json     # untouched test + classical benchmark
```

The existing training command remains valid:

```powershell
uv run uiregress train `
  --dataset .\data\generated\synthetic-v0.2 `
  --checkpoint .\artifacts\checkpoints\milestone3-v0.2-best.pt `
  --epochs 5 `
  --batch-size 8 `
  --device cuda `
  --minimum-binary-recall 0.90
```

Use `--no-classical-benchmark` for quick training-only experiments. The default keeps the classical benchmark enabled so Milestone 3 exit criteria can be measured.

Milestone 3 is complete only after the benchmark demonstrates that the learned detector adds value over at least one classical baseline on the frozen test split.
