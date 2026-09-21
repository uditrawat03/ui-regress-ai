# Training

Milestone 3 trains a shared-encoder Siamese classifier on the synthetic paired screenshot dataset.

## Model

The first model intentionally favors debuggability over complexity:

```text
baseline ─┐
          ├─ shared ResNet-18 encoder ─┐
current ──┘                            │
                                      ├─ concat(base, current, abs diff)
                                      │
                                      └─ shared comparison layer
                                          ├─ binary regression head
                                          └─ multi-class regression head
```

The binary head predicts whether a meaningful regression exists. The multi-class head predicts the manifest label.

Class IDs are derived from the dataset manifest and written into every checkpoint. `no_regression` is always index `0` when present.

## Preprocessing

Milestone 3 preprocessing is versioned as `imagenet-resize-v1`:

- RGB conversion
- resize to 224×224 by default
- ImageNet mean/std normalization

The checkpoint stores the preprocessing version, input size, mean, and standard deviation.

## Generate enough data

The three-fixture smoke dataset is useful for validating the pipeline, but too small for meaningful learning.

Generate a larger first dataset:

```powershell
uv run uiregress generate-dataset `
  --fixtures-dir .\fixtures `
  --output .\data\generated\synthetic-v0.1 `
  --version synthetic-v0.1 `
  --samples-per-fixture 50 `
  --seed 42 `
  --overwrite
```

## Train on CUDA

```powershell
uv run uiregress train `
  --dataset .\data\generated\synthetic-v0.1 `
  --checkpoint .\artifacts\checkpoints\milestone3-best.pt `
  --epochs 10 `
  --batch-size 8 `
  --device cuda
```

AMP is enabled automatically when CUDA is selected. Disable it for debugging with `--no-amp`.

If ImageNet weights are unavailable locally and network access is restricted, add `--no-pretrained`.

## Outputs

Training writes:

```text
artifacts/checkpoints/
├── milestone3-best.pt
└── milestone3-best.history.json
```

The checkpoint includes:

- architecture name
- model and optimizer state
- class mapping
- preprocessing configuration
- dataset version and seed
- training configuration
- git commit
- PyTorch and CUDA versions
- best validation metrics

The history JSON contains per-epoch train and validation metrics.

## Validation metrics

Milestone 3 reports:

- binary accuracy
- binary precision
- binary recall
- binary F1
- binary false-positive rate
- multi-class accuracy
- multi-class macro F1
- per-class precision, recall, F1, and support

The milestone exit condition is not satisfied until the learned model is compared against the classical baseline on semantic cases.
