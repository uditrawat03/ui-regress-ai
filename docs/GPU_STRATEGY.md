# GPU Strategy

## Purpose

CUDA should reduce training and inference cost for real workloads. GPU support is not a repository checkbox.

## Training

Planned GPU techniques:

- automatic mixed precision
- pinned-memory data loading when useful
- configurable batch size
- gradient accumulation
- activation checkpointing if larger backbones require it
- multi-GPU support only after single-GPU training is stable

## Inference

Inference path:

```text
load pair
  ↓
preprocess on CPU
  ↓
batched transfer to CUDA
  ↓
encoder + prediction heads
  ↓
copy compact outputs back to CPU
  ↓
postprocess / JSON
```

Avoid repeated model loads for batch or server use.

## Device policy

Supported requested modes:

```text
auto
cpu
cuda
```

`auto` selects CUDA only when PyTorch reports it available.

Explicit `cuda` should raise a clear error when CUDA is unavailable rather than silently changing execution mode.

## Mixed precision

Initial training should support AMP and compare:

- throughput
- peak VRAM
- validation metrics

Mixed precision remains configurable because numerical behavior may differ by architecture.

## Reproducibility

Record:

- GPU model name
- CUDA runtime reported by PyTorch
- PyTorch version
- cuDNN version when available
- deterministic settings
- random seed

## Profiling

Profile before optimizing.

Measure:

- dataloader time
- host-to-device transfer
- forward pass
- postprocessing
- VRAM allocation

Likely tools:

- `torch.profiler`
- NVIDIA tooling outside the Python package when needed

## Future optimization

Possible later work:

- `torch.compile`
- ONNX export
- TensorRT deployment
- model quantization where quality permits
- tiled inference for very large screenshots

These should not block the first useful release.
