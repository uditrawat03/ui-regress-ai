# Roadmap

## Phase 1: prove the problem can be measured

Build deterministic screenshot comparison and define regression labels. This gives us baseline metrics before machine learning.

## Phase 2: create training data

Generate known UI defects with HTML/CSS mutations and Playwright. Preserve exact mutation metadata and affected regions.

## Phase 3: train a pairwise vision model

Train a shared-encoder PyTorch model using CUDA. Start with binary detection and multi-class prediction before adding more complex localization.

## Phase 4: ground predictions

Add heatmaps or bounding boxes so a developer can inspect what triggered the result.

## Phase 5: test against reality

Create a frozen benchmark of realistic UI bugs and cross-environment rendering noise. Use this to quantify the synthetic-to-real gap.

## Phase 6: integrate with developer workflow

Ship stable CLI output, CI exit codes, GitHub Action support, annotated artifacts, and configurable policies.

## Phase 7: optimize

Only after quality is established, evaluate faster inference, larger models, model compilation, TensorRT, batching, and service deployment.

## Possible later research

- DOM + visual feature fusion
- accessibility tree integration
- text-aware region features
- learned perceptual tolerance by component type
- few-shot adaptation to a project's design system
- temporal regression analysis across multiple commits
- automatic clustering of recurring regression patterns
