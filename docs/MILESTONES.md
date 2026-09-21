# Milestones

## Milestone 0 — Repository foundation

Goal: make the project reproducible and understandable before model work starts.

- [x] define problem
- [x] define architecture
- [x] define labels
- [x] define dataset strategy
- [x] define evaluation metrics
- [x] define GPU policy
- [x] create Python package scaffold
- [x] add basic CI
- [ ] select license

Exit condition: a new contributor understands what is being built and how success will be measured.

## Milestone 1 — Classical visual baseline

- [x] image pair loader
- [x] dimension validation
- [x] absolute difference heatmap
- [x] changed-area ratio
- [x] structural-similarity baseline
- [x] CLI `compare`
- [x] JSON output schema
- [x] unit tests

Exit condition: we can benchmark a deterministic non-ML solution.

## Milestone 2 — Synthetic dataset generator

- [x] Playwright renderer
- [x] reusable fixture pages
- [x] seeded mutation engine
- [x] bounding-box capture
- [x] dataset manifest writer
- [x] no-regression augmentations
- [x] fixture-level train/val/test splitting

Exit condition: one command generates a versioned paired-image dataset.

## Milestone 3 — First PyTorch model

- [ ] dataset class
- [ ] shared encoder model
- [ ] binary regression head
- [ ] multi-class regression head
- [ ] training loop
- [ ] AMP support
- [ ] checkpoint metadata
- [ ] validation metrics

Exit condition: CUDA training is reproducible and beats at least one classical baseline on semantic cases.

## Milestone 4 — Localization

- [ ] heatmap/localization target generation
- [ ] localization head
- [ ] IoU evaluation
- [ ] annotated result image

Exit condition: predictions contain useful evidence about where the regression occurred.

## Milestone 5 — Real-world benchmark

- [ ] create realistic bug commits
- [ ] collect frozen screenshot pairs
- [ ] manual labels
- [ ] cross-browser noise suite
- [ ] benchmark report

Exit condition: project reports synthetic and real-world results separately.

## Milestone 6 — CI integration

- [ ] stable CLI schema
- [ ] exit-code contract
- [ ] GitHub Action
- [ ] artifact upload
- [ ] PR summary
- [ ] repository policy config

Exit condition: a sample application can use UIRegressAI in a pull request workflow.

## Milestone 7 — Service mode

- [ ] FastAPI service
- [ ] batch endpoint
- [ ] model lifecycle management
- [ ] GPU queueing
- [ ] observability

Exit condition: teams can operate the detector as a reusable internal service.
