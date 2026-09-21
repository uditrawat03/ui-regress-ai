# Synthetic Dataset Generation

Milestone 2 creates paired screenshots with labels and affected regions known by construction. Milestone 3.1 extends the generator with a balanced mode and a larger fixture corpus so the first neural model can learn semantic classes without extreme sampling imbalance.

## Install the dataset tooling

The browser automation dependency is optional so inference-only installs do not need Playwright.

```powershell
uv sync --all-extras
uv run playwright install chromium
```

Playwright browser binaries are installed separately from the Python package.

## Generate the balanced v0.2 training dataset

The recommended training dataset uses 18 distinct fixture pages and generates the same number of samples for every semantic label on every fixture.

```powershell
uv run uiregress generate-dataset `
  --fixtures-dir fixtures `
  --output data/generated/synthetic-v0.2 `
  --version synthetic-v0.2 `
  --samples-per-class 10 `
  --seed 42 `
  --overwrite
```

There are six labels today:

- `no_regression`
- `element_missing`
- `layout_shift`
- `incorrect_size`
- `text_clipping`
- `unexpected_style`

With 18 fixtures and `--samples-per-class 10`, generation produces:

```text
18 fixtures × 6 labels × 10 samples = 1,080 paired samples
1,080 baseline + 1,080 current screenshots = 2,160 images
```

The existing fixture-level split policy produces 12 train fixtures, 3 validation fixtures, and 3 test fixtures for an 18-fixture corpus. Because each fixture contains the same per-class count, every split is balanced by construction.

## Random generation remains available

The original random mode is still supported for exploratory datasets:

```powershell
uv run uiregress generate-dataset `
  --fixtures-dir fixtures `
  --output data/generated/synthetic-random `
  --version synthetic-random `
  --samples-per-fixture 50 `
  --no-regression-fraction 0.25 `
  --seed 42
```

When `--samples-per-class` is supplied, balanced mode takes precedence over random class sampling. `--samples-per-fixture` and `--no-regression-fraction` continue to control random mode.

Use `--overwrite` when intentionally replacing an existing generated dataset.

## Output layout

```text
data/generated/synthetic-v0.2/
├── dataset.json
├── manifest.jsonl
└── images/
    ├── analytics-element_missing-...-baseline.png
    ├── analytics-element_missing-...-current.png
    └── ...
```

`dataset.json` records:

- dataset version and generation seed
- fixture count and total sample count
- generation mode
- samples per fixture
- samples per class when balanced mode is enabled
- train, validation, and test counts
- global class distribution
- per-split class distribution

`manifest.jsonl` contains one JSON object per sample.

Each sample records:

- fixture name
- train, validation, or test split
- deterministic sample seed
- baseline/current image paths
- semantic regression label
- viewport
- affected bounding box when applicable
- mutation operator, selector, and parameters
- Chromium and Playwright version metadata

## Fixture corpus

The v0.2 corpus contains 18 structurally different standalone pages:

```text
analytics        auth             billing
calendar         checkout         dashboard
inbox            inventory        kanban
notifications    onboarding       pricing
product-detail   profile          search-results
settings         support          team
```

The pages intentionally vary navigation patterns, grids, tables, forms, cards, split panes, dense lists, and content hierarchy. This reduces the page-identity overfitting seen when the first training run used only three fixtures.

## Fixture contract

Fixtures are standalone HTML files. They must not depend on network resources. Elements eligible for mutation use a stable target identifier:

```html
<div data-uiregress-target="checkout-actions">Pay now</div>
```

Target values must be unique inside a fixture. The generator turns them into deterministic attribute selectors. The current corpus keeps at least five mutation targets per fixture.

## Current mutation operators

- `hide_element` → `element_missing`
- `translate_element` → `layout_shift`
- `increase_width` → `incorrect_size`
- `overflow_hidden` → `text_clipping`
- `change_style_token` → `unexpected_style`

Negative examples use `subtle_pixel_noise` and receive the `no_regression` label. The noise is seeded and limited to one RGB level by default, so it changes rendering pixels without changing page semantics.

In balanced mode, the mutation operator is selected from the requested semantic label instead of being sampled randomly. Target elements and mutation parameters remain seeded and deterministic.

## Split policy

Fixtures, not screenshots, are assigned to splits. A fixture can only appear in one of `train`, `validation`, or `test`. This prevents the model from seeing the same page identity in training and evaluation.

The assignment is deterministic for a given global seed. For 18 fixtures the current 70/15/15-style policy resolves to:

```text
train:       12 fixtures
validation:   3 fixtures
test:         3 fixtures
```

## Reproducibility

For a fixed set of fixture files, generation seed, viewport, dependency versions, and browser version, the generator chooses the same split, sample seeds, selectors, mutation operators, and mutation parameters.

Browser rendering can still change when Chromium or fonts change. The manifest stores browser metadata so those differences can be traced.
