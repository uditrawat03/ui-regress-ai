# Synthetic Dataset Generation

Milestone 2 creates paired screenshots with labels and affected regions known by construction.

## Install the dataset tooling

The browser automation dependency is optional so inference-only installs do not need Playwright.

```powershell
uv sync --all-extras
uv run playwright install chromium
```

Playwright browser binaries are installed separately from the Python package.

## Generate the starter dataset

```powershell
uv run uiregress generate-dataset `
  --fixtures-dir fixtures `
  --output data/generated/synthetic-v0.1 `
  --version synthetic-v0.1 `
  --samples-per-fixture 8 `
  --seed 42
```

Use `--overwrite` when intentionally replacing an existing generated dataset.

## Output layout

```text
data/generated/synthetic-v0.1/
├── dataset.json
├── manifest.jsonl
└── images/
    ├── dashboard-...-baseline.png
    ├── dashboard-...-current.png
    └── ...
```

`dataset.json` records the dataset version, generation seed, fixture count, sample count, and split counts. `manifest.jsonl` contains one JSON object per sample.

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

## Fixture contract

Fixtures are standalone HTML files. Elements eligible for mutation use a stable target identifier:

```html
<div data-uiregress-target="checkout-actions">Pay now</div>
```

Target values must be unique inside a fixture. The generator turns them into deterministic attribute selectors.

## Current mutation operators

- `hide_element` → `element_missing`
- `translate_element` → `layout_shift`
- `increase_width` → `incorrect_size`
- `overflow_hidden` → `text_clipping`
- `change_style_token` → `unexpected_style`

Negative examples use `subtle_pixel_noise` and receive the `no_regression` label. The noise is seeded and limited to one RGB level by default, so it changes rendering pixels without changing page semantics.

## Split policy

Fixtures, not screenshots, are assigned to splits. A fixture can only appear in one of `train`, `validation`, or `test`. This prevents the model from seeing the same page identity in training and evaluation.

The assignment is deterministic for a given global seed.

## Reproducibility

For a fixed set of fixture files, generation seed, viewport, dependency versions, and browser version, the generator chooses the same split, sample seeds, selectors, mutation operators, and mutation parameters.

Browser rendering can still change when Chromium or fonts change. The manifest stores browser metadata so those differences can be traced.
