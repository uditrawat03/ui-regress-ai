# Dataset Strategy

## Goal

Create paired UI screenshots where the regression label and affected region are known by construction.

Each sample contains:

```text
baseline screenshot
mutated screenshot
regression class
mutation parameters
affected element selector
affected bounding box
viewport
browser metadata
seed
```

## Why synthetic generation first

Real visual-regression datasets with precise defect labels are difficult to collect at useful scale. UI source code gives us a better starting point: intentionally mutate a valid page, render it, and record exactly what changed.

Synthetic data is a starting point, not the final benchmark.

## Dataset generation pipeline

```text
HTML/CSS/JS fixture
       ↓
baseline render
       ↓
choose mutation using seeded RNG
       ↓
apply mutation
       ↓
mutated render
       ↓
collect bounding boxes + metadata
       ↓
write sample manifest
```

## Initial mutation operators

### `hide_element`

Makes a visible component disappear.

Target label: `element_missing`

### `translate_element`

Moves an element by a controlled amount.

Target label: `layout_shift`

### `increase_width`

Expands an element into adjacent content.

Possible label: `element_overlap` or `incorrect_size`

### `decrease_container_height`

Creates clipping.

Target label: `text_clipping`

### `overflow_hidden`

Applies clipping to overflowing content.

Target label: `text_clipping`

### `remove_background_image`

Removes an expected image.

Target label: `image_missing`

### `break_responsive_rule`

Disables or modifies a media-query behavior.

Target label: `responsive_layout_failure`

### `change_style_token`

Changes a visually meaningful style such as contrast, border, background, or typography.

Target label: `unexpected_style`

## No-regression pairs

The negative class must contain realistic rendering changes that should not fail CI.

Candidate sources:

- tiny anti-aliasing differences
- controlled JPEG compression
- subpixel translation below policy threshold
- non-semantic noise
- timestamp masking
- tolerated dynamic regions

Care is required: augmentations should not accidentally erase meaningful changes.

## Manifest format

```json
{
  "sample_id": "fixture-001-overlap-00042",
  "seed": 42,
  "baseline": "images/baseline.png",
  "current": "images/current.png",
  "label": "element_overlap",
  "viewport": {"width": 1440, "height": 900},
  "region": {"x": 612, "y": 433, "width": 280, "height": 94},
  "mutation": {
    "operator": "increase_width",
    "selector": ".checkout-actions",
    "parameters": {"scale": 1.6}
  }
}
```

## Splitting strategy

Do not randomly split screenshots originating from the same UI fixture across train and test. That leaks page identity.

Prefer:

```text
train: UI fixtures A-M
validation: UI fixtures N-P
test: UI fixtures Q-T
```

Real-world benchmark pages should remain completely separate.

## Dataset versions

Use immutable version IDs such as:

```text
synthetic-v0.1
synthetic-v0.2
real-benchmark-v0.1
```

A training checkpoint should always reference exact dataset versions.

## Real-world benchmark

Before claiming practical performance, collect a small benchmark from real UI defects.

Possible process:

1. create small open-source demo applications
2. make realistic bug commits
3. capture before/after screenshots
4. label regression category and region manually
5. freeze the benchmark

The real benchmark must not be used for routine training.
