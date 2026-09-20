# Contributing

UIRegressAI is currently in the foundation stage.

## Before implementing a model change

Please include:

- the problem being solved
- the baseline being compared against
- dataset version
- evaluation split
- metric impact
- CPU/GPU runtime impact if relevant

Model complexity without a measurable improvement is not a project goal.

## Development

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -e ".[dev]"
ruff check .
pytest -q
```

## Pull requests

Keep changes focused. Changes to preprocessing, labels, dataset generation, or output schema should include documentation because those choices affect reproducibility.
