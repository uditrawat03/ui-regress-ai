from __future__ import annotations

import random
from collections.abc import Iterable

from uiregress.data.schema import DatasetSplit


def split_fixtures(
    fixture_names: Iterable[str],
    *,
    seed: int,
) -> dict[str, DatasetSplit]:
    """Assign whole fixtures to splits so page identity never leaks across splits."""
    names = sorted(set(fixture_names))
    if not names:
        raise ValueError("At least one fixture is required.")

    rng = random.Random(seed)
    rng.shuffle(names)

    if len(names) == 1:
        return {names[0]: "train"}
    if len(names) == 2:
        return {names[0]: "train", names[1]: "test"}

    validation_count = max(1, round(len(names) * 0.15))
    test_count = max(1, round(len(names) * 0.15))

    while validation_count + test_count >= len(names):
        if validation_count >= test_count and validation_count > 1:
            validation_count -= 1
        elif test_count > 1:
            test_count -= 1
        else:
            break

    train_end = len(names) - validation_count - test_count
    validation_end = train_end + validation_count

    assignments: dict[str, DatasetSplit] = {}
    for name in names[:train_end]:
        assignments[name] = "train"
    for name in names[train_end:validation_end]:
        assignments[name] = "validation"
    for name in names[validation_end:]:
        assignments[name] = "test"

    return assignments
