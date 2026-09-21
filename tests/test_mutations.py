import random

from uiregress.data.mutations import choose_mutation, no_regression_mutation


def test_seeded_mutation_choice_is_reproducible() -> None:
    selectors = [
        '[data-uiregress-target="card"]',
        '[data-uiregress-target="button"]',
    ]

    first = choose_mutation(random.Random(42), selectors)
    second = choose_mutation(random.Random(42), selectors)

    assert first == second


def test_no_regression_mutation_has_no_dom_selector() -> None:
    mutation = no_regression_mutation()

    assert mutation.label == "no_regression"
    assert mutation.selector is None
    assert mutation.parameters["amplitude"] == 1
