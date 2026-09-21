from uiregress.data.splitting import split_fixtures


def test_fixture_splits_are_reproducible_and_exclusive() -> None:
    fixtures = [f"fixture-{index}" for index in range(10)]

    first = split_fixtures(fixtures, seed=99)
    second = split_fixtures(fixtures, seed=99)

    assert first == second
    assert set(first) == set(fixtures)
    assert set(first.values()) == {"train", "validation", "test"}


def test_three_fixtures_get_three_distinct_splits() -> None:
    result = split_fixtures(["dashboard", "checkout", "profile"], seed=42)

    assert sorted(result.values()) == ["test", "train", "validation"]
