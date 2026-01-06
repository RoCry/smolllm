import pytest

from smolllm.core import (
    RandomSelector,
    SequentialSelector,
    _create_selector,
)


class TestSequentialSelector:
    def test_single_model(self):
        selector = SequentialSelector(["model1"])
        assert selector.next_model() == "model1"
        assert selector.next_model() is None

    def test_multiple_models(self):
        selector = SequentialSelector(["model1", "model2", "model3"])
        assert selector.next_model() == "model1"
        assert selector.next_model() == "model2"
        assert selector.next_model() == "model3"
        assert selector.next_model() is None

    def test_empty_list(self):
        selector = SequentialSelector([])
        assert selector.next_model() is None


class TestRandomSelector:
    def test_set_exhausts_all(self):
        models = {"a", "b", "c"}
        selector = RandomSelector(models)
        results = set()
        for _ in range(3):
            m = selector.next_model()
            assert m is not None
            results.add(m)
        assert results == models
        assert selector.next_model() is None

    def test_dict_exhausts_all(self):
        models = {"a": 1, "b": 2, "c": 3}
        selector = RandomSelector(models)
        results = set()
        for _ in range(3):
            m = selector.next_model()
            assert m is not None
            results.add(m)
        assert results == set(models.keys())
        assert selector.next_model() is None

    def test_weighted_distribution(self):
        """Verify weighted selection roughly follows distribution over many trials."""
        counts: dict[str, int] = {"high": 0, "low": 0}
        trials = 1000
        for _ in range(trials):
            selector = RandomSelector({"high": 9, "low": 1})
            first = selector.next_model()
            assert first is not None
            counts[first] += 1
        # high should be picked ~90% of the time, allow some variance
        assert counts["high"] > trials * 0.8
        assert counts["low"] < trials * 0.3


class TestCreateSelector:
    def test_string_single(self):
        selector = _create_selector("model1")
        assert isinstance(selector, SequentialSelector)
        assert selector.next_model() == "model1"
        assert selector.next_model() is None

    def test_string_comma_separated(self):
        selector = _create_selector("a, b, c")
        assert isinstance(selector, SequentialSelector)
        assert selector.next_model() == "a"
        assert selector.next_model() == "b"
        assert selector.next_model() == "c"
        assert selector.next_model() is None

    def test_list(self):
        selector = _create_selector(["a", "b"])
        assert isinstance(selector, SequentialSelector)
        assert selector.next_model() == "a"
        assert selector.next_model() == "b"

    def test_set(self):
        selector = _create_selector({"a", "b"})
        assert isinstance(selector, RandomSelector)
        results = {selector.next_model(), selector.next_model()}
        assert results == {"a", "b"}

    def test_dict(self):
        selector = _create_selector({"a": 1, "b": 2})
        assert isinstance(selector, RandomSelector)
        results = {selector.next_model(), selector.next_model()}
        assert results == {"a", "b"}

    def test_empty_string_raises(self):
        with pytest.raises(ValueError, match="at least one"):
            _create_selector("")

    def test_empty_set_raises(self):
        with pytest.raises(ValueError, match="at least one"):
            _create_selector(set())

    def test_empty_dict_raises(self):
        with pytest.raises(ValueError, match="at least one"):
            _create_selector({})

    def test_empty_list_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            _create_selector([])

    def test_negative_weight_raises(self):
        with pytest.raises(ValueError, match="positive"):
            _create_selector({"a": -1})

    def test_zero_weight_raises(self):
        with pytest.raises(ValueError, match="positive"):
            _create_selector({"a": 0})
