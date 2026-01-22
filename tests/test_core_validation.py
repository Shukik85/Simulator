import pytest

from hydrosim.core.validation import ensure_positive


def test_ensure_positive_ok():
    ensure_positive(1.0, "x")


def test_ensure_positive_raises():
    with pytest.raises(ValueError):
        ensure_positive(0.0, "x")
