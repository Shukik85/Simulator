from hydrosim.core.units import BAR, PASCAL


def test_bar_to_pascal():
    assert BAR == 1e5 * PASCAL
