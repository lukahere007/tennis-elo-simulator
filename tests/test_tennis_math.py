import numpy as np
import pytest

from tennis_elo.tennis_math import (bo3_to_bo5, bo3_to_bo5_fast, convert, match_prob_from_set,
                                    set_prob_from_match)


def test_known_values():
    # p = 0.6 per set: bo3 = 0.36 * (3 - 1.2) = 0.648; bo5 = 0.216 * (10 - 9 + 2.16) = 0.68256
    assert match_prob_from_set(0.6, 3) == pytest.approx(0.648)
    assert match_prob_from_set(0.6, 5) == pytest.approx(0.68256)
    assert match_prob_from_set(0.6, 1) == pytest.approx(0.6)


@pytest.mark.parametrize("best_of", [3, 5])
def test_fixed_points(best_of):
    assert match_prob_from_set(np.array([0.0, 0.5, 1.0]), best_of) == pytest.approx([0, 0.5, 1])


def test_inverse_roundtrip():
    p = np.linspace(0.01, 0.99, 50)
    for best_of in (3, 5):
        assert match_prob_from_set(set_prob_from_match(p, best_of), best_of) == pytest.approx(p, abs=1e-9)


def test_bo3_to_bo5_known_and_symmetric():
    assert bo3_to_bo5(0.648) == pytest.approx(0.68256, abs=1e-9)
    p = np.linspace(0.05, 0.95, 19)
    out = bo3_to_bo5(p)
    assert out + bo3_to_bo5(1 - p) == pytest.approx(np.ones_like(p))
    assert np.all(np.abs(out - 0.5) >= np.abs(p - 0.5))  # longer format amplifies edges
    assert convert(convert(p, 3, 5), 5, 3) == pytest.approx(p, abs=1e-9)


def test_fast_table_matches_exact():
    p = np.random.default_rng(0).random(500)
    assert bo3_to_bo5_fast(p) == pytest.approx(bo3_to_bo5(p), abs=1e-7)


def test_rejects_even_format():
    with pytest.raises(ValueError):
        match_prob_from_set(0.5, 4)
