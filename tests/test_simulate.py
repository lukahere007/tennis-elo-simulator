import numpy as np
import pandas as pd
import pytest

from tennis_elo.simulate import (BracketError, exact_title_probs, multiclass_brier,
                                 reconstruct_bracket, simulate_bracket)


def toy_matrix():
    # P[i, j] = P(i beats j); players 0 v 1 and 2 v 3 in the semis
    p01, p02, p03, p12, p13, p23 = 0.7, 0.6, 0.8, 0.4, 0.55, 0.65
    P = np.full((4, 4), 0.5)
    for (i, j), v in {(0, 1): p01, (0, 2): p02, (0, 3): p03, (1, 2): p12, (1, 3): p13,
                      (2, 3): p23}.items():
        P[i, j], P[j, i] = v, 1 - v
    return P


def analytic(P):
    s = [P[0, 1], P[1, 0], P[2, 3], P[3, 2]]
    return np.array([
        s[0] * (s[2] * P[0, 2] + s[3] * P[0, 3]),
        s[1] * (s[2] * P[1, 2] + s[3] * P[1, 3]),
        s[2] * (s[0] * P[2, 0] + s[1] * P[2, 1]),
        s[3] * (s[0] * P[3, 0] + s[1] * P[3, 1]),
    ])


def test_exact_matches_analytic():
    P = toy_matrix()
    assert exact_title_probs(P) == pytest.approx(analytic(P))
    assert exact_title_probs(P).sum() == pytest.approx(1.0)


def test_monte_carlo_matches_analytic():
    P = toy_matrix()
    mc = simulate_bracket(P, n_sims=200_000, seed=3)
    assert mc == pytest.approx(analytic(P), abs=0.005)


def test_exact_equal_players_uniform():
    P = np.full((8, 8), 0.5)
    assert exact_title_probs(P) == pytest.approx(np.full(8, 1 / 8))


def test_reconstruct_bracket():
    ev = pd.DataFrame({
        "round": ["SF", "SF", "F"],
        "winner_id": [10, 30, 30],
        "loser_id": [20, 40, 10],
    })
    leaves = reconstruct_bracket(ev)
    assert leaves == [30, 40, 10, 20]
    with pytest.raises(BracketError):
        reconstruct_bracket(ev.iloc[[0, 2]])


def test_multiclass_brier():
    assert multiclass_brier(np.array([1.0, 0, 0]), 0) == 0.0
    assert multiclass_brier(np.array([0.5, 0.5]), 1) == pytest.approx(0.5)
