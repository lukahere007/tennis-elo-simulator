import numpy as np
import pandas as pd
import pytest

from tennis_elo.elo import EloParams, elo_update, expected_score, k_factor, run_elo, _bo5
from tennis_elo.tennis_math import bo3_to_bo5


def test_expected_score_formula():
    assert expected_score(1500, 1500) == pytest.approx(0.5)
    assert expected_score(1900, 1500) == pytest.approx(10 / 11)  # 400 points = 10:1 odds
    assert expected_score(1600, 1500) + expected_score(1500, 1600) == pytest.approx(1.0)


def test_k_factor_decays():
    ks = [k_factor(n, 250, 5, 0.4) for n in (0, 10, 100, 1000)]
    assert ks[0] == pytest.approx(250 / 5 ** 0.4)
    assert all(a > b for a, b in zip(ks, ks[1:]))


def test_update_symmetry_and_conservation():
    w, lo = elo_update(1600, 1500, 32, 32)
    assert (w - 1600) == pytest.approx(1500 - lo)  # zero-sum with equal K
    assert w + lo == pytest.approx(3100)
    # upset moves ratings more than an expected win
    w2, lo2 = elo_update(1500, 1600, 32, 32)
    assert w2 - 1500 > w - 1600
    # mirror symmetry: gains of an underdog win = losses of the favourite
    assert w2 - 1500 == pytest.approx(1600 - lo2)


def test_scalar_bo5_table():
    for p in (0.1, 0.5, 0.73, 0.99):
        assert _bo5(p) == pytest.approx(bo3_to_bo5(p), abs=1e-7)


def _toy_matches(n=400, seed=1):
    rng = np.random.default_rng(seed)
    skill = np.array([0, 100, 200, 300.0])
    rows = []
    for t in range(n):
        a, b = rng.choice(4, 2, replace=False)
        p = 1 / (1 + 10 ** (-(skill[a] - skill[b]) / 400))
        w, lo = (a, b) if rng.random() < p else (b, a)
        rows.append(dict(winner_id=w, loser_id=lo, surface_code=t % 3, best_of=3,
                         walkover=False, score="6-4 6-4"))
    return pd.DataFrame(rows)


def test_run_elo_recovers_order_and_conserves():
    df = _toy_matches()
    res = run_elo(df, EloParams(c=32, o=0, s=0, surface_weight=0.0))  # constant K = 32
    ratings = res.snapshots  # empty
    assert ratings == {}
    # with constant equal K the total rating is conserved: check through a snapshot
    res = run_elo(df, EloParams(c=32, o=0, s=0), snapshot_at={len(df) - 1})
    state = res.snapshots[len(df) - 1]
    assert state.overall.sum() == pytest.approx(1500 * 4)
    order = [res.players[i] for i in range(4)]
    assert np.argsort(state.overall[order]).tolist() == [0, 1, 2, 3]
    assert np.all((res.p_blend > 0) & (res.p_blend < 1))


def test_predictions_are_pre_match():
    df = _toy_matches(5)
    res = run_elo(df, EloParams())
    assert res.p_blend[0] == pytest.approx(0.5)  # first match: no information yet


def test_walkover_ignored():
    df = _toy_matches(3)
    df.loc[0, "walkover"] = True
    res = run_elo(df, EloParams())
    assert np.isnan(res.p_blend[0])
    assert res.pre_w_overall[1] == 1500 and res.pre_l_overall[1] == 1500


def test_rolling_scale_recovers_overconfidence():
    from tennis_elo.elo import prob_from_diff
    from tennis_elo.evaluate import fit_scale, rolling_scale

    rng = np.random.default_rng(0)
    n = 60_000
    d_true = rng.normal(0, 200, n)
    won = rng.random(n) < prob_from_diff(d_true, np.zeros(n, bool))
    d_model = 1.25 * d_true  # ratings 25% too spread out
    d_w = np.where(won, d_model, -d_model)  # winner-oriented
    bo5 = np.zeros(n, bool)
    assert fit_scale(d_w, bo5) == pytest.approx(0.8, abs=0.03)
    year = np.repeat(np.arange(2010, 2016), n // 6)
    scales = rolling_scale(d_w, bo5, year, np.ones(n, bool))
    assert scales[2010] == 1.0  # no past data -> no adjustment
    assert scales[2015] == pytest.approx(0.8, abs=0.03)
