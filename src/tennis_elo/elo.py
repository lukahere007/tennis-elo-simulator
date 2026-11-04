"""Surface-aware Elo with a match-count-decaying K-factor.

Every player carries two ratings:

* an **overall** rating, updated after every match, and
* a **surface** rating for the surface the match was played on (hard / clay / grass).

The pre-match forecast blends the two rating *differences* (not the probabilities):

    d = (1 - w) * (R_A - R_B) + w * (S_A - S_B),     P(A beats B) = 1 / (1 + 10^(-d / 400))

with the weight ``w`` learned from data. The step size follows FiveThirtyEight's tennis
Elo, ``K(n) = c / (n + o)^s``, where ``n`` is the number of matches the player had
already played (on that surface for the surface rating): newcomers move fast, veterans
slowly.

Best-of-5: the logistic curve above is read as a *best-of-3* probability. For best-of-5
matches (Grand Slams, old Masters finals) the forecast is converted through the per-set
probability (see :mod:`tennis_elo.tennis_math`), and that converted forecast is also the
expected score used in the update, so a favourite who wins a five-setter is not
over-rewarded for an outcome the longer format made more likely.

Optional margin-of-victory multiplier: the K step is scaled by ``(2 * g)^beta`` where
``g`` is the winner's share of games (0.5 .. 1). ``beta = 0`` switches it off.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from .data import games_won
from .tennis_math import _BO5_ON_GRID

INITIAL_RATING = 1500.0
_GRID_N = len(_BO5_ON_GRID) - 1
_BO5_LIST = _BO5_ON_GRID.tolist()


@dataclass(frozen=True)
class EloParams:
    """Hyper-parameters of the rating system (538 defaults for ``c, o, s``)."""

    c: float = 250.0
    o: float = 5.0
    s: float = 0.4
    surface_weight: float = 0.5
    mov_beta: float = 0.0
    bo5_adjust: bool = True

    def to_dict(self) -> dict:
        return asdict(self)


def expected_score(r_a: float, r_b: float) -> float:
    """Elo expected score of A against B: ``1 / (1 + 10^((r_b - r_a) / 400))``."""
    return 1.0 / (1.0 + 10.0 ** ((r_b - r_a) / 400.0))


def k_factor(n_matches: float, c: float, o: float, s: float) -> float:
    """538-style decaying K: ``c / (n + o)^s``."""
    return c / (n_matches + o) ** s


def _bo5(p: float) -> float:
    """Fast scalar best-of-3 -> best-of-5 conversion via the precomputed table."""
    x = p * _GRID_N
    i = int(x)
    if i >= _GRID_N:
        return 1.0
    f = x - i
    return _BO5_LIST[i] * (1.0 - f) + _BO5_LIST[i + 1] * f


def elo_update(r_w: float, r_l: float, k_w: float, k_l: float, expected_w: float | None = None,
               multiplier: float = 1.0) -> tuple[float, float]:
    """Return new (winner, loser) ratings after the winner beat the loser.

    ``expected_w`` defaults to :func:`expected_score`; pass a converted value for
    best-of-5 matches. With equal K the update is zero-sum.
    """
    if expected_w is None:
        expected_w = expected_score(r_w, r_l)
    surprise = 1.0 - expected_w
    return r_w + multiplier * k_w * surprise, r_l - multiplier * k_l * surprise


@dataclass
class EloState:
    """Rating arrays indexed by player position (see :class:`EloResult.players`)."""

    overall: np.ndarray
    surface: np.ndarray  # shape (n_players, 3)
    n_overall: np.ndarray
    n_surface: np.ndarray


@dataclass
class EloResult:
    """Pre-match forecasts (from the winner's point of view) and optional snapshots."""

    p_overall: np.ndarray
    p_surface: np.ndarray
    p_blend: np.ndarray
    pre_w_overall: np.ndarray
    pre_l_overall: np.ndarray
    pre_w_surface: np.ndarray
    pre_l_surface: np.ndarray
    players: dict
    snapshots: dict


def run_elo(matches: pd.DataFrame, params: EloParams = EloParams(),
            snapshot_at: set[int] | None = None) -> EloResult:
    """Run the ratings through ``matches`` in row order (must be chronological).

    Each forecast is made *before* the match's result is applied, so the output is a
    genuine one-step-ahead (walk-forward) prediction. Walkovers get ``nan`` forecasts
    and do not change ratings.

    ``snapshot_at`` is a set of row positions; the full rating state *before* that row
    is copied into ``result.snapshots[pos]`` (used for tournament simulations).
    """
    ids = pd.unique(np.concatenate([matches["winner_id"].to_numpy(), matches["loser_id"].to_numpy()]))
    index = {pid: i for i, pid in enumerate(ids.tolist())}
    n_players = len(ids)
    overall = [INITIAL_RATING] * n_players
    surface = [[INITIAL_RATING] * 3 for _ in range(n_players)]
    n_o = [0] * n_players
    n_s = [[0] * 3 for _ in range(n_players)]

    w_idx = [index[x] for x in matches["winner_id"].tolist()]
    l_idx = [index[x] for x in matches["loser_id"].tolist()]
    surf = matches["surface_code"].tolist()
    bo5 = (matches["best_of"].to_numpy() == 5).tolist()
    walkover = matches["walkover"].tolist()
    if params.mov_beta:
        gw = [games_won(s) for s in matches["score"].fillna("").tolist()]
        mult = [((2.0 * a / (a + b)) ** params.mov_beta if a + b > 0 else 1.0) for a, b in gw]
    else:
        mult = None

    n = len(w_idx)
    p_o = np.full(n, np.nan)
    p_s = np.full(n, np.nan)
    p_b = np.full(n, np.nan)
    pre = np.full((4, n), np.nan)
    c, o, s, w = params.c, params.o, params.s, params.surface_weight
    adjust = params.bo5_adjust
    snapshot_at = snapshot_at or set()
    snapshots = {}
    ln10_400 = math.log(10.0) / 400.0

    for t in range(n):
        if t in snapshot_at:
            snapshots[t] = EloState(np.array(overall), np.array(surface),
                                    np.array(n_o), np.array(n_s))
        i, j, sc = w_idx[t], l_idx[t], surf[t]
        ro_i, ro_j = overall[i], overall[j]
        rs_i, rs_j = surface[i][sc], surface[j][sc]
        pre[0, t], pre[1, t], pre[2, t], pre[3, t] = ro_i, ro_j, rs_i, rs_j
        if walkover[t]:
            continue
        e_o = 1.0 / (1.0 + math.exp(-(ro_i - ro_j) * ln10_400))
        e_s = 1.0 / (1.0 + math.exp(-(rs_i - rs_j) * ln10_400))
        d = (1.0 - w) * (ro_i - ro_j) + w * (rs_i - rs_j)
        e_b = 1.0 / (1.0 + math.exp(-d * ln10_400))
        if adjust and bo5[t]:
            e_o, e_s, e_b = _bo5(e_o), _bo5(e_s), _bo5(e_b)
        p_o[t], p_s[t], p_b[t] = e_o, e_s, e_b

        m = mult[t] if mult is not None else 1.0
        k_i = c / (n_o[i] + o) ** s
        k_j = c / (n_o[j] + o) ** s
        overall[i] = ro_i + m * k_i * (1.0 - e_o)
        overall[j] = ro_j - m * k_j * (1.0 - e_o)
        ks_i = c / (n_s[i][sc] + o) ** s
        ks_j = c / (n_s[j][sc] + o) ** s
        surface[i][sc] = rs_i + m * ks_i * (1.0 - e_s)
        surface[j][sc] = rs_j - m * ks_j * (1.0 - e_s)
        n_o[i] += 1
        n_o[j] += 1
        n_s[i][sc] += 1
        n_s[j][sc] += 1

    return EloResult(p_o, p_s, p_b, pre[0], pre[1], pre[2], pre[3], index, snapshots)


def blended_prob(state: EloState, a: int, b: int, surface_code: int, w: float,
                 best_of: int = 3, bo5_adjust: bool = True):
    """Forecast P(a beats b) from a rating snapshot; ``a`` and ``b`` may be arrays."""
    d = (1.0 - w) * (state.overall[a] - state.overall[b]) + w * (
        state.surface[a, surface_code] - state.surface[b, surface_code]
    )
    p = 1.0 / (1.0 + 10.0 ** (-d / 400.0))
    if best_of == 5 and bo5_adjust:
        from .tennis_math import bo3_to_bo5_fast

        p = bo3_to_bo5_fast(p)
    return p
