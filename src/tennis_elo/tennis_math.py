"""Converting win probabilities between best-of-3 and best-of-5 matches.

Model: each set is an independent Bernoulli trial that player A wins with probability
``p`` (the "per-set probability"). A best-of-(2k-1) match is won by the first player to
take ``k`` sets, so

    P(win | best of 3) = p^2 (3 - 2p)
    P(win | best of 5) = p^3 (10 - 15p + 6p^2)

(the second line is p^3 * [1 + 3(1-p) + 6(1-p)^2] expanded). Both are strictly
increasing on [0, 1], fixed at 0, 1/2 and 1, so any match probability can be inverted
back to a unique ``p`` and pushed forward to the other format. The longer format
amplifies the edge of the better player: 0.70 in best-of-3 becomes about 0.76 in
best-of-5.
"""

from __future__ import annotations

from math import comb

import numpy as np


def match_prob_from_set(p_set, best_of: int = 3):
    """Probability of winning a best-of-``best_of`` match given per-set probability."""
    if best_of % 2 == 0 or best_of < 1:
        raise ValueError("best_of must be a positive odd number")
    p = np.asarray(p_set, dtype=float)
    q = 1.0 - p
    need = best_of // 2 + 1
    # win the final set after losing j of the earlier sets (j = 0 .. need-1)
    total = sum(comb(need - 1 + j, j) * q**j for j in range(need))
    out = p**need * total
    return out if out.ndim else float(out)


def set_prob_from_match(p_match, best_of: int = 3, tol: float = 1e-12):
    """Invert :func:`match_prob_from_set` by vectorised bisection."""
    target = np.asarray(p_match, dtype=float)
    if np.any((target < 0) | (target > 1)):
        raise ValueError("probabilities must lie in [0, 1]")
    lo = np.zeros_like(target)
    hi = np.ones_like(target)
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        too_low = match_prob_from_set(mid, best_of) < target
        lo = np.where(too_low, mid, lo)
        hi = np.where(too_low, hi, mid)
        if np.all(hi - lo < tol):
            break
    out = 0.5 * (lo + hi)
    return out if out.ndim else float(out)


def convert(p_match, from_best_of: int = 3, to_best_of: int = 5):
    """Convert a match win probability from one format to another via the set level."""
    if from_best_of == to_best_of:
        return p_match
    return match_prob_from_set(set_prob_from_match(p_match, from_best_of), to_best_of)


def bo3_to_bo5(p_match):
    """Shortcut: best-of-3 match probability -> best-of-5 match probability."""
    return convert(p_match, 3, 5)


_GRID = np.linspace(0.0, 1.0, 20001)
_BO5_ON_GRID = match_prob_from_set(set_prob_from_match(_GRID, 3), 5)


def bo3_to_bo5_fast(p_match):
    """Table-interpolated :func:`bo3_to_bo5` (max abs error ~1e-8); used in hot loops."""
    return np.interp(p_match, _GRID, _BO5_ON_GRID)
