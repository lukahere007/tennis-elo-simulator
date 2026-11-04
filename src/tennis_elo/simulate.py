"""Knock-out draw reconstruction and tournament Monte Carlo.

The match table has no draw sheet, but a single-elimination bracket can be rebuilt from
results alone: the two finalists each came out of a semi-final, each semi-finalist out
of a quarter-final, and so on down to the first round. Reading the tree from the top
gives the first-round pairings *and* the bracket structure (who could meet whom).

Given a pairwise win-probability matrix for the draw, :func:`simulate_bracket` plays the
whole tournament many times (vectorised over simulations), and :func:`exact_title_probs`
computes the same title probabilities exactly by dynamic programming over the tree,
which is used to check the simulator.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

KO_ROUNDS = ["R128", "R64", "R32", "R16", "QF", "SF", "F"]


class BracketError(ValueError):
    """Raised when the results do not form a complete single-elimination draw."""


def reconstruct_bracket(event: pd.DataFrame) -> list[int]:
    """Return player ids in draw order (adjacent pairs meet in round one).

    ``event`` holds the matches of one knock-out tournament with columns ``round``,
    ``winner_id`` and ``loser_id``. Walkovers must be kept: they are part of the draw.
    """
    rounds = [r for r in KO_ROUNDS if r in set(event["round"])]
    if not rounds or rounds[-1] != "F":
        raise BracketError("no final found")
    # the rounds present must be a contiguous tail of KO_ROUNDS ending in the final
    first = KO_ROUNDS.index(rounds[0])
    rounds = KO_ROUNDS[first:]
    played = {}
    for rnd, w, lo in event[["round", "winner_id", "loser_id"]].itertuples(index=False):
        if rnd in rounds:
            played[(rnd, int(w))] = (int(w), int(lo))
            played[(rnd, int(lo))] = (int(w), int(lo))

    def build(level: int, player: int) -> list[int]:
        key = (rounds[level], player)
        if key not in played:
            raise BracketError(f"{player} missing in {rounds[level]}")
        w, lo = played[key]
        if level == 0:
            return [w, lo]
        return build(level - 1, w) + build(level - 1, lo)

    final = event[event["round"] == "F"].iloc[0]
    leaves = build(len(rounds) - 1, int(final["winner_id"]))
    if len(set(leaves)) != len(leaves) or len(leaves) != 2 ** len(rounds):
        raise BracketError("inconsistent draw")
    return leaves


def simulate_bracket(win_prob: np.ndarray, n_sims: int = 20_000, seed: int = 0) -> np.ndarray:
    """Monte Carlo title probabilities.

    ``win_prob[i, j]`` is P(draw position i beats position j) and must satisfy
    ``win_prob + win_prob.T == 1`` off the diagonal. Returns an array of title
    probabilities by draw position.
    """
    m = win_prob.shape[0]
    if m & (m - 1):
        raise ValueError("draw size must be a power of two")
    rng = np.random.default_rng(seed)
    alive = np.tile(np.arange(m), (n_sims, 1))
    while alive.shape[1] > 1:
        a, b = alive[:, 0::2], alive[:, 1::2]
        a_wins = rng.random(a.shape) < win_prob[a, b]
        alive = np.where(a_wins, a, b)
    return np.bincount(alive[:, 0], minlength=m) / n_sims


def exact_title_probs(win_prob: np.ndarray) -> np.ndarray:
    """Exact title probabilities by recursion over the bracket tree.

    For a sub-bracket made of halves L and R, player i (in L) wins it with probability
    ``P_L(i) * sum_j P_R(j) * win_prob[i, j]``.
    """
    m = win_prob.shape[0]
    reach = np.ones(m)  # prob. of having won the current block so far
    size = 1
    while size < m:
        new = np.empty(m)
        for start in range(0, m, 2 * size):
            left = np.arange(start, start + size)
            right = np.arange(start + size, start + 2 * size)
            new[left] = reach[left] * (win_prob[np.ix_(left, right)] @ reach[right])
            new[right] = reach[right] * (win_prob[np.ix_(right, left)] @ reach[left])
        reach = new
        size *= 2
    return reach


def multiclass_brier(title_probs: np.ndarray, champion_pos: int) -> float:
    """Brier score summed over all players in the draw (0 = perfect, 2 = worst)."""
    y = np.zeros_like(title_probs)
    y[champion_pos] = 1.0
    return float(np.sum((title_probs - y) ** 2))
