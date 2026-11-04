"""Grand Slam forecasts: rebuild each draw, freeze ratings the day before, simulate."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .elo import EloParams, EloResult, blended_prob
from .evaluate import rank_logit_prob
from .simulate import BracketError, exact_title_probs, reconstruct_bracket, simulate_bracket


def slam_events(matches: pd.DataFrame, years: tuple[int, int]) -> list[tuple[str, int]]:
    """(tourney_id, first row position) for every Grand Slam whose year is in ``years``."""
    g = matches[(matches["tourney_level"] == "G") & matches["year"].between(*years)]
    first = g.reset_index().groupby("tourney_id")["index"].min().sort_values()
    return list(first.items())


def elo_matrix(result: EloResult, start_row: int, ids: list[int], surface_code: int,
               params: EloParams, best_of: int = 5) -> np.ndarray:
    """Pairwise P(i beats j) for the draw from the ratings frozen before ``start_row``."""
    state = result.snapshots[start_row]
    pos = np.array([result.players[i] for i in ids])
    a, b = np.meshgrid(pos, pos, indexing="ij")
    P = blended_prob(state, a, b, surface_code, params.surface_weight, best_of, params.bo5_adjust)
    np.fill_diagonal(P, 0.5)
    return P


def rank_matrix(ranks: np.ndarray, coef: tuple[float, float]) -> np.ndarray:
    """Pairwise P(i beats j) from the best-of-5 rank-logit baseline."""
    lr = np.log(ranks)
    x = lr[None, :] - lr[:, None]  # log(rank_j) - log(rank_i)
    return rank_logit_prob(x, np.ones_like(x), coef)


def entry_ranks(event: pd.DataFrame, ids: list[int], missing: float = 2000.0) -> np.ndarray:
    """ATP rank of each player at the event (as listed in the match table)."""
    w = event[["winner_id", "winner_rank"]].set_axis(["id", "rank"], axis=1)
    lo = event[["loser_id", "loser_rank"]].set_axis(["id", "rank"], axis=1)
    r = pd.concat([w, lo]).dropna().drop_duplicates("id").set_index("id")["rank"]
    return np.array([float(r.get(i, missing)) for i in ids]).clip(min=1)


def forecast_slam(matches: pd.DataFrame, result: EloResult, tourney_id: str, start_row: int,
                  params: EloParams, rank_coef: tuple[float, float], n_sims: int = 20_000,
                  seed: int = 0) -> pd.DataFrame | None:
    """Pre-tournament title probabilities for every player in one Grand Slam draw."""
    event = matches[matches["tourney_id"] == tourney_id]
    try:
        ids = reconstruct_bracket(event)
    except BracketError:
        return None
    surface_code = int(event["surface_code"].iloc[0])
    P = elo_matrix(result, start_row, ids, surface_code, params)
    ranks = entry_ranks(event, ids)
    names = pd.concat([event[["winner_id", "winner_name"]].set_axis(["id", "name"], axis=1),
                       event[["loser_id", "loser_name"]].set_axis(["id", "name"], axis=1)]
                      ).drop_duplicates("id").set_index("id")["name"]
    champion = int(event.loc[event["round"] == "F", "winner_id"].iloc[0])
    out = pd.DataFrame({
        "tourney_id": tourney_id,
        "tourney_name": event["tourney_name"].iloc[0],
        "date": event["tourney_date"].iloc[0],
        "player_id": ids,
        "player": [names[i] for i in ids],
        "rank": ranks,
        "p_title_mc": simulate_bracket(P, n_sims=n_sims, seed=seed),
        "p_title_exact": exact_title_probs(P),
        "p_title_rank": exact_title_probs(rank_matrix(ranks, rank_coef)),
        "champion": [i == champion for i in ids],
    })
    return out.sort_values("p_title_exact", ascending=False).reset_index(drop=True)


def standard_seed_positions(draw_size: int, n_seeds: int) -> list[int]:
    """Bracket positions of seeds 1..n so that seeds 1 and 2 can only meet in the final,
    1-4 only from the semis, and so on (the classic "1 v 16, 8 v 9" ordering)."""
    order = [0]
    size = 1
    while size < draw_size:
        order = [x for o in order for x in (o, 2 * size - 1 - o)]
        size *= 2
    # order[k] is the seed rank (0-based) placed in slot k; invert for seed -> slot
    slot_of = {seed: slot for slot, seed in enumerate(order)}
    return [slot_of[s] for s in range(n_seeds)]


def random_seeded_draw(players: np.ndarray, n_seeds: int, rng: np.random.Generator) -> np.ndarray:
    """A random draw where ``players`` (best first) are seeded Grand-Slam style.

    Seeds 1 and 2 are fixed, 3-4, 5-8, 9-16 and 17-32 are shuffled within their tier
    across the tier's slots, everybody else fills the remaining slots at random.
    """
    m = len(players)
    slots = standard_seed_positions(m, n_seeds)
    draw = np.full(m, -1)
    tiers = [(0, 1), (1, 2), (2, 4), (4, 8), (8, 16), (16, 32), (32, 64)]
    for lo, hi in tiers:
        if lo >= n_seeds:
            break
        hi = min(hi, n_seeds)
        tier_slots = rng.permutation(slots[lo:hi])
        draw[tier_slots] = players[lo:hi]
    rest = np.flatnonzero(draw < 0)
    draw[rng.permutation(rest)] = players[n_seeds:]
    return draw
