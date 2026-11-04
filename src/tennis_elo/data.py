"""Loading and cleaning the ATP match table."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "atp_matches.parquet"

SURFACES = ("Hard", "Clay", "Grass")
SURFACE_CODE = {"Hard": 0, "Clay": 1, "Grass": 2, "Carpet": 0}  # carpet ~ fast indoor hard

#: order of rounds inside a tournament (qualifying rounds never appear in this table)
ROUND_ORDER = {"RR": 0, "R128": 1, "R64": 2, "R32": 3, "R16": 4, "QF": 5, "SF": 6, "BR": 7, "F": 8,
               "ER": 0}


def load_matches(path: Path | str = DATA_PATH) -> pd.DataFrame:
    """Read the parquet written by ``scripts/fetch_data.py`` and add helper columns.

    Added columns:

    * ``surface_code`` - 0 hard (incl. carpet), 1 clay, 2 grass
    * ``round_order`` - sortable round index
    * ``walkover`` - True for walkovers / defaults / unplayed matches (no rating info)
    * ``year``
    Rows are returned in a strict chronological order (date, tournament, round).
    """
    df = pd.read_parquet(path)
    df = df[df["surface"].isin(list(SURFACE_CODE))].copy()
    df["surface"] = df["surface"].replace({"Carpet": "Hard"})
    df["tourney_name"] = df["tourney_name"].replace({"Us Open": "US Open"})
    df["surface_code"] = df["surface"].map(SURFACE_CODE).astype(int)
    df["round_order"] = df["round"].map(ROUND_ORDER).fillna(0).astype(int)
    score = df["score"].fillna("").astype(str)
    df["walkover"] = score.str.contains("W/O|DEF|Def|unfinished|In Progress|NA", regex=True) | (
        score.str.strip() == ""
    )
    df["best_of"] = df["best_of"].fillna(3).astype(int)
    df["year"] = df["tourney_date"].dt.year
    df = df.sort_values(["tourney_date", "tourney_id", "round_order", "match_num"], kind="stable")
    return df.reset_index(drop=True)


def games_won(score: str) -> tuple[int, int]:
    """Total games won by (winner, loser) parsed from a score like ``6-4 3-6 7-6(5)``."""
    w = lo = 0
    for token in str(score).split():
        token = token.split("(")[0]
        if "-" not in token:
            continue
        a, _, b = token.partition("-")
        if a.isdigit() and b.isdigit():
            w += int(a)
            lo += int(b)
    return w, lo


def fill_rank(rank: pd.Series, missing: float = 2000.0) -> np.ndarray:
    """ATP rank with unranked players set to ``missing``."""
    return rank.fillna(missing).clip(lower=1).to_numpy(float)
