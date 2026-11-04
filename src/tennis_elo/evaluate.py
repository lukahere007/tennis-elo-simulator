"""Proper scoring rules, calibration tables and rank-based baselines."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar

EPS = 1e-12


def log_loss(p_winner: np.ndarray) -> float:
    """Mean negative log probability assigned to the actual winner."""
    p = np.clip(np.asarray(p_winner, float), EPS, 1 - EPS)
    return float(-np.mean(np.log(p)))


def brier(p_winner: np.ndarray) -> float:
    """Mean squared error of the forecast for the actual winner (binary Brier)."""
    p = np.asarray(p_winner, float)
    return float(np.mean((1.0 - p) ** 2))


def accuracy(p_winner: np.ndarray) -> float:
    """Share of matches where the winner was favoured; exact coin-flips count half."""
    p = np.asarray(p_winner, float)
    return float(np.mean((p > 0.5) + 0.5 * (p == 0.5)))


def scores(p_winner: np.ndarray) -> dict:
    """Log loss, Brier and accuracy in one dict."""
    return {"log_loss": log_loss(p_winner), "brier": brier(p_winner),
            "accuracy": accuracy(p_winner), "n": int(len(p_winner))}


def calibration_table(p_winner: np.ndarray, n_bins: int = 10, seed: int = 0) -> pd.DataFrame:
    """Reliability table. Each match is shown from a random player's point of view.

    Forecasts are symmetric, so orienting randomly avoids the trivial "winner" framing.
    """
    rng = np.random.default_rng(seed)
    p = np.asarray(p_winner, float)
    flip = rng.random(len(p)) < 0.5
    prob = np.where(flip, 1 - p, p)
    outcome = np.where(flip, 0.0, 1.0)
    bins = np.clip((prob * n_bins).astype(int), 0, n_bins - 1)
    df = pd.DataFrame({"bin": bins, "prob": prob, "outcome": outcome})
    out = df.groupby("bin").agg(mean_pred=("prob", "mean"), freq=("outcome", "mean"),
                                n=("outcome", "size")).reset_index()
    return out


def fit_rank_logit(log_rank_diff: np.ndarray, is_bo5: np.ndarray) -> tuple[float, float]:
    """Fit ``P(A wins) = sigmoid(b * x + b5 * x * bo5)``, x = log(rank_B) - log(rank_A).

    Fitted on winner-oriented rows: with no intercept the model is symmetric, so the
    likelihood of ``sigmoid(z)`` for the winner is all that is needed.
    """
    x = np.asarray(log_rank_diff, float)
    f = np.asarray(is_bo5, float)

    def nll(theta):
        z = theta[0] * x + theta[1] * x * f
        return np.mean(np.logaddexp(0.0, -z))

    res = minimize(nll, x0=[0.5, 0.0], method="BFGS")
    return float(res.x[0]), float(res.x[1])


def rank_logit_prob(log_rank_diff, is_bo5, coef: tuple[float, float]) -> np.ndarray:
    """Winner probability from :func:`fit_rank_logit` coefficients."""
    z = coef[0] * np.asarray(log_rank_diff, float) + coef[1] * np.asarray(log_rank_diff, float) * np.asarray(is_bo5, float)
    return 1.0 / (1.0 + np.exp(-z))


def fit_higher_rank_rate(winner_rank: np.ndarray, loser_rank: np.ndarray) -> float:
    """Training-period rate at which the better-ranked player wins.

    Turns the "higher-ranked player wins" rule into a probabilistic forecast:
    the favourite always gets this constant probability (ties get 0.5).
    """
    w, lo = np.asarray(winner_rank), np.asarray(loser_rank)
    decided = w != lo
    return float(np.mean(w[decided] < lo[decided]))


def higher_rank_prob(winner_rank, loser_rank, rate: float) -> np.ndarray:
    """Winner probability under the constant-rate "higher-ranked wins" baseline."""
    w, lo = np.asarray(winner_rank), np.asarray(loser_rank)
    return np.where(w < lo, rate, np.where(w > lo, 1 - rate, 0.5))


def best_constant_shrink(p_winner: np.ndarray) -> float:
    """Temperature ``t`` minimising log loss of ``sigmoid(t * logit(p))`` (diagnostic)."""
    p = np.clip(np.asarray(p_winner, float), 1e-9, 1 - 1e-9)
    z = np.log(p / (1 - p))
    res = minimize_scalar(lambda t: np.mean(np.logaddexp(0.0, -t * z)), bounds=(0.2, 3), method="bounded")
    return float(res.x)


def fit_scale(d: np.ndarray, bo5: np.ndarray, bo5_adjust: bool = True) -> float:
    """Factor ``t`` on the rating difference that minimises log loss of the winners' forecasts.

    ``t < 1`` means the ratings were over-confident (differences too large).
    """
    from .elo import prob_from_diff

    res = minimize_scalar(lambda t: log_loss(prob_from_diff(t * d, bo5, bo5_adjust)),
                          bounds=(0.2, 2.0), method="bounded")
    return float(res.x)


def rolling_scale(d: np.ndarray, bo5: np.ndarray, year: np.ndarray, valid: np.ndarray,
                  window: int = 3, bo5_adjust: bool = True) -> dict[int, float]:
    """Walk-forward recalibration of the rating scale.

    For each year Y the scale is fitted on the (already one-step-ahead) forecasts of years
    ``Y-window .. Y-1`` and then applied to year Y, so no year-Y result is used to adjust
    year-Y forecasts.
    """
    out = {}
    for y in np.unique(year):
        m = valid & (year >= y - window) & (year < y)
        out[int(y)] = fit_scale(d[m], bo5[m], bo5_adjust) if m.sum() > 1000 else 1.0
    return out
