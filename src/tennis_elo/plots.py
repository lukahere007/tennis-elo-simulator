"""Figures for the README (matplotlib, static PNG)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

SURFACE_BG = "#fcfcfb"
TEXT = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]

plt.rcParams.update({
    "figure.facecolor": SURFACE_BG, "axes.facecolor": SURFACE_BG, "savefig.facecolor": SURFACE_BG,
    "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
    "text.color": TEXT, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
    "axes.titlesize": 11, "axes.titleweight": "bold", "legend.frameon": False,
})


def calibration(calib: pd.DataFrame, path: Path) -> None:
    """Reliability diagram: forecast probability vs observed win frequency (test period)."""
    fig, ax = plt.subplots(figsize=(5.2, 5))
    ax.plot([0, 1], [0, 1], color=MUTED, lw=1, ls="--", label="perfect calibration")
    for color, (model, g) in zip(SERIES, calib.groupby("model", sort=False)):
        g = g[g.n >= 30]
        ax.plot(g.mean_pred, g.freq, color=color, lw=2, marker="o", ms=6,
                markeredgecolor=SURFACE_BG, markeredgewidth=1.5, label=model)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("forecast P(player wins)")
    ax.set_ylabel("observed win frequency")
    ax.set_title("Calibration, ATP matches 2015-2026 (out of sample)")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def title_bars(fc: pd.DataFrame, path: Path, top: int = 12) -> None:
    """Horizontal bars of pre-event title probability for the top of one Slam draw."""
    g = fc.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    colors = [SERIES[1] if c else SERIES[0] for c in g.champion]
    ax.barh(g.player, g.p_title_exact * 100, color=colors, height=0.7)
    ax.scatter(g.p_title_rank * 100, g.player, marker="|", s=160, color=TEXT, zorder=3,
               label="rank-only model")
    for y, (v, r) in enumerate(zip(g.p_title_exact * 100, g.p_title_rank * 100)):
        ax.text(max(v, r) + 0.8, y, f"{v:.1f}%", va="center", color=MUTED, fontsize=9)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("pre-tournament title probability (%)")
    name = f"{fc.tourney_name.iloc[0]} {fc.date.iloc[0].year}"
    champ = fc.player[fc.champion].iloc[0]
    ax.set_title(f"{name}: pre-tournament title chances", loc="left")
    from matplotlib.patches import Patch
    handles = [Patch(color=SERIES[0], label="surface-blend Elo"),
               Patch(color=SERIES[1], label=f"eventual champion ({champ})"),
               ax.collections[0]]
    ax.legend(handles=handles, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


DEFAULT_PLAYERS = ["Roger Federer", "Rafael Nadal", "Novak Djokovic", "Andy Murray",
                   "Carlos Alcaraz", "Jannik Sinner"]


def trajectories(df: pd.DataFrame, res, params, path: Path, players=DEFAULT_PLAYERS) -> None:
    """Blended surface rating of the Big Four and their successors, one panel per surface."""
    w = params.surface_weight
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
    for k, (ax, surface) in enumerate(zip(axes, ("Hard", "Clay", "Grass"))):
        on = (df["surface"] == surface).to_numpy() & ~df["walkover"].to_numpy()
        for color, name in zip(SERIES, players):
            as_w = on & (df["winner_name"] == name).to_numpy()
            as_l = on & (df["loser_name"] == name).to_numpy()
            rating = np.where(as_w, (1 - w) * res.pre_w_overall + w * res.pre_w_surface,
                              (1 - w) * res.pre_l_overall + w * res.pre_l_surface)
            m = as_w | as_l
            s = pd.Series(rating[m], index=df["tourney_date"][m]).groupby(level=0).last()
            if s.empty:
                continue
            s = s.rolling(5, min_periods=1).mean()
            ax.plot(s.index, s.values, color=color, lw=2, label=name.split()[-1])
        ax.set_title(surface)
        if k == 0:
            ax.set_ylabel("surface-blended Elo (pre-match)")
    axes[0].legend(loc="lower left", ncol=2, fontsize=9)
    fig.suptitle("Rating trajectories by surface", fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def slam_reliability(slam_df: pd.DataFrame, path: Path) -> None:
    """Champion's pre-event title probability for every Slam, with the favourite's."""
    d = slam_df.reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(11, 3.8))
    x = np.arange(len(d))
    ax.bar(x, d.p_favourite * 100, color=GRID, width=0.75, label="favourite's title probability")
    ax.scatter(x, d.p_champion * 100, color=SERIES[1], s=36, zorder=3,
               label="eventual champion's title probability")
    ax.set_xticks(x)
    short = {"Australian Open": "AO", "Roland Garros": "RG", "Wimbledon": "W", "Us Open": "USO",
             "US Open": "USO"}
    ax.set_xticklabels([f"{short.get(e.rsplit(' ', 1)[0], e)} {e[-2:]}" for e in d.event],
                       rotation=90, fontsize=8)
    ax.set_ylabel("pre-event probability (%)")
    ax.set_title("Grand Slams 2015-2026: how much did the model expect the eventual champion?")
    ax.legend(loc="upper right")
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
