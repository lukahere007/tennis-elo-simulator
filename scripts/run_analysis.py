"""End-to-end analysis: tune on 2000-2014, walk-forward test on 2015+, simulate Slams.

Usage::

    python scripts/run_analysis.py            # full run: tuning + 20k sims per Slam (~5 min)
    python scripts/run_analysis.py --fast     # reuse results/params.json, 2k sims (CI)

Outputs go to ``results/`` (CSV/JSON) and ``figures/`` (PNG).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tennis_elo import plots  # noqa: E402
from tennis_elo.data import fill_rank, load_matches  # noqa: E402
from tennis_elo.elo import EloParams, blended_prob, prob_from_diff, run_elo, winner_rating_diff  # noqa: E402
from tennis_elo.evaluate import (calibration_table, fit_higher_rank_rate, fit_rank_logit,  # noqa: E402
                                 fit_scale, higher_rank_prob, rank_logit_prob, rolling_scale, scores)
from tennis_elo.simulate import exact_title_probs, multiclass_brier  # noqa: E402
from tennis_elo.slams import forecast_slam, random_seeded_draw, slam_events  # noqa: E402
from tennis_elo.tuning import tune  # noqa: E402

TRAIN = (2000, 2014)
TEST_START = 2015
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

#: name -> (forecast column, how to build the params)
VARIANTS = {
    "Elo, 538 defaults (overall)": ("p_overall", None),
    "Elo tuned (overall only)": ("p_overall", dict(variant="overall")),
    "Elo tuned (surface only)": ("p_surface", dict(variant="surface")),
    "Surface-blend Elo, no Bo5 conversion": ("p_blend", dict(variant="blend", bo5_adjust=False)),
    "Surface-blend Elo": ("p_blend", dict(variant="blend")),
    "Surface-blend Elo + margin of victory": ("p_blend", dict(variant="blend", fit_mov=True)),
}
BLEND = "Surface-blend Elo"
FINAL = "Surface-blend Elo + walk-forward recalibration"


def fit_params(df, train_mask, reuse: bool) -> dict[str, EloParams]:
    path = RESULTS / "params.json"
    if reuse and path.exists():
        stored = json.loads(path.read_text())
        return {k: EloParams(**v) for k, v in stored.items()}
    params = {}
    for name, (_, spec) in VARIANTS.items():
        t = time.time()
        if spec is None:
            params[name] = EloParams()
            continue
        base = EloParams(bo5_adjust=spec.get("bo5_adjust", True))
        p, loss = tune(df, train_mask, spec["variant"], base=base, fit_mov=spec.get("fit_mov", False))
        params[name] = p
        print(f"tuned {name}: train log loss {loss:.4f} ({time.time() - t:.0f}s) {p}")
    path.write_text(json.dumps({k: v.to_dict() for k, v in params.items()}, indent=2))
    return params


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true", help="reuse stored params, fewer simulations")
    ap.add_argument("--skip-tuning", action="store_true", help="reuse results/params.json")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="write results/ and figures/ under this directory instead of the repo")
    args = ap.parse_args()
    global RESULTS, FIGURES
    if args.out_dir is not None:
        params_src = RESULTS / "params.json"
        RESULTS, FIGURES = args.out_dir / "results", args.out_dir / "figures"
        RESULTS.mkdir(parents=True, exist_ok=True)
        if params_src.exists():
            (RESULTS / "params.json").write_text(params_src.read_text())
    n_sims = 2_000 if args.fast else 20_000
    RESULTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    df = load_matches()
    played = ~df["walkover"].to_numpy()
    year = df["year"].to_numpy()
    train = played & (year >= TRAIN[0]) & (year <= TRAIN[1])
    test = played & (year >= TEST_START)
    print(f"{len(df):,} matches {df.tourney_date.min().date()}..{df.tourney_date.max().date()}; "
          f"train {train.sum():,}, test {test.sum():,}")

    params = fit_params(df, train, args.fast or args.skip_tuning)
    slams = slam_events(df, (TEST_START, 9999))
    snap_rows = {row for _, row in slams} | {len(df) - 1}
    runs = {name: run_elo(df, p, snapshot_at=snap_rows if name == BLEND else None)
            for name, p in params.items()}

    # ---------------- baselines (fit on the training period) ----------------
    w_rank = fill_rank(df["winner_rank"])
    l_rank = fill_rank(df["loser_rank"])
    ranked = df["winner_rank"].notna().to_numpy() & df["loser_rank"].notna().to_numpy()
    is_bo5 = (df["best_of"].to_numpy() == 5)
    log_diff = np.log(l_rank) - np.log(w_rank)
    rate = fit_higher_rank_rate(w_rank[train & ranked], l_rank[train & ranked])
    coef = fit_rank_logit(log_diff[train], is_bo5[train])
    probs = {
        "Higher-ranked player wins (constant rate)": higher_rank_prob(w_rank, l_rank, rate),
        "Logistic on log-rank difference": rank_logit_prob(log_diff, is_bo5, coef),
    }
    for name, (col, _) in VARIANTS.items():
        probs[name] = getattr(runs[name], col)
    # walk-forward recalibration of the blend: rating-difference scale fitted on the
    # previous three years' one-step-ahead forecasts, applied to the next year
    blend_p = params[BLEND]
    d = winner_rating_diff(runs[BLEND], blend_p.surface_weight)
    scale_by_year = rolling_scale(d, is_bo5, year, played, window=3)
    probs[FINAL] = prob_from_diff(np.array([scale_by_year[y] for y in year]) * d, is_bo5)

    # ---------------- walk-forward match-level metrics ----------------
    subsets = {"all": test, "Grand Slams (best of 5)": test & (df["tourney_level"] == "G").to_numpy()}
    for s in ("Hard", "Clay", "Grass"):
        subsets[s] = test & (df["surface"] == s).to_numpy()
    rows = []
    for model, p in probs.items():
        for subset, m in subsets.items():
            rows.append({"model": model, "subset": subset, **scores(p[m])})
    metrics = pd.DataFrame(rows)
    metrics.to_csv(RESULTS / "match_metrics.csv", index=False)
    print(metrics[metrics.subset == "all"].to_string(index=False))

    calib = []
    for model in (BLEND, FINAL, "Logistic on log-rank difference"):
        t = calibration_table(probs[model][test])
        t["model"] = model
        calib.append(t)
    calib = pd.concat(calib)
    calib.to_csv(RESULTS / "calibration.csv", index=False)

    final_p = blend_p
    summary = {
        "data_range": [str(df.tourney_date.min().date()), str(df.tourney_date.max().date())],
        "n_train": int(train.sum()), "n_test": int(test.sum()),
        "final_params": final_p.to_dict(),
        "rank_logit_coef": coef, "higher_rank_rate": rate,
        "scale_train_period": fit_scale(d[train], is_bo5[train]),
        "scale_test_period_hindsight": fit_scale(d[test], is_bo5[test]),
        "walk_forward_scale_by_year": {y: scale_by_year[y] for y in range(TEST_START, int(year.max()) + 1)},
    }

    # ---------------- Grand Slam Monte Carlo ----------------
    res = runs[BLEND]
    per_event, slam_rows = [], []
    for tid, row in slams:
        scale = scale_by_year[int(year[row])]
        fc = forecast_slam(df, res, tid, row, final_p, coef, n_sims=n_sims, seed=row, scale=scale)
        if fc is None:
            print(f"skip {tid}: draw could not be rebuilt")
            continue
        raw = forecast_slam(df, res, tid, row, final_p, coef, n_sims=10, seed=row)
        raw_champ = raw.p_title_exact[raw.champion].iloc[0]
        per_event.append(fc)
        champ = int(np.flatnonzero(fc["champion"])[0])
        fav = fc.iloc[0]
        fav_rank = fc.sort_values("p_title_rank", ascending=False).iloc[0]
        slam_rows.append({
            "tourney_id": tid, "event": f"{fc.tourney_name[0]} {fc.date[0].year}",
            "champion": fc.player[champ], "p_champion": fc.p_title_exact[champ],
            "champion_elo_rank": champ + 1,
            "favourite": fav.player, "p_favourite": fav.p_title_exact,
            "favourite_won": bool(fav.champion),
            "max_mc_error": float(np.max(np.abs(fc.p_title_mc - fc.p_title_exact))),
            "brier_elo": multiclass_brier(fc.p_title_exact.to_numpy(), champ),
            "brier_rank": multiclass_brier(fc.p_title_rank.to_numpy(), champ),
            "logloss_elo": -np.log(max(fc.p_title_exact[champ], 1e-12)),
            "p_champion_uncalibrated": raw_champ,
            "brier_elo_uncalibrated": multiclass_brier(raw.p_title_exact.to_numpy(),
                                                       int(np.flatnonzero(raw.champion)[0])),
            "logloss_elo_uncalibrated": -np.log(max(raw_champ, 1e-12)),
            "p_favourite_uncalibrated": raw.p_title_exact.iloc[0],
            "favourite_uncalibrated_won": bool(raw.champion.iloc[0]),
            "logloss_rank": -np.log(max(fc.p_title_rank[champ], 1e-12)),
            "p_champion_rank_model": fc.p_title_rank[champ],
            "favourite_rank_model": fav_rank.player,
            "favourite_rank_model_won": bool(fav_rank.champion),
            "p_favourite_rank_model": fav_rank.p_title_rank,
        })
    slam_df = pd.DataFrame(slam_rows)
    slam_df.to_csv(RESULTS / "slam_forecasts.csv", index=False)
    all_fc = pd.concat(per_event)
    all_fc.to_parquet(RESULTS / "slam_player_probs.parquet", index=False)
    summary["slams"] = {
        "n": len(slam_df),
        "favourite_win_rate_elo": float(slam_df.favourite_won.mean()),
        "expected_favourite_wins_elo": float(slam_df.p_favourite.sum()),
        "favourite_wins_elo": int(slam_df.favourite_won.sum()),
        "favourite_win_rate_rank": float(slam_df.favourite_rank_model_won.mean()),
        "expected_favourite_wins_rank": float(slam_df.p_favourite_rank_model.sum()),
        "mean_brier_elo": float(slam_df.brier_elo.mean()),
        "mean_brier_rank": float(slam_df.brier_rank.mean()),
        "mean_brier_elo_uncalibrated": float(slam_df.brier_elo_uncalibrated.mean()),
        "mean_logloss_elo_uncalibrated": float(slam_df.logloss_elo_uncalibrated.mean()),
        "expected_favourite_wins_uncalibrated": float(slam_df.p_favourite_uncalibrated.sum()),
        "favourite_wins_uncalibrated": int(slam_df.favourite_uncalibrated_won.sum()),
        "mean_brier_uniform": float(1 - 1 / 128),
        "mean_logloss_elo": float(slam_df.logloss_elo.mean()),
        "mean_logloss_rank": float(slam_df.logloss_rank.mean()),
        "mean_logloss_uniform": float(np.log(128)),
        "max_mc_error": float(slam_df.max_mc_error.max()),
        "champion_in_elo_top3": int((slam_df.champion_elo_rank <= 3).sum()),
        "champion_in_elo_top8": int((slam_df.champion_elo_rank <= 8).sum()),
        "worst_champion_elo_rank": int(slam_df.champion_elo_rank.max()),
        "n_sims": n_sims,
    }
    print(slam_df[["event", "favourite", "p_favourite", "champion", "p_champion"]].to_string(index=False))

    # ---------------- forecast for the next Slam after the data ends ----------------
    state = res.snapshots[len(df) - 1]
    last_date = df.tourney_date.max()
    recent = df[df.tourney_date >= last_date - pd.Timedelta(days=270)]
    active = pd.unique(np.concatenate([recent.winner_id, recent.loser_id]))
    names = pd.concat([df[["winner_id", "winner_name"]].set_axis(["id", "name"], axis=1),
                       df[["loser_id", "loser_name"]].set_axis(["id", "name"], axis=1)]
                      ).drop_duplicates("id", keep="last").set_index("id")["name"]
    next_slam = next_slam_after(last_date)
    pos = np.array([res.players[i] for i in active])
    surf = {"Hard": 0, "Clay": 1, "Grass": 2}[next_slam[1]]
    w = final_p.surface_weight
    strength = (1 - w) * state.overall[pos] + w * state.surface[pos, surf]
    order = np.argsort(-strength)[:128]
    field = pos[order]
    a, b = np.meshgrid(field, field, indexing="ij")
    next_scale = scale_by_year.get(next_slam[2], fit_scale(*[x[played & (year >= next_slam[2] - 3)] for x in (d, is_bo5)]))
    P = blended_prob(state, a, b, surf, w, 5, final_p.bo5_adjust, next_scale)
    rng = np.random.default_rng(2026)
    n_draws = 100 if args.fast else 1000
    title = np.zeros(len(field))
    for _ in range(n_draws):
        draw = random_seeded_draw(np.arange(len(field)), 32, rng)
        title[draw] += exact_title_probs(P[np.ix_(draw, draw)])
    title /= n_draws
    inv = {v: k for k, v in res.players.items()}
    nxt = pd.DataFrame({"player": [names[inv[i]] for i in field], "rating": strength[order],
                        "p_title": title}).sort_values("p_title", ascending=False)
    nxt.to_csv(RESULTS / "next_slam_forecast.csv", index=False)
    summary["next_slam"] = {"event": f"{next_slam[0]} {next_slam[2]}", "surface": next_slam[1],
                            "data_through": str(last_date.date()), "n_draws": n_draws,
                            "scale": next_scale,
                            "top": nxt.head(8).to_dict("records")}
    print(nxt.head(10).to_string(index=False))

    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=2, default=float))

    # ---------------- figures ----------------
    plots.calibration(calib, FIGURES / "calibration.png")
    latest = per_event[-1]
    plots.title_bars(latest, FIGURES / "slam_title_probs.png")
    plots.trajectories(df, res, final_p, FIGURES / "surface_trajectories.png")
    plots.slam_reliability(slam_df, FIGURES / "slam_favourites.png")
    print("done")


def next_slam_after(date: pd.Timestamp) -> tuple[str, str, int]:
    """Name, surface and year of the first Grand Slam starting after ``date``."""
    calendar = [(1, 12, "Australian Open", "Hard"), (5, 24, "Roland Garros", "Clay"),
                (6, 29, "Wimbledon", "Grass"), (8, 24, "US Open", "Hard")]
    for year in (date.year, date.year + 1):
        for month, day, name, surface in calendar:
            if pd.Timestamp(year, month, day) > date:
                return name, surface, year
    raise AssertionError


if __name__ == "__main__":
    main()
