"""Download ATP tour-level match results from Jeff Sackmann's ``tennis_atp`` repo.

Data: https://github.com/JeffSackmann/tennis_atp
Licence: CC BY-NC-SA 4.0 (Jeff Sackmann / Tennis Abstract). Non-commercial use only;
derivative datasets must carry the same licence.

The script downloads ``atp_matches_YYYY.csv`` for every year from ``--start`` up to the
latest year that exists, keeps the columns the models need, and writes a single
parquet file. Raw CSVs are cached under ``data/raw/`` (git-ignored).

Usage::

    python scripts/fetch_data.py                # 1995 .. latest available year
    python scripts/fetch_data.py --start 2000
"""

from __future__ import annotations

import argparse
import datetime as dt
import io
import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd

#: Original source first. In late 2026 JeffSackmann/tennis_atp stopped being publicly
#: reachable, so public mirrors of the same files are tried next (the data and its
#: CC BY-NC-SA 4.0 licence are still Jeff Sackmann's).
CANDIDATE_REPOS = [
    "JeffSackmann/tennis_atp",
    "zvonimir-rezo/tennis_atp",
    "thekasser/tennis-wta-atp",
]
RAW_URL = "https://raw.githubusercontent.com/{repo}/{branch}/{path}"
FILE_RE = re.compile(r"(?:^|/)atp_matches_(\d{4})\.csv$")

KEEP_COLUMNS = [
    "tourney_id",
    "tourney_name",
    "surface",
    "draw_size",
    "tourney_level",
    "tourney_date",
    "match_num",
    "winner_id",
    "winner_seed",
    "winner_name",
    "winner_rank",
    "winner_rank_points",
    "loser_id",
    "loser_seed",
    "loser_name",
    "loser_rank",
    "loser_rank_points",
    "score",
    "best_of",
    "round",
    "minutes",
]

ROOT = Path(__file__).resolve().parents[1]


def download(url: str, retries: int = 4, api: bool = False) -> bytes | None:
    """Return the body of ``url``, ``None`` on HTTP 404, retrying transient errors."""
    headers = {"User-Agent": "tennis-elo-simulator"}
    token = os.environ.get("GITHUB_TOKEN")
    if api and token:
        headers["Authorization"] = f"Bearer {token}"
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=60) as resp:
                return resp.read()
        except urllib.error.HTTPError as err:
            if err.code in (404, 451):
                return None
            if attempt == retries:
                raise
        except urllib.error.URLError:
            if attempt == retries:
                raise
        time.sleep(2 ** (attempt + 1))
    return None


def list_match_files(repo: str) -> tuple[str, dict[int, str]] | None:
    """Return (branch, {year: path}) of tour-level ATP match files in ``repo``."""
    meta = download(f"https://api.github.com/repos/{repo}", retries=2, api=True)
    if not meta:
        return None
    branch = json.loads(meta)["default_branch"]
    tree = download(f"https://api.github.com/repos/{repo}/git/trees/{branch}?recursive=1",
                    retries=2, api=True)
    if not tree:
        return None
    files = {}
    for item in json.loads(tree).get("tree", []):
        m = FILE_RE.search(item["path"])
        if m and item["type"] == "blob" and "wta" not in item["path"].lower():
            files.setdefault(int(m.group(1)), item["path"])
    return branch, files


def choose_source(start: int, end: int) -> tuple[str, str, dict[int, str]]:
    """Pick the candidate repo covering the most years in ``[start, end]``."""
    best = None
    for repo in CANDIDATE_REPOS:
        found = list_match_files(repo)
        if not found:
            print(f"{repo}: unavailable")
            continue
        branch, files = found
        years = sorted(y for y in files if start <= y <= end)
        print(f"{repo}@{branch}: {len(years)} years" + (f" ({years[0]}-{years[-1]})" if years else ""))
        key = (len(years), max(years, default=0))
        if years and (best is None or key > best[0]):
            best = (key, repo, branch, files)
    if best is None:
        raise RuntimeError("no source repository with ATP match files is reachable")
    _, repo, branch, files = best
    return repo, branch, {y: p for y, p in files.items() if start <= y <= end}


def fetch_years(start: int, end: int, raw_dir: Path) -> pd.DataFrame:
    """Download (or read from cache) every available year in ``[start, end]``."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    repo, branch, files = choose_source(start, end)
    print(f"source: {repo}@{branch}")
    (raw_dir / "SOURCE.txt").write_text(f"{repo}@{branch}\n")
    frames = []
    for year, path in sorted(files.items()):
        cache = raw_dir / f"atp_matches_{year}.csv"
        if cache.exists():
            body = cache.read_bytes()
        else:
            body = download(RAW_URL.format(repo=repo, branch=branch, path=path))
            if body is None:
                print(f"{year}: not available (404), skipping")
                continue
            cache.write_bytes(body)
        df = pd.read_csv(io.BytesIO(body), low_memory=False)
        print(f"{year}: {len(df):,} matches")
        frames.append(df)
    if not frames:
        raise RuntimeError("no data downloaded")
    return pd.concat(frames, ignore_index=True)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Keep needed columns, parse dates and sort chronologically."""
    df = df[[c for c in KEEP_COLUMNS if c in df.columns]].copy()
    df["tourney_date"] = pd.to_datetime(df["tourney_date"].astype(str), format="%Y%m%d")
    for col in ("winner_id", "loser_id"):
        df[col] = df[col].astype("int64")
    for col in ("winner_name", "loser_name", "tourney_name", "surface", "tourney_level",
                "round", "score", "tourney_id", "winner_seed", "loser_seed"):
        df[col] = df[col].astype("string")
    df = df.sort_values(["tourney_date", "tourney_id", "match_num"], kind="stable")
    return df.reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--start", type=int, default=1995)
    parser.add_argument("--end", type=int, default=dt.date.today().year)
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "atp_matches.parquet")
    args = parser.parse_args()

    df = clean(fetch_years(args.start, args.end, ROOT / "data" / "raw"))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, index=False, compression="zstd")
    print(f"wrote {len(df):,} matches "
          f"({df.tourney_date.min().date()} .. {df.tourney_date.max().date()}) -> {args.out}")


if __name__ == "__main__":
    main()
