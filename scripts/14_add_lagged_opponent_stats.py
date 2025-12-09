import os
from typing import List

import pandas as pd

DATA_DIR = "data"
FEATURES_PATH = os.path.join(DATA_DIR, "team_week_features_full_2019_2024.csv")


def load_full_features() -> pd.DataFrame:
    """
    Load the full team-week feature file (with SOS + priors + same-week adj EPA).
    This file now includes seasons 2019, 2021, 2022, 2023, 2024, and 2025.
    """
    df = pd.read_csv(FEATURES_PATH)
    print(f"Loaded full team-week features from {FEATURES_PATH} with shape {df.shape}")
    return df


def load_games(seasons: List[int]) -> pd.DataFrame:
    """
    Load per-season games files and concatenate them.
    We only need season, week, home_team, away_team.
    """
    all_games = []

    for year in seasons:
        path = os.path.join(DATA_DIR, f"games_{year}.csv")
        if not os.path.exists(path):
            print(f"[WARN] Games file not found for {year}: {path} (skipping)")
            continue

        g = pd.read_csv(path)
        g = g[["season", "week", "home_team", "away_team"]]
        all_games.append(g)
        print(f"[{year}] games shape: {g.shape}")

    if not all_games:
        raise RuntimeError("No games files loaded! Did you run 06_fetch_games_multiseason.py?")

    games = pd.concat(all_games, ignore_index=True)
    print(f"Combined games shape: {games.shape}")
    return games


def build_team_opponent_table(games: pd.DataFrame) -> pd.DataFrame:
    """
    From games (home/away), build a flattened table with one row per
    (season, week, team, opponent).
    """
    rows = []

    for _, row in games.iterrows():
        season = row["season"]
        week = row["week"]
        home = row["home_team"]
        away = row["away_team"]

        # Home perspective
        rows.append({
            "season": season,
            "week": week,
            "team": home,
            "opponent": away,
        })
        # Away perspective
        rows.append({
            "season": season,
            "week": week,
            "team": away,
            "opponent": home,
        })

    team_opp = pd.DataFrame(rows)
    print(f"Team-opponent table shape: {team_opp.shape}")
    return team_opp


def add_prev_week_epa(features: pd.DataFrame) -> pd.DataFrame:
    """
    For each (season, team, week), compute opponent EPA *through the previous week*.

    We do this by:
      - sorting by (season, team, week)
      - shifting off_ppa_cum and def_ppa_cum by 1 within each (season, team) group

    This gives:
      off_ppa_cum_prev  = team's cumulative offensive EPA up to week-1
      def_ppa_cum_prev  = team's cumulative defensive EPA up to week-1
    """
    df = features.copy()
    df = df.sort_values(["season", "team", "week"])

    df["off_ppa_cum_prev"] = (
        df.groupby(["season", "team"])["off_ppa_cum"].shift(1)
    )
    df["def_ppa_cum_prev"] = (
        df.groupby(["season", "team"])["def_ppa_cum"].shift(1)
    )

    print("Preview of prev-week EPA columns:")
    print(
        df[
            ["season", "week", "team", "off_ppa_cum", "off_ppa_cum_prev", "def_ppa_cum", "def_ppa_cum_prev"]
        ].head(15)
    )

    return df


def compute_lagged_opponent_averages(
    features_with_prev: pd.DataFrame,
    team_opp: pd.DataFrame,
) -> pd.DataFrame:
    """
    For each (season, week, team), compute the average offensive and defensive EPA
    of its opponents *through the previous week*.

    Uses:
      opp_off_ppa_cum_mean_lagged
      opp_def_ppa_cum_mean_lagged
    """
    # Take only the prev-week EPA columns for opponents
    opp_prev = features_with_prev[
        ["season", "week", "team", "off_ppa_cum_prev", "def_ppa_cum_prev"]
    ].copy()

    # Rename "team" -> "opponent" so we can merge by opponent
    opp_prev = opp_prev.rename(columns={
        "team": "opponent",
        "off_ppa_cum_prev": "opp_off_ppa_cum_prev",
        "def_ppa_cum_prev": "opp_def_ppa_cum_prev",
    })

    # Merge team-opponent table with opponent prev-week EPA
    team_opp_with_prev = team_opp.merge(
        opp_prev,
        on=["season", "week", "opponent"],
        how="left",
    )

    print("After merging opponent prev-week EPA, shape:", team_opp_with_prev.shape)

    # Now aggregate per (season, week, team)
    agg = (
        team_opp_with_prev
        .groupby(["season", "week", "team"], as_index=False)[
            ["opp_off_ppa_cum_prev", "opp_def_ppa_cum_prev"]
        ]
        .mean()
    )

    agg = agg.rename(columns={
        "opp_off_ppa_cum_prev": "opp_off_ppa_cum_mean_lagged",
        "opp_def_ppa_cum_prev": "opp_def_ppa_cum_mean_lagged",
    })

    print("Lagged opponent EPA means per (season, week, team) shape:", agg.shape)
    return agg


def add_lagged_adjusted_features(
    features_with_prev: pd.DataFrame,
    opp_avg_lagged: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge lagged opponent averages back into the main features DataFrame and create:
      - off_ppa_adj_lagged_cum = off_ppa_cum - opp_def_ppa_cum_mean_lagged
      - def_ppa_adj_lagged_cum = def_ppa_cum - opp_off_ppa_cum_mean_lagged
    """
    merged = features_with_prev.merge(
        opp_avg_lagged,
        on=["season", "week", "team"],
        how="left",
    )

    print("Features + lagged opponent averages shape:", merged.shape)

    merged["off_ppa_adj_lagged_cum"] = (
        merged["off_ppa_cum"] - merged["opp_def_ppa_cum_mean_lagged"]
    )
    merged["def_ppa_adj_lagged_cum"] = (
        merged["def_ppa_cum"] - merged["opp_off_ppa_cum_mean_lagged"]
    )

    print("Sample rows with lagged adjusted stats:")
    print(
        merged[
            [
                "season",
                "week",
                "team",
                "off_ppa_cum",
                "def_ppa_cum",
                "opp_def_ppa_cum_mean_lagged",
                "opp_off_ppa_cum_mean_lagged",
                "off_ppa_adj_lagged_cum",
                "def_ppa_adj_lagged_cum",
            ]
        ].head(15)
    )

    return merged


def main():
    # 1) Load features with existing columns (off_ppa_cum, def_ppa_cum, etc.)
    features = load_full_features()

    # 2) Add prev-week EPA columns per (season, team)
    features_with_prev = add_prev_week_epa(features)

    # 3) Load games for all seasons (including 2025 for inference)
    seasons = [2019, 2021, 2022, 2023, 2024, 2025]
    games = load_games(seasons)

    # 4) Build team-opponent table
    team_opp = build_team_opponent_table(games)

    # 5) Compute lagged opponent EPA means
    opp_avg_lagged = compute_lagged_opponent_averages(features_with_prev, team_opp)

    # 6) Add lagged opponent-adjusted EPA features
    features_with_lagged_adj = add_lagged_adjusted_features(features_with_prev, opp_avg_lagged)

    # 7) Save back to same full feature path (we're extending the file, not replacing old columns)
    out_path = FEATURES_PATH
    features_with_lagged_adj.to_csv(out_path, index=False)
    print(f"\nSaved full feature set with lagged opponent-adjusted EPA to {out_path}")
    print("New shape:", features_with_lagged_adj.shape)


if __name__ == "__main__":
    main()
