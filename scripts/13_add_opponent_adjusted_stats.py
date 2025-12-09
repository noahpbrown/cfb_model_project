import os
from typing import List

import pandas as pd


DATA_DIR = "data"
FEATURES_PATH = os.path.join(DATA_DIR, "team_week_features_full_2019_2024.csv")


def load_full_features() -> pd.DataFrame:
    """
    Load the full team-week feature file (with SOS + priors).
    This file currently actually has seasons 2019, 2021, 2022, 2023, 2024, 2025.
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
        # Keep only columns we need
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
    From games (home/away), build a flattened table with one row per (team, opponent, season, week).
    """
    rows = []

    for _, row in games.iterrows():
        season = row["season"]
        week = row["week"]
        home = row["home_team"]
        away = row["away_team"]

        # Home team perspective
        rows.append({
            "season": season,
            "week": week,
            "team": home,
            "opponent": away,
        })
        # Away team perspective
        rows.append({
            "season": season,
            "week": week,
            "team": away,
            "opponent": home,
        })

    team_opp = pd.DataFrame(rows)
    print(f"Team-opponent table shape: {team_opp.shape}")
    return team_opp


def compute_opponent_averages(
    features: pd.DataFrame,
    team_opp: pd.DataFrame,
) -> pd.DataFrame:
    """
    For each (season, week, team), compute the average offensive and defensive EPA
    of its opponents *through that week*.

    We will use:
      - opp_off_ppa_cum_mean: average opponent offensive EPA (cumulative)
      - opp_def_ppa_cum_mean: average opponent defensive EPA (cumulative)
    """
    # We'll take just the columns we need from the features file
    opp_feats = features[["season", "week", "team", "off_ppa_cum", "def_ppa_cum"]].copy()

    # Rename 'team' to 'opponent' so we can merge on that
    opp_feats = opp_feats.rename(columns={
        "team": "opponent",
        "off_ppa_cum": "opp_off_ppa_cum",
        "def_ppa_cum": "opp_def_ppa_cum",
    })

    # Merge team-opponent table with opponent features
    team_opp_with_stats = team_opp.merge(
        opp_feats,
        on=["season", "week", "opponent"],
        how="left",
    )

    print("After merging opponent EPA stats, shape:", team_opp_with_stats.shape)

    # Group by (season, week, team) and average the opponent EPA values
    agg = (
        team_opp_with_stats
        .groupby(["season", "week", "team"], as_index=False)[
            ["opp_off_ppa_cum", "opp_def_ppa_cum"]
        ]
        .mean()
    )

    agg = agg.rename(columns={
        "opp_off_ppa_cum": "opp_off_ppa_cum_mean",
        "opp_def_ppa_cum": "opp_def_ppa_cum_mean",
    })

    print("Opponent EPA means per (season, week, team) shape:", agg.shape)
    return agg


def add_opponent_adjusted_features(
    features: pd.DataFrame,
    opp_avg: pd.DataFrame,
) -> pd.DataFrame:
    """
    Join opponent averages back onto the features and create:
      - off_ppa_adj_cum = off_ppa_cum - opp_def_ppa_cum_mean
      - def_ppa_adj_cum = def_ppa_cum - opp_off_ppa_cum_mean
    """
    merged = features.merge(
        opp_avg,
        on=["season", "week", "team"],
        how="left",
    )

    print("Features + opponent averages shape:", merged.shape)

    # Compute adjusted stats
    # If a team has no opponent data yet (no games), these will be NaN
    merged["off_ppa_adj_cum"] = merged["off_ppa_cum"] - merged["opp_def_ppa_cum_mean"]
    merged["def_ppa_adj_cum"] = merged["def_ppa_cum"] - merged["opp_off_ppa_cum_mean"]

    print("Sample rows with adjusted stats:")
    print(
        merged[
            [
                "season",
                "week",
                "team",
                "off_ppa_cum",
                "def_ppa_cum",
                "opp_def_ppa_cum_mean",
                "opp_off_ppa_cum_mean",
                "off_ppa_adj_cum",
                "def_ppa_adj_cum",
            ]
        ].head(10)
    )

    return merged


def main():
    # 1) Load full features (with SOS + priors)
    features = load_full_features()

    # 2) Load games for all seasons we care about
    #    (this includes 2025 as inference-only data)
    seasons = [2019, 2021, 2022, 2023, 2024, 2025]
    games = load_games(seasons)

    # 3) Build (season, week, team, opponent) table
    team_opp = build_team_opponent_table(games)

    # 4) Compute average opponent EPA for each (season, week, team)
    opp_avg = compute_opponent_averages(features, team_opp)

    # 5) Add opponent-adjusted EPA features
    features_with_adj = add_opponent_adjusted_features(features, opp_avg)

    # 6) Save back to the same full feature file (we are extending it)
    out_path = FEATURES_PATH
    features_with_adj.to_csv(out_path, index=False)
    print(f"\nSaved full feature set with opponent-adjusted EPA to {out_path}")
    print("New shape:", features_with_adj.shape)


if __name__ == "__main__":
    main()
