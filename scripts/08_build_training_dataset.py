import os
from typing import List

import cfbd
import pandas as pd

SEASONS = [2019, 2021, 2022, 2023, 2024]


def get_cfbd_client() -> cfbd.ApiClient:
    api_key = os.getenv("CFBD_API_KEY")
    if api_key is None:
        raise ValueError("CFBD_API_KEY environment variable not found!")

    configuration = cfbd.Configuration()
    configuration.access_token = api_key
    return cfbd.ApiClient(configuration)


def load_team_week_features(path: str = "data/team_week_features_full_2019_2024.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"Features shape: {df.shape}")
    print(df.head())
    return df


def load_ratings(path: str = "data/ratings_2019_2024.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"Ratings shape: {df.shape}")
    return df


def fetch_fbs_teams_by_season(teams_api: cfbd.TeamsApi, seasons: List[int]):
    """
    Returns:
        fbs_by_season: dict[season] -> set of FBS team names
    """
    fbs_by_season = {}
    for year in seasons:
        print(f"Fetching FBS teams for {year}...")
        teams = teams_api.get_fbs_teams(year=year)
        names = {t.school for t in teams}
        fbs_by_season[year] = names
        print(f"[{year}] FBS teams: {len(names)}")
    return fbs_by_season


def main():
    os.makedirs("data", exist_ok=True)

    # 1) Load full feature set (advanced stats + SoS + priors + resume)
    features = load_team_week_features()

    # 2) Load ratings (linear system ratings for each season/week/team)
    ratings = load_ratings()

    # 3) Merge ratings onto features
    merged = features.merge(
        ratings,
        on=["season", "week", "team"],
        how="inner",  # inner: only keep rows that have both features and ratings
    )
    print(f"Merged features + ratings shape: {merged.shape}")

    # 4) Coerce returning production columns to numeric
    for col in ["rp_total", "rp_offense", "rp_defense"]:
        if col in merged.columns:
            merged[col] = pd.to_numeric(merged[col], errors="coerce")

    # 5) Filter down to FBS teams only (by season)
    api_client = get_cfbd_client()
    teams_api = cfbd.TeamsApi(api_client)
    fbs_by_season = fetch_fbs_teams_by_season(teams_api, SEASONS)

    def is_fbs(row):
        season = int(row["season"])
        team = row["team"]
        return team in fbs_by_season.get(season, set())

    mask_fbs = merged.apply(is_fbs, axis=1)
    merged_fbs = merged[mask_fbs].copy()

    print(f"Training dataset shape (FBS-only, before rating_target): {merged_fbs.shape}")

    # 6) Build custom rating_target that nudges toward W-L + margin
    # ------------------------------------------------------------------
    # Requirements: these columns should already exist from earlier scripts
    required_cols = ["rating_linear", "games_played", "win_pct_cum", "margin_cum"]
    missing = [c for c in required_cols if c not in merged_fbs.columns]
    if missing:
        raise ValueError(f"Missing required columns for rating_target: {missing}")

    merged_fbs["games_played_safe"] = merged_fbs["games_played"].clip(lower=1)

    # Hyperparameters for how much W-L and margin matter
    alpha = 1.0  # weight for win percentage
    beta = 0.4   # weight for average scoring margin

    # Fill missing resume fields for early weeks:
    # - win_pct_cum: treat as 0.5 (neutral) when unknown
    # - margin_cum: treat as 0 when unknown
    win_pct = merged_fbs["win_pct_cum"].fillna(0.5)
    margin_cum = merged_fbs["margin_cum"].fillna(0.0)

    merged_fbs["rating_target"] = (
        merged_fbs["rating_linear"]
        + alpha * (win_pct - 0.5) * 10.0
        + beta * (margin_cum / merged_fbs["games_played_safe"])
    )

    # Optional: quick sanity check printout
    print("\nSample with rating_target:")
    print(
        merged_fbs[
            [
                "season",
                "week",
                "team",
                "rating_linear",
                "win_pct_cum",
                "margin_cum",
                "rating_target",
            ]
        ].head(20)
    )

    # 7) Save training dataset
    out_path = "data/training_team_week_2019_2024.csv"
    merged_fbs.to_csv(out_path, index=False)
    print(f"\nSaved training dataset (with rating_target) to {out_path}")


if __name__ == "__main__":
    main()
