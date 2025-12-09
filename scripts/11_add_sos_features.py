import os
from typing import List

import numpy as np
import pandas as pd


SEASONS = [2019, 2021, 2022, 2023, 2024, 2025]


def load_team_week_features(path: str = "data/team_week_features_2019_2024.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"Loaded base team-week features from {path} with shape {df.shape}")
    return df


def load_games_multiseason(seasons: List[int]) -> pd.DataFrame:
    frames = []
    for year in seasons:
        path = f"data/games_{year}.csv"
        if not os.path.exists(path):
            raise FileNotFoundError(f"Expected games file not found: {path}")
        df_year = pd.read_csv(path)
        frames.append(df_year)
        print(f"[{year}] games shape: {df_year.shape}")

    games_all = pd.concat(frames, ignore_index=True)
    print(f"Combined games shape: {games_all.shape}")
    return games_all


def load_ratings(path: str = "data/ratings_2019_2024.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"Loaded ratings from {path} with shape {df.shape}")
    return df


def build_team_opponent_table(games_all: pd.DataFrame) -> pd.DataFrame:
    """
    From games DataFrame, build a table with one row per (season, week, team, opponent).
    """
    rows = []
    for _, g in games_all.iterrows():
        season = int(g["season"])
        week = int(g["week"])
        home = g["home_team"]
        away = g["away_team"]

        # Row for home team vs away
        rows.append(
            {
                "season": season,
                "week": week,
                "team": home,
                "opponent": away,
            }
        )
        # Row for away team vs home
        rows.append(
            {
                "season": season,
                "week": week,
                "team": away,
                "opponent": home,
            }
        )

    df = pd.DataFrame(rows)
    print(f"Team-opponent table shape: {df.shape}")
    return df


def attach_opponent_ratings(team_opp_df: pd.DataFrame, ratings_df: pd.DataFrame) -> pd.DataFrame:
    """
    For each (season, week, team, opponent), attach the opponent's rating_linear
    from the ratings DataFrame.
    """
    ratings_opp = ratings_df[["season", "week", "team", "rating_linear"]].rename(
        columns={"team": "opponent", "rating_linear": "opp_rating"}
    )

    merged = team_opp_df.merge(
        ratings_opp,
        on=["season", "week", "opponent"],
        how="left",
    )
    print(f"After merging opponent ratings, shape: {merged.shape}")
    return merged


def build_cumulative_sos(team_opp_with_rating: pd.DataFrame) -> pd.DataFrame:
    """
    Build cumulative strength-of-schedule features for each (season, week, team).

    For each season & team:
      - iterate weeks in order
      - keep a running list of opponent ratings up to that week
      - record cumulative mean, min, max, and count
    """
    sos_rows = []

    # Ensure we only use rows with some opponent rating present
    df = team_opp_with_rating.copy()
    # It's okay if some opp_rating are NaN; they just won't contribute to cumulative stats

    for season in sorted(df["season"].unique()):
        df_season = df[df["season"] == season]

        for team in sorted(df_season["team"].unique()):
            df_team = df_season[df_season["team"] == team].copy()
            if df_team.empty:
                continue

            # Sort by week
            df_team = df_team.sort_values("week")

            cumulative_ratings = []

            for week in sorted(df_team["week"].unique()):
                week_mask = df_team["week"] == week
                week_ratings = df_team.loc[week_mask, "opp_rating"].dropna().tolist()

                # Add this week's opponent ratings to the cumulative list
                cumulative_ratings.extend(week_ratings)

                if len(cumulative_ratings) == 0:
                    # No valid opponent ratings yet; record NaNs
                    sos_rows.append(
                        {
                            "season": season,
                            "week": week,
                            "team": team,
                            "sos_opp_rating_mean_cum": np.nan,
                            "sos_opp_rating_min_cum": np.nan,
                            "sos_opp_rating_max_cum": np.nan,
                            "sos_games_cum": 0,
                        }
                    )
                else:
                    arr = np.array(cumulative_ratings, dtype=float)
                    sos_rows.append(
                        {
                            "season": season,
                            "week": week,
                            "team": team,
                            "sos_opp_rating_mean_cum": float(np.mean(arr)),
                            "sos_opp_rating_min_cum": float(np.min(arr)),
                            "sos_opp_rating_max_cum": float(np.max(arr)),
                            "sos_games_cum": int(len(arr)),
                        }
                    )

    sos_df = pd.DataFrame(sos_rows)
    print(f"SoS cumulative features shape: {sos_df.shape}")
    return sos_df


def main():
    os.makedirs("data", exist_ok=True)

    # 1) Load base features, games, and ratings
    base_features = load_team_week_features("data/team_week_features_2019_2024.csv")
    games_all = load_games_multiseason(SEASONS)
    ratings_all = load_ratings("data/ratings_2019_2024.csv")

    # 2) Build team-opponent-week table
    team_opp = build_team_opponent_table(games_all)

    # 3) Attach opponent ratings (from the linear system)
    team_opp_with_rating = attach_opponent_ratings(team_opp, ratings_all)

    # 4) Build cumulative SoS features for each (season, week, team)
    sos_df = build_cumulative_sos(team_opp_with_rating)

    # 5) Merge SoS back onto base features
    merged = base_features.merge(
        sos_df,
        on=["season", "week", "team"],
        how="left",
    )

    print(f"Merged features + SoS shape: {merged.shape}")
    print(merged.head(10))

    out_path = "data/team_week_features_with_sos_2019_2024.csv"
    merged.to_csv(out_path, index=False)
    print(f"Saved features with SoS to {out_path}")


if __name__ == "__main__":
    main()
