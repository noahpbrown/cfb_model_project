import pandas as pd
import numpy as np
import os


def compute_ratings_through_week(games_df: pd.DataFrame, max_week: int) -> pd.DataFrame:
    """
    Compute least-squares ratings using all games up to and including max_week.
    Returns a DataFrame: ['team', 'rating_linear'].
    """
    df = games_df.copy()
    df = df[df["week"] <= max_week]
    df = df.dropna(subset=["home_points", "away_points"])

    if df.empty:
        return pd.DataFrame(columns=["team", "rating_linear"])

    df["home_points"] = df["home_points"].astype(int)
    df["away_points"] = df["away_points"].astype(int)
    df["margin"] = df["home_points"] - df["away_points"]

    teams = pd.unique(pd.concat([df["home_team"], df["away_team"]], ignore_index=True))
    teams = np.sort(teams)

    team_to_idx = {team: i for i, team in enumerate(teams)}
    idx_to_team = {i: team for team, i in team_to_idx.items()}

    num_teams = len(teams)
    num_games = len(df)
    num_params = num_teams + 1  # team ratings + HFA

    A = np.zeros((num_games, num_params), dtype=float)
    y = df["margin"].to_numpy(dtype=float)

    for row_idx, (_, game) in enumerate(df.iterrows()):
        home = game["home_team"]
        away = game["away_team"]

        i_home = team_to_idx[home]
        i_away = team_to_idx[away]

        A[row_idx, i_home] = 1.0
        A[row_idx, i_away] = -1.0

        is_neutral = bool(game["neutral_site"])
        A[row_idx, num_teams] = 0.0 if is_neutral else 1.0

    solution, residuals, rank, s = np.linalg.lstsq(A, y, rcond=None)
    ratings_raw = solution[:num_teams]

    ratings = ratings_raw - ratings_raw.mean()

    ratings_df = pd.DataFrame({
        "team": [idx_to_team[i] for i in range(num_teams)],
        "rating_linear": ratings,
    })

    return ratings_df


def build_ratings_for_season(games: pd.DataFrame, features: pd.DataFrame, year: int) -> pd.DataFrame:
    """
    For a given season, compute weekly ratings for all weeks present in features.
    Returns DataFrame: ['season', 'week', 'team', 'rating_linear'].
    """
    weeks = sorted(features["week"].unique())
    rating_rows = []

    for w in weeks:
        print(f"[{year}] Computing ratings through week {w}...")
        ratings_df = compute_ratings_through_week(games, w)
        ratings_df["season"] = year
        ratings_df["week"] = w
        rating_rows.append(ratings_df)

    all_ratings = pd.concat(rating_rows, ignore_index=True)
    all_ratings = all_ratings[["season", "week", "team", "rating_linear"]]
    return all_ratings


def main():
    seasons = [2019, 2021, 2022, 2023, 2024, 2025]

    os.makedirs("data", exist_ok=True)

    all_ratings = []

    for year in seasons:
        print(f"\n=== Building ratings for season {year} ===")
        games_path = f"data/games_{year}.csv"
        features_path = f"data/team_week_features_{year}.csv"

        games = pd.read_csv(games_path)
        features = pd.read_csv(features_path)

        ratings = build_ratings_for_season(games, features, year)
        print(f"[{year}] ratings shape: {ratings.shape}")
        print(ratings.head())

        out_path = f"data/ratings_{year}.csv"
        ratings.to_csv(out_path, index=False)
        print(f"[{year}] Saved to {out_path}")

        all_ratings.append(ratings)

    combined = pd.concat(all_ratings, ignore_index=True)
    combined_path = "data/ratings_2019_2024.csv"
    combined.to_csv(combined_path, index=False)
    print(f"\nCombined ratings shape: {combined.shape}")
    print(f"Saved combined ratings to {combined_path}")


if __name__ == "__main__":
    main()
