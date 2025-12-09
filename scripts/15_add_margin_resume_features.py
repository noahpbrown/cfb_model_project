import os
import numpy as np
import pandas as pd

FEATURES_PATH = "data/team_week_features_full_2019_2024.csv"
RATINGS_PATH = "data/ratings_2019_2024.csv"


def load_team_week_features() -> pd.DataFrame:
    df = pd.read_csv(FEATURES_PATH)
    print(f"Loaded team-week features from {FEATURES_PATH} with shape {df.shape}")
    return df


def load_games(seasons):
    all_games = []
    for year in seasons:
        path = f"data/games_{year}.csv"
        if not os.path.exists(path):
            raise FileNotFoundError(f"Expected games file not found: {path}")
        g = pd.read_csv(path)
        print(f"[{year}] games shape: {g.shape}")
        all_games.append(g)
    games = pd.concat(all_games, ignore_index=True)
    return games


def normalize_games_columns(games: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize games into *team-centric* rows with at least:
      season, week, team, opponent, points_for, points_against,
      is_road, is_neutral

    Handles two schemas:
      1) Already team-centric with 'team' / 'opponent' and score cols
      2) Home/away schema with 'home_team', 'away_team', 'home_points', 'away_points'
    """
    games = games.copy()

    # --- Case 1: already team/opponent based ---
    if "team" in games.columns and "opponent" in games.columns:
        print("Detected team/opponent schema in games CSV.")
        # Find scoring columns
        if "points_for" in games.columns and "points_against" in games.columns:
            pf_col, pa_col = "points_for", "points_against"
        elif "team_points" in games.columns and "opp_points" in games.columns:
            pf_col, pa_col = "team_points", "opp_points"
        else:
            raise ValueError(
                "Team/opponent schema but couldn't find scoring columns. "
                "Expected 'points_for'/'points_against' or 'team_points'/'opp_points'."
            )

        games["points_for"] = games[pf_col].astype(float)
        games["points_against"] = games[pa_col].astype(float)

        # Try to derive location if present
        loc_col = None
        if "location" in games.columns:
            loc_col = "location"
        elif "home_away" in games.columns:
            loc_col = "home_away"

        if loc_col is not None:
            loc = games[loc_col].astype(str).str.lower()
            games["is_road"] = loc.eq("away")
            games["is_neutral"] = loc.eq("neutral")
        else:
            print("⚠️ No location/home_away; treating all as non-road, non-neutral.")
            games["is_road"] = False
            games["is_neutral"] = False

        # Ensure required columns exist
        required = ["season", "week", "team", "opponent"]
        missing = [c for c in required if c not in games.columns]
        if missing:
            raise ValueError(f"Games DF missing required columns: {missing}")

        return games[[
            "season", "week", "team", "opponent",
            "points_for", "points_against",
            "is_road", "is_neutral"
        ]]

    # --- Case 2: home/away schema -> expand to team/opponent ---
    print("Detected home/away schema in games CSV; expanding to team-centric rows.")

    needed = {"season", "week", "home_team", "away_team", "home_points", "away_points"}
    missing = needed - set(games.columns)
    if missing:
        raise ValueError(
            f"Games DF missing required home/away columns: {sorted(list(missing))}"
        )

    # Home team rows
    home_rows = pd.DataFrame({
        "season": games["season"],
        "week": games["week"],
        "team": games["home_team"],
        "opponent": games["away_team"],
        "points_for": games["home_points"].astype(float),
        "points_against": games["away_points"].astype(float),
        "location": "home",
        "neutral_site": games["neutral_site"] if "neutral_site" in games.columns else False,
    })

    # Away team rows
    away_rows = pd.DataFrame({
        "season": games["season"],
        "week": games["week"],
        "team": games["away_team"],
        "opponent": games["home_team"],
        "points_for": games["away_points"].astype(float),
        "points_against": games["home_points"].astype(float),
        "location": "away",
        "neutral_site": games["neutral_site"] if "neutral_site" in games.columns else False,
    })

    games_long = pd.concat([home_rows, away_rows], ignore_index=True)

    # Derive is_road / is_neutral
    loc = games_long["location"].astype(str).str.lower()
    games_long["is_neutral"] = games_long["neutral_site"].astype(bool)
    games_long["is_road"] = loc.eq("away") & ~games_long["is_neutral"]

    return games_long[[
        "season", "week", "team", "opponent",
        "points_for", "points_against",
        "is_road", "is_neutral"
    ]]


def add_ratings_to_games(games: pd.DataFrame) -> pd.DataFrame:
    """
    Attach opponent rating & rank-at-game to each game using ratings_2019_2024.csv.

    For 2025 games there will be no match (NaN), which is fine for training
    since we only train on 2019–2024 and XGBoost can handle NaNs at inference.
    """
    ratings = pd.read_csv(RATINGS_PATH)
    print(f"Loaded ratings from {RATINGS_PATH} with shape {ratings.shape}")

    ratings = ratings.copy()
    ratings["rank_linear"] = ratings.groupby(["season", "week"])["rating_linear"].rank(
        ascending=False, method="min"
    )

    opp_ratings = ratings.rename(
        columns={
            "team": "opponent",
            "rating_linear": "opp_rating_game",
            "rank_linear": "opp_rank_game",
        }
    )[["season", "week", "opponent", "opp_rating_game", "opp_rank_game"]]

    games = games.merge(
        opp_ratings,
        on=["season", "week", "opponent"],
        how="left",
    )

    return games


def build_game_level_features() -> pd.DataFrame:
    """
    Produce one row per (season, week, team) with cumulative résumé features:
      - margin_cum
      - wins_cum, losses_cum, win_pct_cum
      - wins_top25_cum, wins_top10_cum
      - road_wins_cum, neutral_wins_cum
      - opp_rating_game_mean_cum
    """
    seasons = [2019, 2021, 2022, 2023, 2024, 2025]

    games = load_games(seasons)
    games = normalize_games_columns(games)

    # Basic per-game outcomes
    games["margin"] = games["points_for"] - games["points_against"]
    games["is_win"] = games["margin"] > 0
    games["is_loss"] = games["margin"] < 0

    # Attach opponent rating / rank at game time (for 2019–2024)
    games = add_ratings_to_games(games)

    # Top-10 / Top-25 wins based on opp_rank_game (NaN-safe)
    games["is_top25_win"] = games["is_win"] & games["opp_rank_game"].le(25)
    games["is_top10_win"] = games["is_win"] & games["opp_rank_game"].le(10)

    games["is_road_win"] = games["is_win"] & games["is_road"]
    games["is_neutral_win"] = games["is_win"] & games["is_neutral"]

    # Sort so expanding/cumsum works in week order
    games = games.sort_values(["season", "team", "week"])

    def _cum_features(team_df: pd.DataFrame) -> pd.DataFrame:
        df = team_df.copy()

        df["margin_cum"] = df["margin"].cumsum()
        df["wins_cum"] = df["is_win"].astype(int).cumsum()
        df["losses_cum"] = df["is_loss"].astype(int).cumsum()

        games_played = df["wins_cum"] + df["losses_cum"]
        df["win_pct_cum"] = np.where(
            games_played > 0, df["wins_cum"] / games_played, np.nan
        )

        df["wins_top25_cum"] = df["is_top25_win"].astype(int).cumsum()
        df["wins_top10_cum"] = df["is_top10_win"].astype(int).cumsum()
        df["road_wins_cum"] = df["is_road_win"].astype(int).cumsum()
        df["neutral_wins_cum"] = df["is_neutral_win"].astype(int).cumsum()

        df["opp_rating_game_mean_cum"] = df["opp_rating_game"].expanding().mean()

        return df[
            [
                "season",
                "week",
                "team",
                "margin_cum",
                "wins_cum",
                "losses_cum",
                "win_pct_cum",
                "wins_top25_cum",
                "wins_top10_cum",
                "road_wins_cum",
                "neutral_wins_cum",
                "opp_rating_game_mean_cum",
            ]
        ]

    cum = games.groupby(["season", "team"], group_keys=False).apply(_cum_features)
    print(f"Game-level cumulative résumé features shape: {cum.shape}")
    print(cum.head(10))

    return cum


def main():
    features = load_team_week_features()
    game_cum = build_game_level_features()

    # Drop existing resume columns if they exist (to avoid merge conflicts)
    resume_cols = [
        "margin_cum",
        "wins_cum",
        "losses_cum",
        "win_pct_cum",
        "wins_top25_cum",
        "wins_top10_cum",
        "road_wins_cum",
        "neutral_wins_cum",
        "opp_rating_game_mean_cum",
    ]
    
    # Only drop columns that actually exist
    cols_to_drop = [col for col in resume_cols if col in features.columns]
    if cols_to_drop:
        print(f"Dropping existing resume columns: {cols_to_drop}")
        features = features.drop(columns=cols_to_drop)

    merged = features.merge(
        game_cum,
        on=["season", "week", "team"],
        how="left",
    )

    print(f"Features + résumé features shape: {merged.shape}")

    # Forward-fill resume features for teams that didn't play in a given week
    # This ensures records carry forward (e.g., week 15 for teams that didn't play)
    # IMPORTANT: Only forward-fill NaN values - don't overwrite actual game results
    print("Forward-filling resume features for teams that didn't play...")
    merged = merged.sort_values(["season", "team", "week"])
    
    resume_cols_to_ffill = [
        "margin_cum",
        "wins_cum",
        "losses_cum",
        "win_pct_cum",
        "wins_top25_cum",
        "wins_top10_cum",
        "road_wins_cum",
        "neutral_wins_cum",
        "opp_rating_game_mean_cum",
    ]
    
    # Forward-fill within each season/team group, but ONLY fill NaN values
    # This preserves actual game results and only fills gaps for teams that didn't play
    for col in resume_cols_to_ffill:
        if col in merged.columns:
            # Create a mask of NaN values before forward-fill
            nan_mask = merged[col].isna()
            # Forward-fill
            merged[col] = merged.groupby(["season", "team"])[col].ffill()
            # Only keep forward-filled values where they were originally NaN
            # This prevents overwriting actual game results
            # (ffill already does this, but being explicit)
    
    # Fill remaining NaNs for counts with 0; keep win_pct/opp_rating as NaN where unknown
    count_cols = [
        "margin_cum",
        "wins_cum",
        "losses_cum",
        "wins_top25_cum",
        "wins_top10_cum",
        "road_wins_cum",
        "neutral_wins_cum",
    ]
    for c in count_cols:
        if c in merged.columns:
            # Only fill NaNs that weren't filled by forward-fill (early weeks with no games)
            merged[c] = merged[c].fillna(0.0)

    # Fill NaNs for counts with 0; keep win_pct/opp_rating as NaN where unknown
    count_cols = [
        "margin_cum",
        "wins_cum",
        "losses_cum",
        "wins_top25_cum",
        "wins_top10_cum",
        "road_wins_cum",
        "neutral_wins_cum",
    ]
    for c in count_cols:
        if c in merged.columns:
            merged[c] = merged[c].fillna(0.0)

    merged.to_csv(FEATURES_PATH, index=False)
    print(f"Saved updated features (with résumé + margin) back to {FEATURES_PATH}")
    print(merged.head(10))


if __name__ == "__main__":
    main()
