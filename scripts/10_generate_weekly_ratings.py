#!/usr/bin/env python

import argparse
import os

import numpy as np
import pandas as pd
import cfbd
from xgboost import XGBRegressor


FEATURE_COLS = [
    # Meta / usage
    "games_played",

    # Overall offense / defense
    "off_ppa_cum",
    "off_success_cum",
    "off_explosive_cum",
    "def_ppa_cum",
    "def_success_cum",
    "def_explosive_cum",

    # Offense pass/rush splits
    "off_pass_ppa_cum",
    "off_pass_success_cum",
    "off_rush_ppa_cum",
    "off_rush_success_cum",

    # Defense pass/rush splits
    "def_pass_ppa_cum",
    "def_pass_success_cum",
    "def_rush_ppa_cum",
    "def_rush_success_cum",

    # Strength of schedule
    "sos_opp_rating_mean_cum",
    "sos_opp_rating_min_cum",
    "sos_opp_rating_max_cum",
    "sos_games_cum",

    # Preseason priors
    "talent",
    "rp_total",
    "rp_offense",
    "rp_defense",

    # Opponent-adjusted lagged EPA
    "off_ppa_adj_lagged_cum",
    "def_ppa_adj_lagged_cum",

    # Resume / margin features
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


def get_cfbd_client() -> cfbd.ApiClient:
    api_key = os.getenv("CFBD_API_KEY")
    if api_key is None:
        raise ValueError("CFBD_API_KEY environment variable not found!")

    configuration = cfbd.Configuration()
    configuration.access_token = api_key
    return cfbd.ApiClient(configuration)


def load_model(path: str = "models/xgb_rating_model.json") -> XGBRegressor:
    model = XGBRegressor()
    model.load_model(path)
    return model


def load_features(path: str = "data/team_week_features_full_2019_2024.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"Loaded full team-week features from {path} with shape {df.shape}")
    return df


def load_logos(path: str = "data/team_logos_espn.csv") -> pd.DataFrame:
    """
    Load ESPN logo CSV and return a small DF with:
      team, team_id, abbreviation, primary_color, primary_logo_url
    """
    logos = pd.read_csv(path)

    # First URL in the Logos column
    logos["primary_logo_url"] = logos["Logos"].astype(str).str.split(",").str[0]

    # Normalize column names to match features DF
    logos = logos.rename(
        columns={
            "School": "team",
            "Id": "team_id",
            "Abbreviation": "abbr",
            "Color": "primary_color",
        }
    )

    keep_cols = ["team", "team_id", "abbr", "primary_color", "primary_logo_url"]
    logos = logos[keep_cols]

    print(f"Loaded logos from {path} with shape {logos.shape}")
    return logos


def get_fbs_teams(api_client: cfbd.ApiClient, season: int) -> set[str]:
    teams_api = cfbd.TeamsApi(api_client)
    fbs_teams = teams_api.get_fbs_teams(year=season)
    names = {t.school for t in fbs_teams}
    print(f"[{season}] FBS teams: {len(names)}")
    return names


def build_top25(
    model: XGBRegressor,
    features_df: pd.DataFrame,
    logos_df: pd.DataFrame,
    season: int,
    week: int,
    api_client: cfbd.ApiClient,
) -> pd.DataFrame:
    # Filter to this season/week
    week_df = features_df[(features_df["season"] == season) & (features_df["week"] == week)].copy()
    print(f"Rows for season={season}, week={week}: {week_df.shape[0]}")

    # FBS filter
    fbs_names = get_fbs_teams(api_client, season)
    week_df = week_df[week_df["team"].isin(fbs_names)].copy()
    print(f"After FBS filter: {week_df.shape[0]} rows")

    # Build feature matrix
    missing = [c for c in FEATURE_COLS if c not in week_df.columns]
    if missing:
        raise ValueError(f"Week DF missing expected feature columns: {missing}")

    X = week_df[FEATURE_COLS].to_numpy(dtype=float)

    # Predict ratings
    week_df["rating_pred"] = model.predict(X)

    # Merge logos
    week_df = week_df.merge(logos_df, on="team", how="left")

    # Sort & take top 25
    week_df = week_df.sort_values("rating_pred", ascending=False).reset_index(drop=True)
    top25 = week_df.head(25).copy()

    # ---------- Spread vs #1 with compressed #1 gap ----------
    if len(top25) > 1:
        # Raw ratings
        r1 = float(top25.iloc[0]["rating_pred"])

        # Use the "elite pack" (next up to 4 teams) as baseline
        k = min(4, len(top25) - 1)
        elite_ratings = top25.iloc[1 : 1 + k]["rating_pred"].astype(float)
        elite_mean = float(elite_ratings.mean())

        gap_current = r1 - elite_mean

        # Desired #1 vs elite-pack edge in "points"
        desired_gap_points = 5.0  # tweak this if you want #1 closer/further

        if gap_current > 1e-6:
            scale_top = desired_gap_points / gap_current
        else:
            scale_top = 1.0

        # Compressed #1 rating for spread computation only
        effective_r1 = elite_mean + (r1 - elite_mean) * scale_top

        # Convert rating gap to "spread vs #1"
        SPREAD_SCALE = 1.0  # global scale if you ever want to map ratings->points differently
        diffs = effective_r1 - top25["rating_pred"].astype(float)
        top25["spread_vs_1"] = diffs * SPREAD_SCALE

        # Ensure #1 shows 0.0
        top25.loc[top25.index[0], "spread_vs_1"] = 0.0
    else:
        top25["spread_vs_1"] = 0.0

    return top25


def print_top25_table(top25: pd.DataFrame):
    print("\nTop 25 teams by predicted rating:")
    print(f"{'rank':>4} {'team':25s} {'games':>5} {'rating':>8} {'spread_vs_1':>12}")
    for idx, row in top25.iterrows():
        rank = idx + 1
        team = row["team"]
        games = int(row["games_played"])
        rating = float(row["rating_pred"])
        if rank == 1:
            spread_str = "--"
        else:
            diff = float(row["spread_vs_1"])
            spread_str = f"+{diff:.1f}"
        print(f"{rank:4d} {team:25s} {games:5d} {rating:8.2f} {spread_str:>12s}")


def save_top25_payload(top25: pd.DataFrame, season: int, week: int):
    """
    Save a JSON payload suitable for a front-end to render a JP-poll style graphic.
    """
    payload_cols = [
        "rank",
        "team",
        "abbr",
        "wins_cum",
        "losses_cum",
        "team_id",
        "games_played",
        "rating_pred",
        "spread_vs_1",
        "primary_color",
        "primary_logo_url",
    ]

    payload = top25.copy()
    payload["rank"] = np.arange(1, len(payload) + 1)

    # Make sure these columns exist (some may be NaN but that's fine)
    for col in payload_cols:
        if col not in payload.columns:
            payload[col] = None

    out_dir = "outputs"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"top25_{season}_week{week}.json")
    payload[payload_cols].to_json(out_path, orient="records", indent=2)
    print(f"\nSaved top 25 payload to {out_path}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--week", type=int, required=True)
    parser.add_argument(
        "--logos-path",
        type=str,
        default="data/team_logos.csv",
        help="CSV with columns Id, School, Abbreviation, Color, Logos",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    season = args.season
    week = args.week

    print(f"\n\nGenerating ratings for season={season}, week={week}...")

    model = load_model("models/xgb_rating_model.json")
    features_df = load_features("data/team_week_features_full_2019_2024.csv")
    logos_df = load_logos(args.logos_path)
    api_client = get_cfbd_client()

    top25 = build_top25(model, features_df, logos_df, season, week, api_client)
    print_top25_table(top25)
    save_top25_payload(top25, season, week)


if __name__ == "__main__":
    main()
