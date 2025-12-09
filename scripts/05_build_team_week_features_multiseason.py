import os
from collections import defaultdict

import cfbd
import pandas as pd
import numpy as np


def get_cfbd_client() -> cfbd.ApiClient:
    """Initialize and return a CFBD ApiClient using the env API key."""
    api_key = os.getenv("CFBD_API_KEY")
    if api_key is None:
        raise ValueError("CFBD_API_KEY environment variable not found!")

    configuration = cfbd.Configuration()
    configuration.access_token = api_key

    return cfbd.ApiClient(configuration)


def fetch_advanced_stats_for_week(
    stats_api: cfbd.StatsApi, year: int, week: int
) -> pd.DataFrame:
    """
    Fetch advanced game stats for a single (year, week) and return as a DataFrame
    with one row per (team, game). Includes aggregate and pass/rush split metrics.
    """
    advanced_stats = stats_api.get_advanced_game_stats(
        year=year,
        week=week,
        season_type="regular",
    )

    if not advanced_stats:
        return pd.DataFrame()

    rows = []
    for rec in advanced_stats:
        off = rec.offense
        dfns = rec.defense

        # Offense subcomponents
        off_pass = getattr(off, "passing_plays", None) if off is not None else None
        off_rush = getattr(off, "rushing_plays", None) if off is not None else None

        # Defense subcomponents
        def_pass = getattr(dfns, "passing_plays", None) if dfns is not None else None
        def_rush = getattr(dfns, "rushing_plays", None) if dfns is not None else None

        rows.append(
            {
                "season": rec.season,
                "week": rec.week,
                "team": rec.team,
                "opponent": rec.opponent,
                # Offensive aggregate metrics
                "off_ppa": getattr(off, "ppa", None) if off is not None else None,
                "off_success_rate": getattr(off, "success_rate", None)
                if off is not None
                else None,
                "off_explosiveness": getattr(off, "explosiveness", None)
                if off is not None
                else None,
                "off_plays": getattr(off, "plays", None) if off is not None else None,
                # Defensive aggregate metrics
                "def_ppa": getattr(dfns, "ppa", None)
                if dfns is not None
                else None,
                "def_success_rate": getattr(dfns, "success_rate", None)
                if dfns is not None
                else None,
                "def_explosiveness": getattr(dfns, "explosiveness", None)
                if dfns is not None
                else None,
                "def_plays": getattr(dfns, "plays", None)
                if dfns is not None
                else None,
                # --- NEW granular offense metrics ---
                "off_pass_ppa": getattr(off_pass, "ppa", None)
                if off_pass is not None
                else None,
                "off_pass_success_rate": getattr(off_pass, "success_rate", None)
                if off_pass is not None
                else None,
                "off_rush_ppa": getattr(off_rush, "ppa", None)
                if off_rush is not None
                else None,
                "off_rush_success_rate": getattr(off_rush, "success_rate", None)
                if off_rush is not None
                else None,
                # --- NEW granular defense metrics ---
                "def_pass_ppa": getattr(def_pass, "ppa", None)
                if def_pass is not None
                else None,
                "def_pass_success_rate": getattr(def_pass, "success_rate", None)
                if def_pass is not None
                else None,
                "def_rush_ppa": getattr(def_rush, "ppa", None)
                if def_rush is not None
                else None,
                "def_rush_success_rate": getattr(def_rush, "success_rate", None)
                if def_rush is not None
                else None,
            }
        )

    return pd.DataFrame(rows)


def build_team_week_features_for_season(
    stats_api: cfbd.StatsApi, year: int
) -> pd.DataFrame:
    """
    Build cumulative team-week advanced stats for a single season (year).

    Returns a DataFrame with one row per (season, week, team) including:
    - games_played
    - cumulative off/def EPA, success rate, explosiveness (play-weighted)
    - per-game cumulative pass/rush EPA & success rate for offense/defense
    """
    # Running state for each team for this season only
    team_state = defaultdict(
        lambda: {
            # offensive sums (play-weighted overall)
            "off_ppa_sum": 0.0,
            "off_success_sum": 0.0,
            "off_explosive_sum": 0.0,
            "off_plays": 0.0,
            # defensive sums (play-weighted overall)
            "def_ppa_sum": 0.0,
            "def_success_sum": 0.0,
            "def_explosive_sum": 0.0,
            "def_plays": 0.0,
            # NEW: offense pass/rush per-game sums & counts
            "off_pass_ppa_sum": 0.0,
            "off_pass_ppa_count": 0,
            "off_pass_sr_sum": 0.0,
            "off_pass_sr_count": 0,
            "off_rush_ppa_sum": 0.0,
            "off_rush_ppa_count": 0,
            "off_rush_sr_sum": 0.0,
            "off_rush_sr_count": 0,
            # NEW: defense pass/rush per-game sums & counts
            "def_pass_ppa_sum": 0.0,
            "def_pass_ppa_count": 0,
            "def_pass_sr_sum": 0.0,
            "def_pass_sr_count": 0,
            "def_rush_ppa_sum": 0.0,
            "def_rush_ppa_count": 0,
            "def_rush_sr_sum": 0.0,
            "def_rush_sr_count": 0,
            # meta
            "games_played": 0,
        }
    )

    feature_rows = []

    # Conservative week range; we'll break when we hit an empty week.
    for week in range(1, 16):
        print(f"[{year}] Fetching advanced stats for week {week}...")
        week_df = fetch_advanced_stats_for_week(stats_api, year, week)

        if week_df.empty:
            print(
                f"[{year}] No advanced stats found for week {week}, stopping for this season."
            )
            break

        # Update per-team cumulative state from this week's games
        for _, row in week_df.iterrows():
            team = row["team"]

            off_ppa = row["off_ppa"]
            off_sr = row["off_success_rate"]
            off_expl = row["off_explosiveness"]
            off_plays = row["off_plays"]

            def_ppa = row["def_ppa"]
            def_sr = row["def_success_rate"]
            def_expl = row["def_explosiveness"]
            def_plays = row["def_plays"]

            # NEW granular metrics from the row
            off_pass_ppa = row.get("off_pass_ppa", np.nan)
            off_pass_sr = row.get("off_pass_success_rate", np.nan)
            off_rush_ppa = row.get("off_rush_ppa", np.nan)
            off_rush_sr = row.get("off_rush_success_rate", np.nan)

            def_pass_ppa = row.get("def_pass_ppa", np.nan)
            def_pass_sr = row.get("def_pass_success_rate", np.nan)
            def_rush_ppa = row.get("def_rush_ppa", np.nan)
            def_rush_sr = row.get("def_rush_success_rate", np.nan)

            # Handle NaNs by treating them as zeros for the sums (for aggregate stats)
            if pd.isna(off_ppa):
                off_ppa = 0.0
            if pd.isna(off_sr):
                off_sr = 0.0
            if pd.isna(off_expl):
                off_expl = 0.0
            if pd.isna(off_plays):
                off_plays = 0.0

            if pd.isna(def_ppa):
                def_ppa = 0.0
            if pd.isna(def_sr):
                def_sr = 0.0
            if pd.isna(def_expl):
                def_expl = 0.0
            if pd.isna(def_plays):
                def_plays = 0.0

            state = team_state[team]

            # Offensive weighted sums (value * plays)
            state["off_ppa_sum"] += off_ppa * off_plays
            state["off_success_sum"] += off_sr * off_plays
            state["off_explosive_sum"] += off_expl * off_plays
            state["off_plays"] += off_plays

            # Defensive weighted sums
            state["def_ppa_sum"] += def_ppa * def_plays
            state["def_success_sum"] += def_sr * def_plays
            state["def_explosive_sum"] += def_expl * def_plays
            state["def_plays"] += def_plays

            # Count this game
            state["games_played"] += 1

            # --- New: per-game offense pass/rush stats ---
            if not pd.isna(off_pass_ppa):
                state["off_pass_ppa_sum"] += off_pass_ppa
                state["off_pass_ppa_count"] += 1
            if not pd.isna(off_pass_sr):
                state["off_pass_sr_sum"] += off_pass_sr
                state["off_pass_sr_count"] += 1
            if not pd.isna(off_rush_ppa):
                state["off_rush_ppa_sum"] += off_rush_ppa
                state["off_rush_ppa_count"] += 1
            if not pd.isna(off_rush_sr):
                state["off_rush_sr_sum"] += off_rush_sr
                state["off_rush_sr_count"] += 1

            # --- New: per-game defense pass/rush stats ---
            if not pd.isna(def_pass_ppa):
                state["def_pass_ppa_sum"] += def_pass_ppa
                state["def_pass_ppa_count"] += 1
            if not pd.isna(def_pass_sr):
                state["def_pass_sr_sum"] += def_pass_sr
                state["def_pass_sr_count"] += 1
            if not pd.isna(def_rush_ppa):
                state["def_rush_ppa_sum"] += def_rush_ppa
                state["def_rush_ppa_count"] += 1
            if not pd.isna(def_rush_sr):
                state["def_rush_sr_sum"] += def_rush_sr
                state["def_rush_sr_count"] += 1

        # Helper for safe division
        def safe_div(sum_val, count_val):
            return sum_val / count_val if count_val > 0 else np.nan

        # After updating state for this week, record cumulative metrics for all teams
        for team, state in team_state.items():
            if state["games_played"] == 0:
                continue

            off_plays = state["off_plays"]
            def_plays = state["def_plays"]

            # Avoid divide by zero; if plays are 0, we'll mark as NaN
            if off_plays > 0:
                off_ppa_cum = state["off_ppa_sum"] / off_plays
                off_sr_cum = state["off_success_sum"] / off_plays
                off_expl_cum = state["off_explosive_sum"] / off_plays
            else:
                off_ppa_cum = np.nan
                off_sr_cum = np.nan
                off_expl_cum = np.nan

            if def_plays > 0:
                def_ppa_cum = state["def_ppa_sum"] / def_plays
                def_sr_cum = state["def_success_sum"] / def_plays
                def_expl_cum = state["def_explosive_sum"] / def_plays
            else:
                def_ppa_cum = np.nan
                def_sr_cum = np.nan
                def_expl_cum = np.nan

            # NEW: per-game cumulative averages for pass/rush splits
            off_pass_ppa_cum = safe_div(
                state["off_pass_ppa_sum"], state["off_pass_ppa_count"]
            )
            off_pass_sr_cum = safe_div(
                state["off_pass_sr_sum"], state["off_pass_sr_count"]
            )
            off_rush_ppa_cum = safe_div(
                state["off_rush_ppa_sum"], state["off_rush_ppa_count"]
            )
            off_rush_sr_cum = safe_div(
                state["off_rush_sr_sum"], state["off_rush_sr_count"]
            )

            def_pass_ppa_cum = safe_div(
                state["def_pass_ppa_sum"], state["def_pass_ppa_count"]
            )
            def_pass_sr_cum = safe_div(
                state["def_pass_sr_sum"], state["def_pass_sr_count"]
            )
            def_rush_ppa_cum = safe_div(
                state["def_rush_ppa_sum"], state["def_rush_ppa_count"]
            )
            def_rush_sr_cum = safe_div(
                state["def_rush_sr_sum"], state["def_rush_sr_count"]
            )

            feature_rows.append(
                {
                    "season": year,
                    "week": week,
                    "team": team,
                    "games_played": state["games_played"],
                    # Overall offense/defense
                    "off_ppa_cum": off_ppa_cum,
                    "off_success_cum": off_sr_cum,
                    "off_explosive_cum": off_expl_cum,
                    "def_ppa_cum": def_ppa_cum,
                    "def_success_cum": def_sr_cum,
                    "def_explosive_cum": def_expl_cum,
                    # NEW offense pass/rush splits
                    "off_pass_ppa_cum": off_pass_ppa_cum,
                    "off_pass_success_cum": off_pass_sr_cum,
                    "off_rush_ppa_cum": off_rush_ppa_cum,
                    "off_rush_success_cum": off_rush_sr_cum,
                    # NEW defense pass/rush splits
                    "def_pass_ppa_cum": def_pass_ppa_cum,
                    "def_pass_success_cum": def_pass_sr_cum,
                    "def_rush_ppa_cum": def_rush_ppa_cum,
                    "def_rush_success_cum": def_rush_sr_cum,
                }
            )

    features_df = pd.DataFrame(feature_rows)
    return features_df


def main():
    # Seasons to include in the model.
    # We intentionally skip 2020 (weird COVID season).
    seasons = [2019, 2021, 2022, 2023, 2024, 2025]

    api_client = get_cfbd_client()
    stats_api = cfbd.StatsApi(api_client)

    os.makedirs("data", exist_ok=True)

    all_seasons = []

    for year in seasons:
        print(f"\n=== Building team-week features for season {year} ===")
        features_df = build_team_week_features_for_season(stats_api, year)
        print(f"[{year}] team-week feature shape: {features_df.shape}")
        print(features_df.head(10))

        # Save per-season file
        out_path = f"data/team_week_features_{year}.csv"
        features_df.to_csv(out_path, index=False)
        print(f"[{year}] Saved to {out_path}")

        all_seasons.append(features_df)

    # Combine all seasons into one big DataFrame
    combined = pd.concat(all_seasons, ignore_index=True)
    combined_path = "data/team_week_features_2019_2024.csv"
    combined.to_csv(combined_path, index=False)
    print(f"\nCombined team-week feature shape: {combined.shape}")
    print(f"Saved combined features to {combined_path}")


if __name__ == "__main__":
    main()
