import os
from collections import defaultdict

import cfbd
import pandas as pd
import numpy as np

def get_cfbd_client() -> cfbd.ApiClient:
    
    api_key = os.getenv("CFBD_API_KEY")
    if api_key is None:
        raise ValueError("CFBD_API_KEY environment variable not found!")

    configuration = cfbd.Configuration()
    configuration.access_token = api_key

    return cfbd.ApiClient(configuration)

def fetch_advanced_stats_for_week(stats_api: cfbd.StatsApi, year: int, week: int) -> pd.DataFrame:
    """Fetch advanced game stats for a single week and return as a DataFrame."""
    advanced_stats = stats_api.get_advanced_game_stats(
        year=year,
        week=week,
        season_type="regular",
    )

    if not advanced_stats:
        # No games (e.g., beyond current week)
        return pd.DataFrame()

    rows = []
    for rec in advanced_stats:
        off = rec.offense
        dfns = rec.defense

        rows.append({
            "season": rec.season,
            "week": rec.week,
            "team": rec.team,
            "opponent": rec.opponent,
            # Offensive metrics
            "off_ppa": getattr(off, "ppa", None),
            "off_success_rate": getattr(off, "success_rate", None),
            "off_explosiveness": getattr(off, "explosiveness", None),
            "off_plays": getattr(off, "plays", None),
            # Defensive metrics
            "def_ppa": getattr(dfns, "ppa", None),
            "def_success_rate": getattr(dfns, "success_rate", None),
            "def_explosiveness": getattr(dfns, "explosiveness", None),
            "def_plays": getattr(dfns, "plays", None),
        })

    return pd.DataFrame(rows)


def build_team_week_features_2024() -> pd.DataFrame:
    """Build cumulative team-week advanced stats for the 2024 season."""
    year = 2024

    # 1. Set up client & StatsApi
    api_client = get_cfbd_client()
    stats_api = cfbd.StatsApi(api_client)

    # 2. We'll store cumulative state per team in a dict
    team_state = defaultdict(lambda: {
        # offensive sums
        "off_ppa_sum": 0.0,
        "off_success_sum": 0.0,
        "off_explosive_sum": 0.0,
        "off_plays": 0.0,
        # defensive sums
        "def_ppa_sum": 0.0,
        "def_success_sum": 0.0,
        "def_explosive_sum": 0.0,
        "def_plays": 0.0,
        # meta
        "games_played": 0,
    })

    # We'll collect final rows here
    feature_rows = []

    # 3. Iterate over weeks.
    # For now, let's just try weeks 1-15 and stop when we hit an empty week.
    for week in range(1, 16):
        print(f"Fetching advanced stats for {year} week {week}...")
        week_df = fetch_advanced_stats_for_week(stats_api, year, week)

        if week_df.empty:
            print(f"No advanced stats found for week {week}, stopping loop.")
            break

        # 4. For each row (team-game) in this week, update the team_state
        for _, row in week_df.iterrows():
            team = row["team"]

            # --- TODO 1: safely extract values, treating None/NaN as 0 ---
            # Hint: use `val = row["off_ppa"]` then check `pd.isna(val)`.

            off_ppa = row["off_ppa"]
            off_sr = row["off_success_rate"]
            off_expl = row["off_explosiveness"]
            off_plays = row["off_plays"]

            def_ppa = row["def_ppa"]
            def_sr = row["def_success_rate"]
            def_expl = row["def_explosiveness"]
            def_plays = row["def_plays"]

            # Convert NaN to 0 for sums and plays
            # (we can revisit this choice later; for now it keeps things simple)
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

            # --- TODO 2: update team_state sums using these values ---

            state = team_state[team]

            # Offensive weighted sums
            state["off_ppa_sum"] += off_ppa * off_plays
            state["off_success_sum"] += off_sr * off_plays
            state["off_explosive_sum"] += off_expl * off_plays
            state["off_plays"] += off_plays

            # Defensive weighted sums
            state["def_ppa_sum"] += def_ppa * def_plays
            state["def_success_sum"] += def_sr * def_plays
            state["def_explosive_sum"] += def_expl * def_plays
            state["def_plays"] += def_plays

            # Count games_played (we'll just count each game as 1)
            state["games_played"] += 1

        # 5. After updating all teams for this week,
        #    compute the *cumulative* per-team metrics through this week
        for team, state in team_state.items():
            # Only include teams that have played at least one game
            if state["games_played"] == 0:
                continue

            # Avoid divide-by-zero
            off_plays = state["off_plays"] if state["off_plays"] > 0 else np.nan
            def_plays = state["def_plays"] if state["def_plays"] > 0 else np.nan

            off_ppa_cum = state["off_ppa_sum"] / off_plays if off_plays and not np.isnan(off_plays) else np.nan
            off_sr_cum = state["off_success_sum"] / off_plays if off_plays and not np.isnan(off_plays) else np.nan
            off_expl_cum = state["off_explosive_sum"] / off_plays if off_plays and not np.isnan(off_plays) else np.nan

            def_ppa_cum = state["def_ppa_sum"] / def_plays if def_plays and not np.isnan(def_plays) else np.nan
            def_sr_cum = state["def_success_sum"] / def_plays if def_plays and not np.isnan(def_plays) else np.nan
            def_expl_cum = state["def_explosive_sum"] / def_plays if def_plays and not np.isnan(def_plays) else np.nan

            feature_rows.append({
                "season": year,
                "week": week,
                "team": team,
                "games_played": state["games_played"],
                "off_ppa_cum": off_ppa_cum,
                "off_success_cum": off_sr_cum,
                "off_explosive_cum": off_expl_cum,
                "def_ppa_cum": def_ppa_cum,
                "def_success_cum": def_sr_cum,
                "def_explosive_cum": def_expl_cum,
            })

    features_df = pd.DataFrame(feature_rows)
    return features_df

if __name__ == "__main__":
    features_df = build_team_week_features_2024()
    print("Team-week feature DataFrame shape:", features_df.shape)
    print(features_df.head(20))

    # Optionally save to disk
    os.makedirs("data", exist_ok=True)
    features_df.to_csv("data/team_week_features_2024.csv", index=False)
    print("Saved to data/team_week_features_2024.csv")
