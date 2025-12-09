import os
import cfbd
import pandas as pd


api_key = os.getenv("CFBD_API_KEY")
if api_key is None:
    raise ValueError("CFBD_API_KEY environment variable not found!")

configuration = cfbd.Configuration()
configuration.access_token = api_key

api_client = cfbd.ApiClient(configuration)


stats_api = cfbd.StatsApi(api_client)


advanced_stats = stats_api.get_advanced_game_stats(
    year=2024,
    week=1,
    season_type="regular",
)

print("Number of advanced stat records:", len(advanced_stats))

if advanced_stats:
    first = advanced_stats[0]
    print("\nFirst record raw:")
    print(first)

rows = []

for rec in advanced_stats:
    # Basic identifiers
    season = rec.season
    week = rec.week
    team = rec.team
    opponent = rec.opponent

    off = rec.offense
    dfns = rec.defense

    # Safely handle possible None values with getattr
    rows.append({
        "season": season,
        "week": week,
        "team": team,
        "opponent": opponent,
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

adv_df = pd.DataFrame(rows)

print("\nAdvanced stats DataFrame shape:", adv_df.shape)
print(adv_df.head())
print("\nColumns:", list(adv_df.columns))
