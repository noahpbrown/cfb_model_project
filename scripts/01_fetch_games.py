import os
import cfbd
import pandas as pd
api_key = os.getenv("CFBD_API_KEY")

configuration = cfbd.Configuration()
configuration.access_token = api_key

api_client = cfbd.ApiClient(configuration)

print(api_key)
games_api = cfbd.GamesApi(api_client)
games = games_api.get_games(
    year=2024,
    season_type="regular",
    )

import pandas as pd

rows = []
for g in games:
    rows.append({
        "season": g.season,
        "week": g.week,
        "home_team": g.home_team,
        "away_team": g.away_team,
        "home_points": g.home_points,
        "away_points": g.away_points,
        "neutral_site": g.neutral_site,
        "home_conference": g.home_conference,
        "away_conference": g.away_conference
    })

df = pd.DataFrame(rows)

os.makedirs("data", exist_ok=True)
df.to_csv("data/games_2024.csv", index=False)


