import pandas as pd
import numpy as np
import cfbd
from cfbd import TeamsApi, ApiClient, Configuration
import os


df = pd.read_csv("data/games_2024.csv")

df = df.dropna(subset=["home_points", "away_points"])
df["home_points"] = df["home_points"].astype(int)
df["away_points"] = df["away_points"].astype(int)
df["margin"] = df["home_points"] - df["away_points"]

print("\nSample with margin:")
print(df[["season", "week", "home_team", "away_team", "home_points", "away_points", "margin"]].head())


teams = pd.unique(pd.concat([df["home_team"], df["away_team"]], ignore_index=True))
teams = np.sort(teams)  # just to have consistent order

team_to_idx = {team: i for i, team in enumerate(teams)}
idx_to_team = {i: team for team, i in team_to_idx.items()}

num_teams = len(teams)
print("\nNumber of teams:", num_teams)


num_games = len(df)
num_params = num_teams + 1  # all team ratings + 1 HFA

A = np.zeros((num_games, num_params), dtype=float)
y = df["margin"].to_numpy(dtype=float)

for row_idx, (_, game) in enumerate(df.iterrows()):
    home = game["home_team"]
    away = game["away_team"]

    i_home = team_to_idx[home]
    i_away = team_to_idx[away]

    # Rating coefficients
    A[row_idx, i_home] = 1.0
    A[row_idx, i_away] = -1.0

    # HFA term (last column): 1 if not neutral, 0 if neutral_site=True
    is_neutral = bool(game["neutral_site"])
    A[row_idx, num_teams] = 0.0 if is_neutral else 1.0

# 5. Solve least squares
solution, residuals, rank, s = np.linalg.lstsq(A, y, rcond=None)

ratings_raw = solution[:num_teams]
hfa = solution[num_teams]

print("\nEstimated home-field advantage (points):", hfa)

# 6. Normalize ratings to mean 0
ratings = ratings_raw - ratings_raw.mean()

ratings_df = pd.DataFrame({
    "team": [idx_to_team[i] for i in range(num_teams)],
    "rating": ratings
})

ratings_df = ratings_df.sort_values("rating", ascending=False).reset_index(drop=True)

print("\nTop 25 teams by rating (all divisions mixed):")
print(ratings_df.head(25))


# ---- FBS filtering ----

# Load API key the same way you did in fetch script
api_key = os.getenv("CFBD_API_KEY")

# Configure CFBD client
configuration = cfbd.Configuration()
configuration.access_token = api_key

api_client = cfbd.ApiClient(configuration)
teams_api = cfbd.TeamsApi(api_client)

# Get list of 2024 FBS teams
fbs_teams = teams_api.get_fbs_teams(year=2024)
fbs_set = {t.school for t in fbs_teams}

print("\nNumber of FBS teams:", len(fbs_set))

# Filter ratings to only FBS
ratings_fbs = ratings_df[ratings_df["team"].isin(fbs_set)]
ratings_fbs = ratings_fbs.sort_values("rating", ascending=False).reset_index(drop=True)

print("\nTop 25 FBS teams:")
print(ratings_fbs.head(25))
