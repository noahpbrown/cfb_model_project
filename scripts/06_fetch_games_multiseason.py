import os
import cfbd
import pandas as pd


def get_cfbd_client() -> cfbd.ApiClient:
    api_key = os.getenv("CFBD_API_KEY")
    if api_key is None:
        raise ValueError("CFBD_API_KEY environment variable not found!")

    configuration = cfbd.Configuration()
    configuration.access_token = api_key

    return cfbd.ApiClient(configuration)


def fetch_games_for_season(games_api: cfbd.GamesApi, year: int) -> pd.DataFrame:
    """Fetch all regular-season games for a given year and return as DataFrame."""
    games = games_api.get_games(
        year=year,
        season_type="regular",
    )

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
            "away_conference": g.away_conference,
        })

    df = pd.DataFrame(rows)
    return df


def main():
    seasons = [2019, 2021, 2022, 2023, 2024, 2025]

    api_client = get_cfbd_client()
    games_api = cfbd.GamesApi(api_client)

    os.makedirs("data", exist_ok=True)

    for year in seasons:
        print(f"\n=== Fetching games for season {year} ===")
        df = fetch_games_for_season(games_api, year)
        print(f"[{year}] games shape: {df.shape}")
        print(df.head())

        out_path = f"data/games_{year}.csv"
        df.to_csv(out_path, index=False)
        print(f"[{year}] Saved to {out_path}")


if __name__ == "__main__":
    main()
