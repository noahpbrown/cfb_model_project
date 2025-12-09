import os

import cfbd
import pandas as pd


SEASONS = [2019, 2021, 2022, 2023, 2024, 2025]


def get_cfbd_client() -> cfbd.ApiClient:
    """Initialize and return a CFBD ApiClient using the env API key."""
    api_key = os.getenv("CFBD_API_KEY")
    if api_key is None:
        raise ValueError("CFBD_API_KEY environment variable not found!")

    configuration = cfbd.Configuration()
    configuration.access_token = api_key

    return cfbd.ApiClient(configuration)


def load_features_with_sos(path: str = "data/team_week_features_with_sos_2019_2024.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"Loaded features+SoS from {path} with shape {df.shape}")
    return df


def fetch_talent_for_seasons(teams_api: cfbd.TeamsApi) -> pd.DataFrame:
    rows = []
    for year in SEASONS:
        print(f"[{year}] Fetching talent ratings...")
        talent_list = teams_api.get_talent(year=year)
        for t in talent_list:
            school = getattr(t, "school", None)
            if school is None:
                school = getattr(t, "team", None)

            rows.append(
                {
                    "season": year,
                    "team": school,
                    "talent": getattr(t, "talent", None),
                }
            )

    df = pd.DataFrame(rows)
    print(f"Talent priors shape: {df.shape}")
    return df


def fetch_returning_production_for_seasons(players_api: cfbd.PlayersApi) -> pd.DataFrame:
    rows = []
    for year in SEASONS:
        print(f"[{year}] Fetching returning production...")
        rp_list = players_api.get_returning_production(year=year)
        for rp in rp_list:
            team = getattr(rp, "team", None)
            if team is None:
                team = getattr(rp, "school", None)

            rows.append(
                {
                    "season": year,
                    "team": team,
                    "rp_total": getattr(rp, "total", None),
                    "rp_offense": getattr(rp, "offense", None),
                    "rp_defense": getattr(rp, "defense", None),
                }
            )

    df = pd.DataFrame(rows)
    print(f"Returning production priors shape: {df.shape}")
    return df


def main():
    os.makedirs("data", exist_ok=True)

    # Base features (already have SoS and advanced stats)
    features = load_features_with_sos()

    api_client = get_cfbd_client()
    teams_api = cfbd.TeamsApi(api_client)
    players_api = cfbd.PlayersApi(api_client)

    # Fetch priors
    df_talent = fetch_talent_for_seasons(teams_api)
    df_rp = fetch_returning_production_for_seasons(players_api)

    # Merge talent + returning production into a joint priors table
    priors = pd.merge(
        df_talent,
        df_rp,
        on=["season", "team"],
        how="outer",
    )
    print(f"Combined priors (talent + RP) shape: {priors.shape}")
    print(priors.head(10))

    # Merge priors onto features
    merged = features.merge(
        priors,
        on=["season", "team"],
        how="left",
    )

    print(f"Merged features + priors shape: {merged.shape}")
    print(merged.head(10))

    out_path = "data/team_week_features_full_2019_2024.csv"
    merged.to_csv(out_path, index=False)
    print(f"Saved full feature set (with priors) to {out_path}")


if __name__ == "__main__":
    main()
