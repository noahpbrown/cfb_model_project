#!/usr/bin/env python
"""
Master script to update all data and generate current rankings.

This script:
1. Fetches latest games (for current season)
2. Rebuilds ratings
3. Updates features
4. Adds all feature enhancements
5. Generates rankings for specified week (or auto-detects current week)

Usage:
    python scripts/update_and_generate_rankings.py --season 2025
    python scripts/update_and_generate_rankings.py --season 2025 --week 15
    python scripts/update_and_generate_rankings.py --season 2025 --skip-games
"""

import argparse
import subprocess
import sys
import os
import pandas as pd
from pathlib import Path
import cfbd

# Scripts to run in order (for weekly updates)
WEEKLY_UPDATE_SCRIPTS = [
    ("Fetch latest games", "scripts/06_fetch_games_multiseason.py", False),
    ("Build ratings", "scripts/07_build_ratings_multiseason.py", False),
    ("Build base features", "scripts/05_build_team_week_features_multiseason.py", False),
    ("Add strength of schedule", "scripts/11_add_sos_features.py", False),
    ("Add preseason priors", "scripts/12_add_preseason_priors.py", False),
    ("Add opponent-adjusted stats", "scripts/13_add_opponent_adjusted_stats.py", False),
    ("Add lagged opponent stats", "scripts/14_add_lagged_opponent_stats.py", False),
    ("Add resume features", "scripts/15_add_margin_resume_features.py", False),
    ("Generate rankings", "scripts/10_generate_weekly_ratings.py", True),  # Needs args
]

def get_cfbd_client() -> cfbd.ApiClient:
    """Get CFBD API client."""
    api_key = os.getenv("CFBD_API_KEY")
    if api_key is None:
        raise ValueError("CFBD_API_KEY environment variable not found!")
    configuration = cfbd.Configuration()
    configuration.access_token = api_key
    return cfbd.ApiClient(configuration)

def detect_current_week_from_data(season: int) -> int:
    """Detect current week by checking games data file."""
    games_path = Path(f"data/games_{season}.csv")
    
    if games_path.exists():
        try:
            df = pd.read_csv(games_path)
            if not df.empty and "week" in df.columns:
                max_week = int(df["week"].max())
                print(f"📅 Found latest week in games data: {max_week}")
                return max_week
        except Exception as e:
            print(f"⚠️  Could not read games file: {e}")
    
    return None

def detect_current_week_from_api(season: int) -> int:
    """Detect current week by querying CFBD API."""
    try:
        api_client = get_cfbd_client()
        games_api = cfbd.GamesApi(api_client)
        
        # Fetch games for the season
        games = games_api.get_games(
            year=season,
            season_type="regular",
        )
        
        if games:
            max_week = max(g.week for g in games if g.week is not None)
            print(f"📅 Found latest week from API: {max_week}")
            return max_week
        else:
            print("⚠️  No games found in API for this season")
            return None
    except Exception as e:
        print(f"⚠️  Could not query API: {e}")
        return None

def detect_current_week(season: int) -> int:
    """
    Auto-detect current week for a season.
    Tries games data first, then API.
    """
    print(f"\n🔍 Auto-detecting current week for season {season}...")
    
    # Try games data first (fastest)
    week = detect_current_week_from_data(season)
    if week is not None:
        return week
    
    # Fall back to API
    print("   Games data not found, checking API...")
    week = detect_current_week_from_api(season)
    if week is not None:
        return week
    
    # If both fail, return None
    print("❌ Could not detect current week")
    return None

def run_script(name: str, script_path: str, needs_args: bool = False, season: int = None, week: int = None):
    """Run a script and handle errors."""
    print(f"\n{'='*70}")
    print(f"Step: {name}")
    print(f"Script: {script_path}")
    print(f"{'='*70}\n")
    
    if not os.path.exists(script_path):
        print(f"ERROR: Script not found: {script_path}")
        return False
    
    try:
        cmd = [sys.executable, script_path]
        if needs_args and season and week:
            cmd.extend(["--season", str(season), "--week", str(week)])
        
        result = subprocess.run(
            cmd,
            check=True,
            cwd=Path(__file__).parent.parent  # Run from project root
        )
        print(f"✓ {name} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {name} failed with exit code {e.returncode}")
        return False
    except Exception as e:
        print(f"✗ {name} failed with error: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(
        description="Update all data and generate current rankings"
    )
    parser.add_argument(
        "--season",
        type=int,
        required=True,
        help="Season year (e.g., 2025)"
    )
    parser.add_argument(
        "--week",
        type=int,
        default=None,
        help="Week number to generate rankings for (auto-detected if not provided)"
    )
    parser.add_argument(
        "--skip-games",
        action="store_true",
        help="Skip fetching games (use if games are already up to date)"
    )
    parser.add_argument(
        "--skip-features",
        action="store_true",
        help="Skip feature building (use if features are already up to date)"
    )
    parser.add_argument(
        "--only-rankings",
        action="store_true",
        help="Only generate rankings (skip all data updates)"
    )
    
    args = parser.parse_args()
    
    # Auto-detect week if not provided
    week = args.week
    if week is None:
        detected_week = detect_current_week(args.season)
        if detected_week is None:
            print("\n❌ Could not auto-detect week. Please specify --week manually.")
            sys.exit(1)
        week = detected_week
        print(f"✅ Using week {week}")
    
    print("\n" + "="*70)
    print("CFB MODEL - UPDATE DATA & GENERATE RANKINGS")
    print("="*70)
    print(f"Season: {args.season}")
    print(f"Week: {week} {'(auto-detected)' if args.week is None else ''}")
    print("="*70 + "\n")
    
    if args.only_rankings:
        print("Mode: Only generating rankings (skipping data updates)")
        success = run_script(
            "Generate rankings",
            "scripts/10_generate_weekly_ratings.py",
            needs_args=True,
            season=args.season,
            week=week
        )
        sys.exit(0 if success else 1)
    
    # Run pipeline steps
    failed_steps = []
    
    for i, (name, script_path, needs_args) in enumerate(WEEKLY_UPDATE_SCRIPTS, 1):
        # Skip games if requested
        if args.skip_games and "fetch_games" in script_path.lower():
            print(f"\n⏭️  Skipping: {name} (--skip-games flag)")
            continue
        
        # Skip feature building if requested
        if args.skip_features and any(x in script_path.lower() for x in ["features", "sos", "priors", "opponent", "resume"]):
            if "Generate rankings" not in name:  # Don't skip rankings generation
                print(f"\n⏭️  Skipping: {name} (--skip-features flag)")
                continue
        
        # Generate rankings always needs season/week
        if needs_args:
            success = run_script(name, script_path, needs_args=True, season=args.season, week=week)
        else:
            success = run_script(name, script_path)
        
        if not success:
            failed_steps.append(name)
            response = input(f"\n⚠️  {name} failed. Continue anyway? (y/n): ")
            if response.lower() != 'y':
                print("\n❌ Pipeline stopped by user")
                sys.exit(1)
    
    # Summary
    print("\n" + "="*70)
    if failed_steps:
        print("⚠️  PIPELINE COMPLETED WITH ERRORS")
        print(f"Failed steps: {', '.join(failed_steps)}")
    else:
        print("✅ PIPELINE COMPLETED SUCCESSFULLY")
    print("="*70)
    print(f"\nRankings generated: outputs/top25_{args.season}_week{week}.json")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()