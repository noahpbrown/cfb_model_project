#!/usr/bin/env python

"""
Generate rankings for all weeks in a season.
Usage: python scripts/10b_generate_all_weeks.py --season 2025
"""

import argparse
import subprocess
import sys

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, required=True, help="Season year")
    parser.add_argument("--start-week", type=int, default=1, help="Starting week (default: 1)")
    parser.add_argument("--end-week", type=int, default=16, help="Ending week (default: 16)")
    args = parser.parse_args()
    
    print(f"Generating rankings for {args.season} season, weeks {args.start_week} through {args.end_week}...")
    
    for week in range(args.start_week, args.end_week + 1):
        print(f"\n{'='*60}")
        print(f"Week {week}")
        print(f"{'='*60}")
        
        try:
            result = subprocess.run(
                [
                    sys.executable,
                    "scripts/10_generate_weekly_ratings.py",
                    "--season", str(args.season),
                    "--week", str(week)
                ],
                check=True,
                capture_output=False
            )
            print(f"✓ Week {week} completed successfully")
        except subprocess.CalledProcessError as e:
            print(f"✗ Week {week} failed: {e}")
            # Continue with next week even if one fails
            continue
    
    print(f"\n{'='*60}")
    print("All weeks completed!")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()