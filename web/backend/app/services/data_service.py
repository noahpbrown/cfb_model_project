import os
import json
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional

# Get project root - works in both local and Railway
# In Railway with root directory = repo root, /app is the repo root
if os.path.exists("/app"):
    # Railway deployment - /app is now the repo root
    PROJECT_ROOT = Path("/app")
elif os.path.exists(Path(__file__).parent.parent.parent.parent.parent.parent):
    # Local development
    PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.parent.parent
else:
    # Fallback
    PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.parent.parent

def load_rankings_json(season: int, week: int) -> List[Dict[str, Any]]:
    """
    Load rankings from JSON file.
    """
    json_path = PROJECT_ROOT / "outputs" / f"top25_{season}_week{week}.json"
    
    if not json_path.exists():
        raise FileNotFoundError(f"Rankings file not found: {json_path}")
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    return data

def get_available_weeks(season: int) -> List[int]:
    """
    Get list of available weeks for a season by checking output files.
    """
    outputs_dir = PROJECT_ROOT / "outputs"
    weeks = []
    
    for file in outputs_dir.glob(f"top25_{season}_week*.json"):
        # Extract week number from filename
        week_str = file.stem.split("_week")[-1]
        try:
            weeks.append(int(week_str))
        except ValueError:
            continue
    
    return sorted(weeks)