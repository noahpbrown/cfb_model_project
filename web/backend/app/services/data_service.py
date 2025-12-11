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
else:
    # Local development - go up from web/backend/app/services/data_service.py
    # to reach cfb_model_project/ (5 levels up)
    current_file = Path(__file__).resolve()
    # Go up: services -> app -> backend -> web -> cfb_model_project
    PROJECT_ROOT = current_file.parent.parent.parent.parent.parent
    
    # Verify we're in the right place by checking for outputs/ directory
    if not (PROJECT_ROOT / "outputs").exists() and (PROJECT_ROOT / "web" / "backend" / "outputs").exists():
        # If outputs is in web/backend, use that as the base
        PROJECT_ROOT = PROJECT_ROOT / "web" / "backend"

def load_rankings_json(season: int, week: int) -> List[Dict[str, Any]]:
    """
    Load rankings from JSON file.
    """
    # Try project root first, then web/backend
    json_path = PROJECT_ROOT / "outputs" / f"top25_{season}_week{week}.json"
    
    if not json_path.exists():
        # Fallback: try web/backend/outputs
        fallback_path = PROJECT_ROOT / "web" / "backend" / "outputs" / f"top25_{season}_week{week}.json"
        if fallback_path.exists():
            json_path = fallback_path
        else:
            raise FileNotFoundError(f"Rankings file not found: {json_path} or {fallback_path}")
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    return data

def get_available_weeks(season: int) -> List[int]:
    """
    Get list of available weeks for a season by checking output files.
    """
    outputs_dir = PROJECT_ROOT / "outputs"
    
    # Fallback to web/backend/outputs if needed
    if not outputs_dir.exists():
        outputs_dir = PROJECT_ROOT / "web" / "backend" / "outputs"
    
    weeks = []
    
    for file in outputs_dir.glob(f"top25_{season}_week*.json"):
        # Extract week number from filename
        week_str = file.stem.split("_week")[-1]
        try:
            weeks.append(int(week_str))
        except ValueError:
            continue
    
    return sorted(weeks)