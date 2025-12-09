from fastapi import APIRouter, HTTPException, Query
from typing import List, Optional
from app.models.schemas import Team
from app.services.data_service import load_rankings_json, get_available_weeks
import json

router = APIRouter()

@router.get("/", response_model=List[Team])
async def get_rankings(
    season: int = Query(2025, description="Season year"),
    week: int = Query(14, description="Week number")
):
    """
    Get Top 25 rankings for a specific season and week.
    """
    try:
        rankings = load_rankings_json(season, week)
        return rankings
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading rankings: {str(e)}")

@router.get("/weeks")
async def get_available_weeks_for_season(
    season: int = Query(2025, description="Season year")
):
    """
    Get list of available weeks for a season.
    """
    try:
        weeks = get_available_weeks(season)
        return {"season": season, "weeks": weeks}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting weeks: {str(e)}")