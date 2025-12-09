from pydantic import BaseModel
from typing import Optional

class Team(BaseModel):
    rank: int
    team: str
    abbr: str
    wins_cum: float
    losses_cum: float
    games_played: int
    rating_pred: float
    spread_vs_1: float
    primary_color: str
    primary_logo_url: str
    team_id: Optional[int] = None

    class Config:
        from_attributes = True