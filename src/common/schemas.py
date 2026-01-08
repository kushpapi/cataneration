"""Pydantic schemas for data validation."""

from typing import Optional
from pydantic import BaseModel, Field


# Staging schemas
class StagingTeam(BaseModel):
    """Schema for staging teams table."""
    platform: str
    season: int
    platform_league_id: str
    platform_team_id: str
    team_name: str


class StagingMatchup(BaseModel):
    """Schema for staging matchups table."""
    platform: str
    season: int
    platform_league_id: str
    week: int
    platform_matchup_id: Optional[str] = None
    platform_team_id_home: str
    platform_team_id_away: str
    score_home: float
    score_away: float
    is_playoffs: Optional[bool] = None


# Mart schemas
class MartMatchup(BaseModel):
    """Schema for mart matchups table."""
    season: int
    week: int
    game_id: str
    platform: str
    owner_id_home: str
    owner_id_away: str
    score_home: float
    score_away: float
    winner_owner_id: str  # "tie" if tied
    is_playoffs: bool


class MartOwnerAllTime(BaseModel):
    """Schema for mart owner all-time stats table."""
    owner_id: str
    games: int = Field(ge=0)
    wins: int = Field(ge=0)
    losses: int = Field(ge=0)
    ties: int = Field(ge=0)
    win_pct: float = Field(ge=0.0, le=1.0)
    points_for: float
    points_against: float
    avg_points_for: float
    avg_points_against: float


class MartOwnerSeason(BaseModel):
    """Schema for mart owner season stats table."""
    season: int
    owner_id: str
    wins: int = Field(ge=0)
    losses: int = Field(ge=0)
    ties: int = Field(ge=0)
    win_pct: float = Field(ge=0.0, le=1.0)
    points_for: float
    points_against: float


class MartH2H(BaseModel):
    """Schema for mart head-to-head table."""
    owner_a: str
    owner_b: str
    games: int = Field(ge=0)
    a_wins: int = Field(ge=0)
    b_wins: int = Field(ge=0)
    ties: int = Field(ge=0)
    a_points_for: float
    b_points_for: float


# Owner mapping schemas
class Owner(BaseModel):
    """Schema for owners.csv."""
    owner_id: str
    display_name: str


class OwnerAlias(BaseModel):
    """Schema for owner_aliases.csv."""
    platform: str
    platform_user_id: str
    owner_id: str


class TeamAlias(BaseModel):
    """Schema for team_aliases.csv."""
    platform: str
    season: int
    team_name: str
    owner_id: str
