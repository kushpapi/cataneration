"""Normalize MFL data into staging tables."""

import json
from pathlib import Path
import pandas as pd
from src.common.schemas import StagingTeam, StagingMatchup


def normalize_mfl_season(season: int, league_id: str) -> None:
    """
    Transform MFL raw data into staging tables.

    Args:
        season: Season year
        league_id: MFL league ID
    """
    raw_dir = Path(f"data/raw/mfl/{season}")
    staging_dir = Path(f"data/staging/mfl/{season}")
    staging_dir.mkdir(parents=True, exist_ok=True)

    # Load raw data
    with open(raw_dir / f"{league_id}_league.json") as f:
        franchises_data = json.load(f)

    with open(raw_dir / f"{league_id}_schedule.json") as f:
        schedule_data = json.load(f)

    # Normalize teams
    teams = []
    franchises = franchises_data.get("league", {}).get("franchises", {}).get("franchise", [])

    # Handle single franchise case (API returns dict instead of list)
    if isinstance(franchises, dict):
        franchises = [franchises]

    for franchise in franchises:
        team = StagingTeam(
            platform="mfl",
            season=season,
            platform_league_id=league_id,
            platform_team_id=franchise["id"],
            team_name=franchise["name"]
        )
        teams.append(team.model_dump())

    # Write teams to CSV
    teams_df = pd.DataFrame(teams)
    teams_df.to_csv(staging_dir / "stg_teams.csv", index=False)

    expected_matchups = len(teams) // 2 if teams else 0

    # Normalize matchups
    matchups = []
    weekly_schedule = schedule_data.get("schedule", {}).get("weeklySchedule", [])

    # Handle single week case
    if isinstance(weekly_schedule, dict):
        weekly_schedule = [weekly_schedule]

    playoff_start_week = None
    if expected_matchups > 0:
        for week_idx, week_data in enumerate(weekly_schedule, start=1):
            matchup_list = week_data.get("matchup", [])
            if isinstance(matchup_list, dict):
                matchup_list = [matchup_list]
            if 0 < len(matchup_list) < expected_matchups:
                playoff_start_week = week_idx
                break

    for week_idx, week_data in enumerate(weekly_schedule, start=1):
        matchup_list = week_data.get("matchup", [])

        # Handle single matchup case
        if isinstance(matchup_list, dict):
            matchup_list = [matchup_list]

        for matchup in matchup_list:
            # MFL matchups have a list of franchise entries
            franchise_scores = matchup.get("franchise", [])

            # Handle single franchise case
            if isinstance(franchise_scores, dict):
                franchise_scores = [franchise_scores]

            # MFL provides exactly 2 franchises per matchup (or 1 for bye)
            if len(franchise_scores) != 2:
                # Skip byes for now
                continue

            # Determine home/away based on isHome flag
            if franchise_scores[0].get("isHome") == "1":
                home = franchise_scores[0]
                away = franchise_scores[1]
            else:
                home = franchise_scores[1]
                away = franchise_scores[0]

            matchup_obj = StagingMatchup(
                platform="mfl",
                season=season,
                platform_league_id=league_id,
                week=week_idx,
                platform_matchup_id=None,  # MFL doesn't provide matchup IDs
                platform_team_id_home=home["id"],
                platform_team_id_away=away["id"],
                score_home=float(home.get("score", 0)),
                score_away=float(away.get("score", 0)),
                is_playoffs=(
                    week_idx >= playoff_start_week
                    if playoff_start_week is not None
                    else None
                )
            )
            matchups.append(matchup_obj.model_dump())

    # Write matchups to CSV
    matchups_df = pd.DataFrame(matchups)
    matchups_df.to_csv(staging_dir / "stg_matchups.csv", index=False)

    print(f"✓ Normalized MFL {season}: {len(teams)} teams, {len(matchups)} matchups")
