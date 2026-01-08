"""Normalize Fleaflicker data into staging tables."""

import json
from pathlib import Path
import pandas as pd
from src.common.schemas import StagingTeam, StagingMatchup


def normalize_fleaflicker_season(season: int, league_id: str) -> None:
    """
    Transform Fleaflicker raw data into staging tables.

    Args:
        season: Season year
        league_id: Fleaflicker league ID
    """
    raw_dir = Path(f"data/raw/fleaflicker/{season}")
    staging_dir = Path(f"data/staging/fleaflicker/{season}")
    staging_dir.mkdir(parents=True, exist_ok=True)

    # Load standings (has team info)
    with open(raw_dir / f"{league_id}_standings.json") as f:
        standings_data = json.load(f)

    # Normalize teams
    teams = []
    divisions = standings_data.get("divisions", [])

    for division in divisions:
        for team_data in division.get("teams", []):
            team = StagingTeam(
                platform="fleaflicker",
                season=season,
                platform_league_id=league_id,
                platform_team_id=str(team_data["id"]),
                team_name=team_data["name"]
            )
            teams.append(team.model_dump())

    # Write teams to CSV
    teams_df = pd.DataFrame(teams)
    teams_df.to_csv(staging_dir / "stg_teams.csv", index=False)

    # Normalize matchups - collect from all scoreboard files
    matchups = []

    # Find all scoreboard files
    scoreboard_files = sorted(raw_dir.glob(f"{league_id}_scoreboard_week*.json"))

    for scoreboard_file in scoreboard_files:
        # Extract week number from filename
        week_num = int(scoreboard_file.stem.split("week")[1])

        with open(scoreboard_file) as f:
            scoreboard_data = json.load(f)

        # Check if there are games
        games = scoreboard_data.get("games", [])
        if not games:
            continue

        for game in games:
            # Skip games without scores (not yet played)
            if "awayScore" not in game or "homeScore" not in game:
                continue

            # Extract scores
            away_score = game["awayScore"].get("score", {}).get("value", 0)
            home_score = game["homeScore"].get("score", {}).get("value", 0)

            # Determine if playoffs based on game metadata
            # Fleaflicker marks playoff games, but for now we'll determine based on week
            # Typically weeks 15-16 are playoffs in this league
            is_playoffs = week_num >= 15

            matchup_obj = StagingMatchup(
                platform="fleaflicker",
                season=season,
                platform_league_id=league_id,
                week=week_num,
                platform_matchup_id=str(game["id"]),  # Fleaflicker provides game IDs!
                platform_team_id_home=str(game["home"]["id"]),
                platform_team_id_away=str(game["away"]["id"]),
                score_home=float(home_score),
                score_away=float(away_score),
                is_playoffs=is_playoffs
            )
            matchups.append(matchup_obj.model_dump())

    # Write matchups to CSV
    matchups_df = pd.DataFrame(matchups)
    matchups_df.to_csv(staging_dir / "stg_matchups.csv", index=False)

    print(f"✓ Normalized Fleaflicker {season}: {len(teams)} teams, {len(matchups)} matchups")


if __name__ == "__main__":
    print("Testing Fleaflicker 2020 normalization...")
    normalize_fleaflicker_season(2020, "311194")
