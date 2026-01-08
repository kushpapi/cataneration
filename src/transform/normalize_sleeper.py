"""Normalize Sleeper data into staging tables."""

import json
from pathlib import Path
import pandas as pd
from src.common.schemas import StagingTeam, StagingMatchup


def normalize_sleeper_season(season: int, league_id: str) -> None:
    """
    Transform Sleeper raw data into staging tables.

    Args:
        season: Season year
        league_id: Sleeper league ID
    """
    raw_dir = Path(f"data/raw/sleeper/{season}")
    staging_dir = Path(f"data/staging/sleeper/{season}")
    staging_dir.mkdir(parents=True, exist_ok=True)

    # Load league settings to determine playoff start week
    with open(raw_dir / f"{league_id}_league.json") as f:
        league_data = json.load(f)

    playoff_week_start = league_data.get("settings", {}).get("playoff_week_start", 15)

    # Load rosters (teams)
    with open(raw_dir / f"{league_id}_rosters.json") as f:
        rosters_data = json.load(f)

    # Load users (owners)
    with open(raw_dir / f"{league_id}_users.json") as f:
        users_data = json.load(f)

    # Create user_id → user mapping
    user_map = {user["user_id"]: user for user in users_data}

    # Normalize teams
    teams = []

    for roster in rosters_data:
        roster_id = roster["roster_id"]
        owner_id = roster.get("owner_id")

        # Get team name from user metadata
        team_name = None
        if owner_id and owner_id in user_map:
            user = user_map[owner_id]
            team_name = user.get("metadata", {}).get("team_name")
            # Fall back to display name if no team name
            if not team_name:
                team_name = user.get("display_name", f"Team {roster_id}")
        else:
            team_name = f"Team {roster_id}"

        team = StagingTeam(
            platform="sleeper",
            season=season,
            platform_league_id=league_id,
            platform_team_id=str(roster_id),
            team_name=team_name
        )
        teams.append(team.model_dump())

    # Write teams to CSV
    teams_df = pd.DataFrame(teams)
    teams_df.to_csv(staging_dir / "stg_teams.csv", index=False)

    # Normalize matchups - collect from all matchup files
    matchups = []

    # Find all matchup files
    matchup_files = sorted(raw_dir.glob(f"{league_id}_matchups_week*.json"))

    for matchup_file in matchup_files:
        # Extract week number from filename
        week_num = int(matchup_file.stem.split("week")[1])

        with open(matchup_file) as f:
            matchups_data = json.load(f)

        # Check if there are matchups
        if not matchups_data:
            continue

        # Group matchups by matchup_id
        # Each matchup_id represents a pair of teams playing each other
        matchup_groups = {}
        for matchup in matchups_data:
            matchup_id = matchup.get("matchup_id")
            roster_id = matchup.get("roster_id")
            points = matchup.get("points", 0)

            # Skip if no matchup_id (bye week)
            if matchup_id is None:
                continue

            if matchup_id not in matchup_groups:
                matchup_groups[matchup_id] = []

            matchup_groups[matchup_id].append({
                "roster_id": roster_id,
                "points": points
            })

        # Convert matchup groups to staging matchups
        for matchup_id, teams in matchup_groups.items():
            # Each matchup should have exactly 2 teams
            if len(teams) != 2:
                print(f"  [WARN] Week {week_num} matchup {matchup_id} has {len(teams)} teams (expected 2)")
                continue

            # Determine home/away (Sleeper doesn't distinguish, so we'll use roster_id order)
            teams_sorted = sorted(teams, key=lambda x: x["roster_id"])
            team_away = teams_sorted[0]
            team_home = teams_sorted[1]

            # Determine if playoffs
            is_playoffs = week_num >= playoff_week_start

            matchup_obj = StagingMatchup(
                platform="sleeper",
                season=season,
                platform_league_id=league_id,
                week=week_num,
                platform_matchup_id=f"{week_num}_{matchup_id}",  # Combine week + matchup_id for uniqueness
                platform_team_id_home=str(team_home["roster_id"]),
                platform_team_id_away=str(team_away["roster_id"]),
                score_home=float(team_home["points"]),
                score_away=float(team_away["points"]),
                is_playoffs=is_playoffs
            )
            matchups.append(matchup_obj.model_dump())

    # Write matchups to CSV
    matchups_df = pd.DataFrame(matchups)
    matchups_df.to_csv(staging_dir / "stg_matchups.csv", index=False)

    print(f"✓ Normalized Sleeper {season}: {len(teams)} teams, {len(matchups)} matchups")


if __name__ == "__main__":
    print("Testing Sleeper 2025 normalization...")
    normalize_sleeper_season(2025, "1269081387739136000")
