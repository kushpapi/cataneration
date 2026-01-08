"""Canonical ID generation functions.

Implements the ID scheme from CLAUDE.md:
- owner_id: Stable slug (e.g., "ben_mccoy")
- team_id: {platform}:{season}:{league_id}:{platform_team_id}
- game_id: {platform}:{season}:{league_id}:{week}:{matchup_id}
"""

import hashlib


def generate_team_id(
    platform: str,
    season: int,
    league_id: str,
    platform_team_id: str
) -> str:
    """
    Generate canonical team ID.

    Args:
        platform: Platform name (mfl, fleaflicker, sleeper)
        season: Season year
        league_id: Platform-specific league ID
        platform_team_id: Platform-specific team ID

    Returns:
        Canonical team ID string
    """
    return f"{platform}:{season}:{league_id}:{platform_team_id}"


def generate_game_id(
    platform: str,
    season: int,
    league_id: str,
    week: int,
    matchup_id: str | None = None,
    team_id_home: str | None = None,
    team_id_away: str | None = None
) -> str:
    """
    Generate canonical game ID.

    Args:
        platform: Platform name
        season: Season year
        league_id: Platform-specific league ID
        week: Week number
        matchup_id: Platform-specific matchup ID (if available)
        team_id_home: Home team ID (for deriving matchup_id if missing)
        team_id_away: Away team ID (for deriving matchup_id if missing)

    Returns:
        Canonical game ID string

    Raises:
        ValueError: If matchup_id is missing and team IDs not provided
    """
    if matchup_id is None:
        if team_id_home is None or team_id_away is None:
            raise ValueError(
                "Either matchup_id or both team_id_home and team_id_away required"
            )
        # Derive deterministic matchup_id from sorted team IDs
        sorted_teams = sorted([team_id_home, team_id_away])
        matchup_str = f"{sorted_teams[0]}_{sorted_teams[1]}"
        matchup_id = hashlib.md5(matchup_str.encode()).hexdigest()[:8]

    return f"{platform}:{season}:{league_id}:{week}:{matchup_id}"


def slugify_owner_id(name: str) -> str:
    """
    Convert display name to owner_id slug.

    Args:
        name: Display name (e.g., "Ben McCoy")

    Returns:
        Slugified owner_id (e.g., "ben_mccoy")
    """
    return name.lower().replace(' ', '_').replace('-', '_')
