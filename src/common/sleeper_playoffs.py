"""Sleeper playoff helpers for winners/losers brackets."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import requests

BASE_URL = "https://api.sleeper.app/v1"


def get_winners_bracket(league_id: str) -> list:
    """Fetch raw winners bracket JSON for a Sleeper league."""
    url = f"{BASE_URL}/league/{league_id}/winners_bracket"
    response = requests.get(url, timeout=30)
    if response.status_code != 200:
        raise RuntimeError(f"Sleeper winners_bracket request failed ({response.status_code})")
    return response.json()


def get_losers_bracket(league_id: str) -> list:
    """Fetch raw losers bracket JSON for a Sleeper league."""
    url = f"{BASE_URL}/league/{league_id}/losers_bracket"
    response = requests.get(url, timeout=30)
    if response.status_code != 200:
        raise RuntimeError(f"Sleeper losers_bracket request failed ({response.status_code})")
    return response.json()


def _extract_roster_id(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    if isinstance(value, dict):
        for key in ("id", "roster_id"):
            if key in value and isinstance(value[key], int):
                return value[key]
            if key in value and isinstance(value[key], str) and value[key].isdigit():
                return int(value[key])
    return None


def derive_champion_roster_id(winners_bracket_json: list) -> Optional[int]:
    """
    Derive the champion roster_id from the placement matchup (p == 1).
    Returns None if the championship is not decided yet.
    """
    if not winners_bracket_json:
        return None

    for matchup in winners_bracket_json:
        if not isinstance(matchup, dict):
            continue
        if matchup.get("p") == 1:
            return _extract_roster_id(matchup.get("w"))
    return None


def derive_playoff_roster_ids(winners_bracket_json: list) -> List[int]:
    """
    Derive unique playoff roster_ids from winners bracket entries.
    Includes concrete roster IDs from t1/t2 and resolved w/l fields.
    """
    roster_ids = set()
    if not winners_bracket_json:
        return []

    for matchup in winners_bracket_json:
        if not isinstance(matchup, dict):
            continue
        for field in ("t1", "t2", "w", "l"):
            roster_id = _extract_roster_id(matchup.get(field))
            if roster_id is not None:
                roster_ids.add(roster_id)

    return sorted(roster_ids)


def derive_bracket_seed_map(winners_bracket_json: list) -> Dict[str, int]:
    """
    Map initial playoff round bracket slots to roster_ids.
    Uses the earliest round number as the initial playoff round.
    """
    seed_map: Dict[str, int] = {}
    if not winners_bracket_json:
        return seed_map

    rounds = [
        matchup.get("r")
        for matchup in winners_bracket_json
        if isinstance(matchup, dict) and isinstance(matchup.get("r"), int)
    ]
    if not rounds:
        return seed_map

    min_round = min(rounds)

    for matchup in winners_bracket_json:
        if not isinstance(matchup, dict):
            continue
        if matchup.get("r") != min_round:
            continue
        match_id = matchup.get("m")
        if not isinstance(match_id, int):
            continue

        t1_id = _extract_roster_id(matchup.get("t1"))
        if t1_id is not None:
            seed_map[f"R{min_round}_M{match_id}_T1"] = t1_id

        t2_id = _extract_roster_id(matchup.get("t2"))
        if t2_id is not None:
            seed_map[f"R{min_round}_M{match_id}_T2"] = t2_id

    return seed_map
