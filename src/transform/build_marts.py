"""Build mart tables from staging data with owner resolution."""

from pathlib import Path
from typing import Optional
import json
import pandas as pd
from src.common.ids import generate_game_id, generate_team_id
from src.common.sleeper_playoffs import derive_champion_roster_id


def load_owner_mappings() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load owner mapping files.

    Returns:
        Tuple of (owner_aliases_df, team_aliases_df)
    """
    owner_aliases = pd.read_csv("assets/owner_aliases.csv")
    team_aliases = pd.read_csv("assets/team_aliases.csv")
    return owner_aliases, team_aliases


def resolve_owner_id(
    platform: str,
    season: int,
    team_name: str,
    platform_user_id: Optional[str],
    owner_aliases: pd.DataFrame,
    team_aliases: pd.DataFrame
) -> str:
    """
    Resolve canonical owner_id from platform identifiers.

    Args:
        platform: Platform name
        season: Season year
        team_name: Team name
        platform_user_id: Platform user ID (if available)
        owner_aliases: Owner aliases DataFrame
        team_aliases: Team aliases DataFrame

    Returns:
        Canonical owner_id or UNMAPPED_{platform}_{identifier}
    """
    # Try owner_aliases first (for platforms with stable user IDs)
    if platform_user_id:
        match = owner_aliases[
            (owner_aliases["platform"] == platform) &
            (owner_aliases["platform_user_id"] == platform_user_id)
        ]
        if not match.empty:
            return match.iloc[0]["owner_id"]

    # Fall back to team_aliases (for platforms like MFL)
    match = team_aliases[
        (team_aliases["platform"] == platform) &
        (team_aliases["season"] == season) &
        (team_aliases["team_name"] == team_name)
    ]
    if not match.empty:
        return match.iloc[0]["owner_id"]

    # Return unmapped placeholder
    identifier = platform_user_id if platform_user_id else team_name.replace(" ", "_")
    return f"UNMAPPED_{platform}_{identifier}"


def build_mart_matchups() -> pd.DataFrame:
    """
    Build mart_matchups table from all staging matchups.

    Returns:
        mart_matchups DataFrame
    """
    owner_aliases, team_aliases = load_owner_mappings()

    # Load all staging data
    staging_dir = Path("data/staging")
    all_matchups = []
    all_teams = {}  # Cache team lookups

    # First, load all teams for owner resolution
    for staging_file in staging_dir.rglob("stg_teams.csv"):
        parts = staging_file.parts
        platform = parts[2]
        season = int(parts[3])

        teams_df = pd.read_csv(staging_file)
        for _, team in teams_df.iterrows():
            team_id = generate_team_id(
                platform=platform,
                season=season,
                league_id=team["platform_league_id"],
                platform_team_id=team["platform_team_id"]
            )
            all_teams[team_id] = {
                "platform": platform,
                "season": season,
                "team_name": team["team_name"],
                "platform_user_id": None  # MFL doesn't have user IDs
            }

    # Now process matchups
    for staging_file in staging_dir.rglob("stg_matchups.csv"):
        parts = staging_file.parts
        platform = parts[2]
        season = int(parts[3])

        matchups_df = pd.read_csv(staging_file)

        for _, matchup in matchups_df.iterrows():
            # Generate team IDs
            team_id_home = generate_team_id(
                platform=platform,
                season=season,
                league_id=matchup["platform_league_id"],
                platform_team_id=matchup["platform_team_id_home"]
            )
            team_id_away = generate_team_id(
                platform=platform,
                season=season,
                league_id=matchup["platform_league_id"],
                platform_team_id=matchup["platform_team_id_away"]
            )

            # Generate game ID (convert pandas NaN to None)
            matchup_id_value = matchup.get("platform_matchup_id")
            if pd.isna(matchup_id_value):
                matchup_id_value = None

            game_id = generate_game_id(
                platform=platform,
                season=season,
                league_id=matchup["platform_league_id"],
                week=matchup["week"],
                matchup_id=matchup_id_value,
                team_id_home=team_id_home,
                team_id_away=team_id_away
            )

            # Resolve owner IDs
            home_team = all_teams[team_id_home]
            away_team = all_teams[team_id_away]

            owner_id_home = resolve_owner_id(
                platform=platform,
                season=season,
                team_name=home_team["team_name"],
                platform_user_id=home_team["platform_user_id"],
                owner_aliases=owner_aliases,
                team_aliases=team_aliases
            )
            owner_id_away = resolve_owner_id(
                platform=platform,
                season=season,
                team_name=away_team["team_name"],
                platform_user_id=away_team["platform_user_id"],
                owner_aliases=owner_aliases,
                team_aliases=team_aliases
            )

            # Determine winner
            score_home = matchup["score_home"]
            score_away = matchup["score_away"]
            if score_home > score_away:
                winner = owner_id_home
            elif score_away > score_home:
                winner = owner_id_away
            else:
                winner = "tie"

            # Add to results
            is_playoffs = matchup.get("is_playoffs", None)
            if pd.isna(is_playoffs):
                is_playoffs = None
            if is_playoffs is None:
                is_playoffs = matchup["week"] >= 15
            all_matchups.append({
                "season": season,
                "week": matchup["week"],
                "game_id": game_id,
                "platform": platform,
                "owner_id_home": owner_id_home,
                "owner_id_away": owner_id_away,
                "score_home": score_home,
                "score_away": score_away,
                "winner_owner_id": winner,
                "is_playoffs": bool(is_playoffs)
            })

    return pd.DataFrame(all_matchups)


def build_mart_owner_season(mart_matchups: pd.DataFrame) -> pd.DataFrame:
    """
    Build mart_owner_season table from mart_matchups.

    Args:
        mart_matchups: mart_matchups DataFrame

    Returns:
        mart_owner_season DataFrame
    """
    season_stats = []

    for season in mart_matchups["season"].unique():
        season_games = mart_matchups[mart_matchups["season"] == season]

        # Get all unique owners in this season
        owners = set(season_games["owner_id_home"]) | set(season_games["owner_id_away"])

        for owner_id in owners:
            # Find all games for this owner
            home_games = season_games[season_games["owner_id_home"] == owner_id]
            away_games = season_games[season_games["owner_id_away"] == owner_id]

            # Calculate stats
            wins = len(home_games[home_games["winner_owner_id"] == owner_id]) + \
                   len(away_games[away_games["winner_owner_id"] == owner_id])
            losses = len(home_games[home_games["winner_owner_id"] == home_games["owner_id_away"]]) + \
                     len(away_games[away_games["winner_owner_id"] == away_games["owner_id_home"]])
            ties = len(home_games[home_games["winner_owner_id"] == "tie"]) + \
                   len(away_games[away_games["winner_owner_id"] == "tie"])

            points_for = home_games["score_home"].sum() + away_games["score_away"].sum()
            points_against = home_games["score_away"].sum() + away_games["score_home"].sum()

            games = wins + losses + ties
            win_pct = wins / games if games > 0 else 0.0

            season_stats.append({
                "season": season,
                "owner_id": owner_id,
                "wins": wins,
                "losses": losses,
                "ties": ties,
                "win_pct": round(win_pct, 4),
                "points_for": round(points_for, 2),
                "points_against": round(points_against, 2)
            })

    return pd.DataFrame(season_stats)


def build_mart_owner_all_time(mart_owner_season: pd.DataFrame) -> pd.DataFrame:
    """
    Build mart_owner_all_time table by aggregating season stats.

    Args:
        mart_owner_season: mart_owner_season DataFrame

    Returns:
        mart_owner_all_time DataFrame
    """
    all_time_stats = []

    for owner_id in mart_owner_season["owner_id"].unique():
        owner_seasons = mart_owner_season[mart_owner_season["owner_id"] == owner_id]

        wins = owner_seasons["wins"].sum()
        losses = owner_seasons["losses"].sum()
        ties = owner_seasons["ties"].sum()
        games = wins + losses + ties

        points_for = owner_seasons["points_for"].sum()
        points_against = owner_seasons["points_against"].sum()

        win_pct = wins / games if games > 0 else 0.0
        avg_points_for = points_for / games if games > 0 else 0.0
        avg_points_against = points_against / games if games > 0 else 0.0

        all_time_stats.append({
            "owner_id": owner_id,
            "games": games,
            "wins": wins,
            "losses": losses,
            "ties": ties,
            "win_pct": round(win_pct, 4),
            "points_for": round(points_for, 2),
            "points_against": round(points_against, 2),
            "avg_points_for": round(avg_points_for, 2),
            "avg_points_against": round(avg_points_against, 2)
        })

    return pd.DataFrame(all_time_stats).sort_values("win_pct", ascending=False)


def _build_owner_season_stats_for_matchups(matchups: pd.DataFrame) -> pd.DataFrame:
    season_stats = []

    if matchups.empty:
        return pd.DataFrame(columns=[
            "season",
            "owner_id",
            "wins",
            "losses",
            "ties",
            "games",
            "win_pct",
            "points_for",
            "points_against"
        ])

    for season in matchups["season"].unique():
        season_games = matchups[matchups["season"] == season]
        owners = set(season_games["owner_id_home"]) | set(season_games["owner_id_away"])

        for owner_id in owners:
            home_games = season_games[season_games["owner_id_home"] == owner_id]
            away_games = season_games[season_games["owner_id_away"] == owner_id]

            wins = len(home_games[home_games["winner_owner_id"] == owner_id]) + \
                   len(away_games[away_games["winner_owner_id"] == owner_id])
            losses = len(home_games[home_games["winner_owner_id"] == home_games["owner_id_away"]]) + \
                     len(away_games[away_games["winner_owner_id"] == away_games["owner_id_home"]])
            ties = len(home_games[home_games["winner_owner_id"] == "tie"]) + \
                   len(away_games[away_games["winner_owner_id"] == "tie"])

            points_for = home_games["score_home"].sum() + away_games["score_away"].sum()
            points_against = home_games["score_away"].sum() + away_games["score_home"].sum()

            games = wins + losses + ties
            win_pct = wins / games if games > 0 else 0.0

            season_stats.append({
                "season": season,
                "owner_id": owner_id,
                "wins": wins,
                "losses": losses,
                "ties": ties,
                "games": games,
                "win_pct": round(win_pct, 4),
                "points_for": round(points_for, 2),
                "points_against": round(points_against, 2)
            })

    return pd.DataFrame(season_stats)


def build_mart_h2h(mart_matchups: pd.DataFrame) -> pd.DataFrame:
    """
    Build mart_h2h table from mart_matchups.

    Args:
        mart_matchups: mart_matchups DataFrame

    Returns:
        mart_h2h DataFrame
    """
    h2h_stats = []
    owners = set(mart_matchups["owner_id_home"]) | set(mart_matchups["owner_id_away"])

    for owner_a in owners:
        for owner_b in owners:
            if owner_a >= owner_b:  # Only process each pair once, alphabetically
                continue

            # Find all matchups between these owners
            matchups_ab = mart_matchups[
                ((mart_matchups["owner_id_home"] == owner_a) &
                 (mart_matchups["owner_id_away"] == owner_b)) |
                ((mart_matchups["owner_id_home"] == owner_b) &
                 (mart_matchups["owner_id_away"] == owner_a))
            ]

            if matchups_ab.empty:
                continue

            # Calculate stats
            a_wins = len(matchups_ab[matchups_ab["winner_owner_id"] == owner_a])
            b_wins = len(matchups_ab[matchups_ab["winner_owner_id"] == owner_b])
            ties = len(matchups_ab[matchups_ab["winner_owner_id"] == "tie"])
            games = len(matchups_ab)

            # Calculate points
            a_home = matchups_ab[matchups_ab["owner_id_home"] == owner_a]
            a_away = matchups_ab[matchups_ab["owner_id_away"] == owner_a]
            a_points = a_home["score_home"].sum() + a_away["score_away"].sum()

            b_home = matchups_ab[matchups_ab["owner_id_home"] == owner_b]
            b_away = matchups_ab[matchups_ab["owner_id_away"] == owner_b]
            b_points = b_home["score_home"].sum() + b_away["score_away"].sum()

            h2h_stats.append({
                "owner_a": owner_a,
                "owner_b": owner_b,
                "games": games,
                "a_wins": a_wins,
                "b_wins": b_wins,
                "ties": ties,
                "a_points_for": round(a_points, 2),
                "b_points_for": round(b_points, 2)
            })

    return pd.DataFrame(h2h_stats)

def _build_team_meta(
    owner_aliases: pd.DataFrame,
    team_aliases: pd.DataFrame
) -> dict:
    team_meta = {}
    staging_dir = Path("data/staging")

    for staging_file in staging_dir.rglob("stg_teams.csv"):
        parts = staging_file.parts
        platform = parts[2]
        season = int(parts[3])

        teams_df = pd.read_csv(staging_file)
        for _, team in teams_df.iterrows():
            platform_team_id = str(team["platform_team_id"])
            platform_league_id = str(team["platform_league_id"])
            team_name = team["team_name"]
            owner_id = resolve_owner_id(
                platform=platform,
                season=season,
                team_name=team_name,
                platform_user_id=None,
                owner_aliases=owner_aliases,
                team_aliases=team_aliases
            )
            team_meta[(platform, season, platform_league_id, platform_team_id)] = {
                "team_name": team_name,
                "owner_id": owner_id
            }

    return team_meta


def _lookup_team_meta(
    team_meta: dict,
    platform: str,
    season: int,
    league_id: str,
    team_id: Optional[str] = None
) -> Optional[dict]:
    if team_id is None:
        return None
    team_id_str = str(team_id)
    candidates = [team_id_str]
    if team_id_str.lstrip("0") and team_id_str.lstrip("0") not in candidates:
        candidates.append(team_id_str.lstrip("0"))
    try:
        candidates.append(str(int(team_id_str)))
    except (TypeError, ValueError):
        pass

    for candidate in candidates:
        key = (platform, season, str(league_id), candidate)
        if key in team_meta:
            return team_meta[key]
    return None


def _sleeper_bracket_winner(bracket: list) -> Optional[str]:
    if not bracket:
        return None
    rounds = [item.get("r") for item in bracket if isinstance(item, dict)]
    round_nums = [r for r in rounds if isinstance(r, int)]
    if round_nums:
        max_round = max(round_nums)
        candidates = [item for item in bracket if item.get("r") == max_round]
    else:
        candidates = bracket
    for matchup in candidates[::-1]:
        winner = matchup.get("w")
        if winner is not None:
            return str(winner)
    return None


def _winner_from_scores(home_score: float, away_score: float, home_id: str, away_id: str) -> Optional[str]:
    if home_score > away_score:
        return home_id
    if away_score > home_score:
        return away_id
    return None


def _fleaflicker_score_value(score_field: object) -> float:
    if isinstance(score_field, dict):
        score_obj = score_field.get("score", score_field)
        if isinstance(score_obj, dict):
            value = score_obj.get("value")
            if isinstance(value, (int, float)):
                return float(value)
    if isinstance(score_field, (int, float)):
        return float(score_field)
    return 0.0


def _select_fleaflicker_losers_game(games: list) -> Optional[dict]:
    candidates = []
    for game in games:
        if game.get("isPlayoffs"):
            continue
        candidates.append(game)

    if not candidates:
        return None

    def game_rank(game: dict) -> int:
        away_rank = game.get("away", {}).get("recordPostseason", {}).get("rank")
        home_rank = game.get("home", {}).get("recordPostseason", {}).get("rank")
        ranks = [r for r in [away_rank, home_rank] if isinstance(r, (int, float))]
        if ranks:
            return int(max(ranks))
        away_rank = game.get("away", {}).get("recordOverall", {}).get("rank")
        home_rank = game.get("home", {}).get("recordOverall", {}).get("rank")
        ranks = [r for r in [away_rank, home_rank] if isinstance(r, (int, float))]
        return int(max(ranks)) if ranks else 0

    return sorted(candidates, key=game_rank)[-1]


def _extract_mfl_matchups(data: dict, bracket_label: Optional[str] = None) -> list:
    matchups = []
    losers_keywords = ("consolation", "loser", "toilet")

    playoff_bracket = data.get("playoffBracket")
    if isinstance(playoff_bracket, dict):
        rounds = playoff_bracket.get("playoffRound", [])
        if isinstance(rounds, dict):
            rounds = [rounds]
        for round_entry in rounds:
            round_num = None
            for key in ("week", "round"):
                value = round_entry.get(key)
                try:
                    round_num = int(value)
                    break
                except (TypeError, ValueError):
                    continue

            games = round_entry.get("playoffGame", [])
            if isinstance(games, dict):
                games = [games]

            for game in games:
                home = game.get("home", {}) if isinstance(game, dict) else {}
                away = game.get("away", {}) if isinstance(game, dict) else {}
                home_id = home.get("franchise_id") or home.get("franchise") or home.get("id")
                away_id = away.get("franchise_id") or away.get("franchise") or away.get("id")
                home_points = home.get("points") or home.get("score")
                away_points = away.get("points") or away.get("score")
                if home_id is None or away_id is None or home_points is None or away_points is None:
                    continue
                try:
                    home_score = float(home_points)
                    away_score = float(away_points)
                except (TypeError, ValueError):
                    continue

                matchups.append({
                    "round": round_num,
                    "label": bracket_label,
                    "team_a": {"id": str(home_id), "score": home_score},
                    "team_b": {"id": str(away_id), "score": away_score}
                })

        if matchups:
            for matchup in matchups:
                label = matchup.get("label") or bracket_label or ""
                matchup["is_losers"] = any(word in label.lower() for word in losers_keywords)
            return matchups

    def parse_team_list(team_list: list) -> Optional[tuple]:
        teams = []
        for team in team_list:
            if not isinstance(team, dict):
                continue
            team_id = team.get("id") or team.get("franchise") or team.get("franchise_id")
            score = team.get("score") or team.get("points") or team.get("totalPoints")
            if team_id is None or score is None:
                continue
            try:
                score_val = float(score)
            except (TypeError, ValueError):
                continue
            teams.append({"id": str(team_id), "score": score_val})
        if len(teams) != 2:
            return None
        return teams[0], teams[1]

    def walk(node, bracket_label=None, round_num=None):
        if isinstance(node, dict):
            label = bracket_label
            for key in ("bracketType", "name", "bracketName", "title"):
                value = node.get(key)
                if isinstance(value, str):
                    label = value
                    break

            if "round" in node and not isinstance(node.get("round"), (list, dict)):
                try:
                    round_num = int(node.get("round"))
                except (TypeError, ValueError):
                    pass

            for key in ("matchup", "game", "games"):
                items = node.get(key)
                if not items:
                    continue
                if isinstance(items, dict):
                    items = [items]
                for item in items:
                    if not isinstance(item, dict):
                        continue
                    team_list = item.get("team") or item.get("franchise")
                    if isinstance(team_list, dict):
                        team_list = [team_list]
                    if isinstance(team_list, list):
                        parsed = parse_team_list(team_list)
                        if parsed:
                            matchups.append({
                                "round": round_num,
                                "label": label,
                                "team_a": parsed[0],
                                "team_b": parsed[1]
                            })

            for value in node.values():
                if isinstance(value, (dict, list)):
                    walk(value, label, round_num)
        elif isinstance(node, list):
            for item in node:
                walk(item, bracket_label, round_num)

    walk(data, bracket_label, None)
    for matchup in matchups:
        label = matchup.get("label") or bracket_label or ""
        matchup["is_losers"] = any(word in label.lower() for word in losers_keywords)
    return matchups


def build_mart_titles(mart_matchups: pd.DataFrame) -> pd.DataFrame:
    """
    Build mart_titles table with champions and losers bracket winners.

    Args:
        mart_matchups: mart_matchups DataFrame

    Returns:
        mart_titles DataFrame
    """
    owner_aliases, team_aliases = load_owner_mappings()
    team_meta = _build_team_meta(owner_aliases, team_aliases)
    titles = []

    # Sleeper titles
    sleeper_raw_dir = Path("data/raw/sleeper")
    if sleeper_raw_dir.exists():
        for season_dir in sleeper_raw_dir.iterdir():
            if not season_dir.is_dir():
                continue
            try:
                season = int(season_dir.name)
            except ValueError:
                continue
            for winners_file in season_dir.glob("*_winners_bracket.json"):
                league_id = winners_file.name.split("_winners_bracket.json")[0]
                winners_bracket = json.loads(winners_file.read_text())
                losers_file = season_dir / f"{league_id}_losers_bracket.json"
                losers_bracket = json.loads(losers_file.read_text()) if losers_file.exists() else []

                champ_team_id = derive_champion_roster_id(winners_bracket)
                losers_team_id = derive_champion_roster_id(losers_bracket)

                champ_meta = _lookup_team_meta(team_meta, "sleeper", season, league_id, champ_team_id)
                losers_meta = _lookup_team_meta(team_meta, "sleeper", season, league_id, losers_team_id)

                titles.append({
                    "season": season,
                    "platform": "sleeper",
                    "platform_league_id": league_id,
                    "champion_owner_id": champ_meta["owner_id"] if champ_meta else None,
                    "champion_team_name": champ_meta["team_name"] if champ_meta else None,
                    "losers_champ_owner_id": losers_meta["owner_id"] if losers_meta else None,
                    "losers_champ_team_name": losers_meta["team_name"] if losers_meta else None
                })

    # Fleaflicker titles
    fleaflicker_raw_dir = Path("data/raw/fleaflicker")
    if fleaflicker_raw_dir.exists():
        for season_dir in fleaflicker_raw_dir.iterdir():
            if not season_dir.is_dir():
                continue
            try:
                season = int(season_dir.name)
            except ValueError:
                continue
            scoreboard_files = list(season_dir.glob("*_scoreboard_week*.json"))
            championship_game = None
            games = []
            league_id = None
            latest_week = -1

            for scoreboard_file in scoreboard_files:
                name = scoreboard_file.name
                if "_scoreboard_week" not in name:
                    continue
                try:
                    week = int(name.split("_scoreboard_week")[1].split(".json")[0])
                except ValueError:
                    continue
                data = json.loads(scoreboard_file.read_text())
                week_games = data.get("games", []) or []
                if not week_games:
                    continue
                champ = next((game for game in week_games if game.get("isChampionshipGame")), None)
                if champ and week >= latest_week:
                    championship_game = champ
                    games = week_games
                    latest_week = week
                    league_id = name.split("_scoreboard_week")[0]

            if not championship_game:
                continue

            champ_home = championship_game.get("home", {})
            champ_away = championship_game.get("away", {})
            champ_home_score = _fleaflicker_score_value(championship_game.get("homeScore", 0))
            champ_away_score = _fleaflicker_score_value(championship_game.get("awayScore", 0))
            champ_winner_id = _winner_from_scores(
                champ_home_score,
                champ_away_score,
                str(champ_home.get("id")),
                str(champ_away.get("id"))
            )

            losers_game = _select_fleaflicker_losers_game(games)
            losers_winner_id = None
            if losers_game:
                losers_home = losers_game.get("home", {})
                losers_away = losers_game.get("away", {})
                losers_home_score = _fleaflicker_score_value(losers_game.get("homeScore", 0))
                losers_away_score = _fleaflicker_score_value(losers_game.get("awayScore", 0))
                losers_winner_id = _winner_from_scores(
                    losers_home_score,
                    losers_away_score,
                    str(losers_home.get("id")),
                    str(losers_away.get("id"))
                )

            champ_meta = _lookup_team_meta(team_meta, "fleaflicker", season, league_id, champ_winner_id)
            losers_meta = _lookup_team_meta(team_meta, "fleaflicker", season, league_id, losers_winner_id)

            titles.append({
                "season": season,
                "platform": "fleaflicker",
                "platform_league_id": league_id,
                "champion_owner_id": champ_meta["owner_id"] if champ_meta else None,
                "champion_team_name": champ_meta["team_name"] if champ_meta else None,
                "losers_champ_owner_id": losers_meta["owner_id"] if losers_meta else None,
                "losers_champ_team_name": losers_meta["team_name"] if losers_meta else None
            })

    # MFL titles
    mfl_raw_dir = Path("data/raw/mfl")
    if mfl_raw_dir.exists():
        for season_dir in mfl_raw_dir.iterdir():
            if not season_dir.is_dir():
                continue
            try:
                season = int(season_dir.name)
            except ValueError:
                continue
            season_winners = []
            season_losers = []
            season_league_id = None
            for bracket_file in season_dir.glob("*_playoffBracket*.json"):
                if bracket_file.name.endswith("_playoffBrackets.json"):
                    continue
                league_id = bracket_file.name.split("_playoffBracket")[0]
                if season_league_id is None:
                    season_league_id = league_id
                bracket_id = None
                bracket_label = None
                if "_playoffBracket_" in bracket_file.name:
                    bracket_id = bracket_file.name.split("_playoffBracket_")[1].split(".json")[0]

                bracket_list_file = season_dir / f"{league_id}_playoffBrackets.json"
                if bracket_list_file.exists() and bracket_id:
                    bracket_list = json.loads(bracket_list_file.read_text())
                    bracket_container = bracket_list.get("playoffBrackets") or bracket_list.get("playoffbrackets")
                    brackets = []
                    if isinstance(bracket_container, dict):
                        brackets = bracket_container.get("bracket") or bracket_container.get("playoffBracket")
                        if isinstance(brackets, dict):
                            brackets = [brackets]
                    if isinstance(brackets, list):
                        for bracket in brackets:
                            if str(bracket.get("id")) == str(bracket_id):
                                bracket_label = (
                                    bracket.get("bracketWinnerTitle")
                                    or bracket.get("name")
                                    or bracket.get("bracketType")
                                )
                                break

                data = json.loads(bracket_file.read_text())
                matchups = _extract_mfl_matchups(data, bracket_label=bracket_label)
                if not matchups:
                    continue

                label = (bracket_label or "").lower()
                if "3rd" in label:
                    continue

                winners_matchups = [m for m in matchups if not m.get("is_losers")]
                losers_matchups = [m for m in matchups if m.get("is_losers")]

                season_winners.extend(winners_matchups)
                season_losers.extend(losers_matchups)

            def pick_winner(matchup_list: list) -> Optional[str]:
                if not matchup_list:
                    return None
                rounds = [m.get("round") for m in matchup_list if isinstance(m.get("round"), int)]
                max_round = max(rounds) if rounds else None
                candidates = matchup_list
                if max_round is not None:
                    candidates = [m for m in matchup_list if m.get("round") == max_round]
                matchup = candidates[-1]
                team_a = matchup["team_a"]
                team_b = matchup["team_b"]
                return _winner_from_scores(
                    team_a["score"],
                    team_b["score"],
                    team_a["id"],
                    team_b["id"]
                )

            if not season_league_id:
                continue

            champ_team_id = pick_winner(season_winners)
            losers_team_id = pick_winner(season_losers)

            champ_meta = _lookup_team_meta(team_meta, "mfl", season, season_league_id, champ_team_id)
            losers_meta = _lookup_team_meta(team_meta, "mfl", season, season_league_id, losers_team_id)

            if champ_meta or losers_meta:
                titles.append({
                    "season": season,
                    "platform": "mfl",
                    "platform_league_id": season_league_id,
                    "champion_owner_id": champ_meta["owner_id"] if champ_meta else None,
                    "champion_team_name": champ_meta["team_name"] if champ_meta else None,
                    "losers_champ_owner_id": losers_meta["owner_id"] if losers_meta else None,
                    "losers_champ_team_name": losers_meta["team_name"] if losers_meta else None
                })

    return pd.DataFrame(titles)


def build_mart_owner_achievements(mart_matchups: pd.DataFrame) -> pd.DataFrame:
    """
    Build mart_owner_achievements table with top-3 playoff finishes and points-for seasons.

    Args:
        mart_matchups: mart_matchups DataFrame

    Returns:
        mart_owner_achievements DataFrame
    """
    regular_games = mart_matchups[mart_matchups["is_playoffs"] == False]
    playoff_games = mart_matchups[mart_matchups["is_playoffs"] == True]

    regular_stats = _build_owner_season_stats_for_matchups(regular_games)
    playoff_stats = _build_owner_season_stats_for_matchups(playoff_games)

    playoff_stats = playoff_stats[playoff_stats["games"] > 0].copy()

    regular_tiebreakers = regular_stats[[
        "season",
        "owner_id",
        "win_pct",
        "wins",
        "points_for"
    ]].rename(columns={
        "win_pct": "regular_win_pct",
        "wins": "regular_wins",
        "points_for": "regular_points_for"
    })

    playoff_top3_by_owner: dict[str, list[int]] = {}
    pf_top3_by_owner: dict[str, list[int]] = {}

    for season in sorted(mart_matchups["season"].unique()):
        season_playoffs = playoff_stats[playoff_stats["season"] == season]
        if not season_playoffs.empty:
            season_playoffs = season_playoffs.merge(
                regular_tiebreakers,
                on=["season", "owner_id"],
                how="left"
            )
            season_playoffs["regular_win_pct"] = season_playoffs["regular_win_pct"].fillna(0.0)
            season_playoffs["regular_wins"] = season_playoffs["regular_wins"].fillna(0)
            season_playoffs["regular_points_for"] = season_playoffs["regular_points_for"].fillna(0.0)

            top_playoff = season_playoffs.sort_values(
                by=["wins", "points_for", "regular_win_pct", "regular_points_for", "owner_id"],
                ascending=[False, False, False, False, True]
            ).head(3)

            for owner_id in top_playoff["owner_id"].tolist():
                playoff_top3_by_owner.setdefault(owner_id, []).append(season)

        season_regular = regular_stats[regular_stats["season"] == season]
        if not season_regular.empty:
            top_pf = season_regular.sort_values(
                by=["points_for", "wins", "owner_id"],
                ascending=[False, False, True]
            ).head(3)

            for owner_id in top_pf["owner_id"].tolist():
                pf_top3_by_owner.setdefault(owner_id, []).append(season)

    all_owners = sorted(set(mart_matchups["owner_id_home"]) | set(mart_matchups["owner_id_away"]))
    achievement_rows = []
    for owner_id in all_owners:
        playoff_seasons = sorted(playoff_top3_by_owner.get(owner_id, []))
        pf_seasons = sorted(pf_top3_by_owner.get(owner_id, []))

        achievement_rows.append({
            "owner_id": owner_id,
            "top3_playoff_finishes": len(playoff_seasons),
            "top3_playoff_seasons": ",".join(str(season) for season in playoff_seasons),
            "top3_pf_seasons": len(pf_seasons),
            "top3_pf_seasons_list": ",".join(str(season) for season in pf_seasons)
        })

    return pd.DataFrame(achievement_rows)


def validate_data_quality(
    mart_matchups: pd.DataFrame,
    mart_owner_season: pd.DataFrame
) -> None:
    """
    Run data quality checks and print results.

    Args:
        mart_matchups: mart_matchups DataFrame
        mart_owner_season: mart_owner_season DataFrame
    """
    print("\n=== Data Quality Checks ===")

    # Check for duplicate game IDs
    duplicate_games = mart_matchups["game_id"].duplicated().sum()
    if duplicate_games > 0:
        print(f"⚠ WARNING: {duplicate_games} duplicate game IDs found")
    else:
        print("✓ No duplicate game IDs")

    # Check W/L symmetry
    total_games = len(mart_matchups)
    total_wins = mart_owner_season["wins"].sum()
    total_losses = mart_owner_season["losses"].sum()
    total_ties = mart_owner_season["ties"].sum()

    expected_wins_losses = (total_games - total_ties) * 2
    actual_wins_losses = total_wins + total_losses

    if actual_wins_losses == expected_wins_losses:
        print(f"✓ W/L symmetry check passed ({total_wins}W + {total_losses}L = {expected_wins_losses})")
    else:
        print(f"⚠ WARNING: W/L mismatch (expected {expected_wins_losses}, got {actual_wins_losses})")

    # Check for unmapped owners
    all_owners = set(mart_matchups["owner_id_home"]) | set(mart_matchups["owner_id_away"])
    unmapped = [o for o in all_owners if o.startswith("UNMAPPED_")]
    if unmapped:
        print(f"⚠ WARNING: {len(unmapped)} unmapped owners found:")
        for owner in unmapped:
            print(f"  - {owner}")
    else:
        print("✓ All owners mapped")

    # PF/PA symmetry check per season
    for season in mart_matchups["season"].unique():
        season_matchups = mart_matchups[mart_matchups["season"] == season]
        total_pf = season_matchups["score_home"].sum() + season_matchups["score_away"].sum()
        season_stats = mart_owner_season[mart_owner_season["season"] == season]
        total_pf_owners = season_stats["points_for"].sum()
        total_pa_owners = season_stats["points_against"].sum()

        if abs(total_pf - total_pf_owners) < 0.01 and abs(total_pf - total_pa_owners) < 0.01:
            print(f"✓ Season {season}: PF/PA symmetry check passed")
        else:
            print(f"⚠ WARNING: Season {season} PF/PA mismatch")

    print("===========================\n")


def build_all_marts() -> None:
    """Build all mart tables and run data quality checks."""
    print("Building mart tables...")

    # Build marts
    mart_matchups = build_mart_matchups()
    mart_owner_season = build_mart_owner_season(mart_matchups)
    mart_owner_all_time = build_mart_owner_all_time(mart_owner_season)
    mart_h2h = build_mart_h2h(mart_matchups)
    mart_titles = build_mart_titles(mart_matchups)
    mart_owner_achievements = build_mart_owner_achievements(mart_matchups)

    # Validate
    validate_data_quality(mart_matchups, mart_owner_season)

    # Write to CSV
    mart_dir = Path("data/mart")
    mart_dir.mkdir(parents=True, exist_ok=True)

    mart_matchups.to_csv(mart_dir / "mart_matchups.csv", index=False)
    mart_owner_season.to_csv(mart_dir / "mart_owner_season.csv", index=False)
    mart_owner_all_time.to_csv(mart_dir / "mart_owner_all_time.csv", index=False)
    mart_h2h.to_csv(mart_dir / "mart_h2h.csv", index=False)
    mart_owner_achievements.to_csv(mart_dir / "mart_owner_achievements.csv", index=False)
    mart_titles.to_csv(mart_dir / "mart_titles.csv", index=False)

    print(f"✓ Built {len(mart_matchups)} matchups")
    print(f"✓ Built {len(mart_owner_season)} owner-season records")
    print(f"✓ Built {len(mart_owner_all_time)} all-time owner records")
    print(f"✓ Built {len(mart_h2h)} head-to-head records")
    print(f"✓ Built {len(mart_owner_achievements)} owner achievement records")
    print(f"✓ Built {len(mart_titles)} title records")
