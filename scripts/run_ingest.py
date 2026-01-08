#!/usr/bin/env python3
"""Ingest script for fetching data from fantasy platforms."""

import argparse
import yaml
from pathlib import Path

from src.ingest.mfl import ingest_mfl_season
from src.ingest.fleaflicker import ingest_fleaflicker_season
from src.ingest.sleeper import ingest_sleeper_season


def load_config() -> dict:
    """Load leagues configuration."""
    with open("config/leagues.yaml") as f:
        return yaml.safe_load(f)


def ingest_mfl(season: int, config: dict) -> None:
    """
    Ingest MFL data for a specific season.

    Args:
        season: Season year
        config: Full configuration dict
    """
    if season not in config["mfl"]["seasons"]:
        print(f"❌ Season {season} not found in MFL config")
        return

    season_config = config["mfl"]["seasons"][season]
    league_id = season_config["league_id"]
    host = season_config.get("host")  # May be None for discovery

    ingest_mfl_season(season=season, league_id=league_id, host=host)


def ingest_fleaflicker(season: int, config: dict) -> None:
    """
    Ingest Fleaflicker data for a specific season.

    Args:
        season: Season year
        config: Full configuration dict
    """
    if season not in config["fleaflicker"]["seasons"]:
        print(f"❌ Season {season} not found in Fleaflicker config")
        return

    season_config = config["fleaflicker"]["seasons"][season]
    league_id = config["fleaflicker"]["league_id"]
    skip_matchups = not season_config.get("has_matchups", True)

    ingest_fleaflicker_season(season=season, league_id=league_id, skip_matchups=skip_matchups)


def ingest_sleeper(season: int, config: dict) -> None:
    """
    Ingest Sleeper data for a specific season.

    Args:
        season: Season year
        config: Full configuration dict
    """
    if season not in config["sleeper"]["seasons"]:
        print(f"❌ Season {season} not found in Sleeper config")
        return

    season_config = config["sleeper"]["seasons"][season]
    league_id = season_config["league_id"]
    username = config["sleeper"]["username"]

    ingest_sleeper_season(season=season, league_id=league_id, username=username)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Ingest fantasy football data")
    parser.add_argument("--platform", required=True, choices=["mfl", "fleaflicker", "sleeper"],
                        help="Platform to ingest from")
    parser.add_argument("--season", type=int,
                        help="Season year to ingest")
    parser.add_argument("--all-seasons", action="store_true",
                        help="Ingest all available seasons for the platform")

    args = parser.parse_args()
    config = load_config()

    # Validate arguments
    if not args.all_seasons and not args.season:
        parser.error("Either --season or --all-seasons must be specified")

    if args.all_seasons:
        if args.platform == "mfl":
            seasons = config["mfl"]["seasons"].keys()
            for season in seasons:
                ingest_mfl(season, config)
        elif args.platform == "fleaflicker":
            seasons = [s for s in config["fleaflicker"]["seasons"].keys()
                      if config["fleaflicker"]["seasons"][s].get("has_matchups", True)]
            for season in seasons:
                ingest_fleaflicker(season, config)
        elif args.platform == "sleeper":
            seasons = config["sleeper"]["seasons"].keys()
            for season in seasons:
                ingest_sleeper(season, config)
        else:
            print(f"❌ --all-seasons not yet implemented for {args.platform}")
    else:
        if args.platform == "mfl":
            ingest_mfl(args.season, config)
        elif args.platform == "fleaflicker":
            ingest_fleaflicker(args.season, config)
        elif args.platform == "sleeper":
            ingest_sleeper(args.season, config)
        else:
            print(f"❌ {args.platform} ingestion not yet implemented")


if __name__ == "__main__":
    main()
