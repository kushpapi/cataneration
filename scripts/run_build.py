#!/usr/bin/env python3
"""Build script for transforming staging data into marts."""

import argparse
import yaml
from pathlib import Path

from src.transform.normalize_mfl import normalize_mfl_season
from src.transform.normalize_fleaflicker import normalize_fleaflicker_season
from src.transform.normalize_sleeper import normalize_sleeper_season
from src.transform.seed_mappings import seed_all_mappings
from src.transform.build_marts import build_all_marts


def load_config() -> dict:
    """Load leagues configuration."""
    with open("config/leagues.yaml") as f:
        return yaml.safe_load(f)


def normalize_platform_season(platform: str, season: int, config: dict) -> None:
    """
    Normalize a single platform/season into staging tables.

    Args:
        platform: Platform name
        season: Season year
        config: Full configuration dict
    """
    if platform == "mfl":
        if season not in config["mfl"]["seasons"]:
            print(f"❌ Season {season} not found in MFL config")
            return

        season_config = config["mfl"]["seasons"][season]
        league_id = season_config["league_id"]

        print(f"Normalizing MFL {season}...")
        normalize_mfl_season(season, league_id)

    elif platform == "fleaflicker":
        if season not in config["fleaflicker"]["seasons"]:
            print(f"❌ Season {season} not found in Fleaflicker config")
            return

        league_id = config["fleaflicker"]["league_id"]

        print(f"Normalizing Fleaflicker {season}...")
        normalize_fleaflicker_season(season, league_id)

    elif platform == "sleeper":
        if season not in config["sleeper"]["seasons"]:
            print(f"❌ Season {season} not found in Sleeper config")
            return

        season_config = config["sleeper"]["seasons"][season]
        league_id = season_config["league_id"]

        print(f"Normalizing Sleeper {season}...")
        normalize_sleeper_season(season, league_id)

    else:
        print(f"❌ {platform} normalization not yet implemented")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Build fantasy football data marts")
    parser.add_argument("--platform", choices=["mfl", "fleaflicker", "sleeper"],
                        help="Platform to normalize (required unless --all or --seed)")
    parser.add_argument("--season", type=int,
                        help="Season to normalize (required unless --all or --seed)")
    parser.add_argument("--all-seasons", action="store_true",
                        help="Normalize all seasons for a platform")
    parser.add_argument("--seed", action="store_true",
                        help="Generate owner mapping todo files")
    parser.add_argument("--all", action="store_true",
                        help="Build all marts from all staging data")

    args = parser.parse_args()
    config = load_config()

    # Seed mappings
    if args.seed:
        seed_all_mappings()
        return

    # Build all marts
    if args.all:
        build_all_marts()
        return

    # Normalize specific platform/season
    if not args.platform or not args.season:
        if not args.all_seasons:
            parser.error("--platform and --season required unless using --all, --seed, or --all-seasons")

    if args.all_seasons:
        if not args.platform:
            parser.error("--platform required with --all-seasons")

        if args.platform == "mfl":
            seasons = config["mfl"]["seasons"].keys()
            for season in seasons:
                normalize_platform_season(args.platform, season, config)
        elif args.platform == "fleaflicker":
            seasons = [s for s in config["fleaflicker"]["seasons"].keys()
                      if config["fleaflicker"]["seasons"][s].get("has_matchups", True)]
            for season in seasons:
                normalize_platform_season(args.platform, season, config)
        elif args.platform == "sleeper":
            seasons = config["sleeper"]["seasons"].keys()
            for season in seasons:
                normalize_platform_season(args.platform, season, config)
        else:
            print(f"❌ --all-seasons not yet implemented for {args.platform}")
    else:
        normalize_platform_season(args.platform, args.season, config)


if __name__ == "__main__":
    main()
