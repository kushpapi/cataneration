"""Fleaflicker API ingestion module with caching."""

import json
import time
import requests
from pathlib import Path
from typing import Optional, Dict

# Fleaflicker API is public - no auth required
BASE_URL = "https://www.fleaflicker.com/api"

# Rate limiting: be respectful to Fleaflicker's servers
REQUEST_DELAY_SECONDS = 0.5


class FleaflickerClient:
    def __init__(self, season: int, league_id: str):
        self.season = season
        self.league_id = league_id
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "WCHB-Fantasy-History/1.0"
        })
        self._last_request_time = 0

    def _rate_limit(self):
        """Enforce rate limiting between requests."""
        elapsed = time.time() - self._last_request_time
        if elapsed < REQUEST_DELAY_SECONDS:
            sleep_time = REQUEST_DELAY_SECONDS - elapsed
            time.sleep(sleep_time)
        self._last_request_time = time.time()

    def fetch_standings(self) -> dict:
        """Fetch league standings (includes team info and W/L records)."""
        url = f"{BASE_URL}/FetchLeagueStandings"
        params = {
            "league_id": self.league_id,
            "season": self.season
        }
        return self._fetch(url, params, "standings")

    def fetch_scoreboard(self, week: int) -> dict:
        """Fetch scoreboard for a specific week.

        Args:
            week: Week number (1-17 for regular season + playoffs)
        """
        url = f"{BASE_URL}/FetchLeagueScoreboard"
        params = {
            "league_id": self.league_id,
            "season": self.season,
            "scoring_period": week
        }
        return self._fetch(url, params, f"scoreboard_week{week:02d}")

    def _fetch(self, url: str, params: dict, cache_name: str) -> dict:
        """Fetch data from URL with caching and rate limiting."""
        cache_dir = Path(f"data/raw/fleaflicker/{self.season}")
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / f"{self.league_id}_{cache_name}.json"

        # Return cached if exists
        if cache_file.exists():
            print(f"  [CACHE HIT] {cache_file}")
            with open(cache_file) as f:
                return json.load(f)

        # Rate limit before making request
        self._rate_limit()

        print(f"  [FETCHING] {url}?{self._format_params(params)}")

        try:
            response = self.session.get(url, params=params, timeout=30)

            # Check for rate limiting
            if response.status_code == 429:
                print(f"  [RATE LIMITED] Too many requests. Waiting 5s...")
                time.sleep(5)
                return self._fetch(url, params, cache_name)  # Retry

            response.raise_for_status()
            data = response.json()

            # Cache the response
            with open(cache_file, 'w') as f:
                json.dump(data, f, indent=2)

            print(f"  [CACHED] {cache_file}")
            return data

        except requests.exceptions.HTTPError as e:
            if response.status_code == 404:
                print(f"  [ERROR] 404 Not Found - Check league ID and season")
            elif response.status_code == 403:
                print(f"  [ERROR] 403 Forbidden")
            raise
        except requests.exceptions.JSONDecodeError:
            print(f"  [ERROR] Invalid JSON response")
            print(f"  Response text: {response.text[:200]}")
            raise

    def _format_params(self, params: dict) -> str:
        """Format params for logging."""
        return "&".join(f"{k}={v}" for k, v in params.items())


def ingest_fleaflicker_season(season: int, league_id: str, skip_matchups: bool = False) -> bool:
    """Ingest all data for a single Fleaflicker season.

    Args:
        season: Year (e.g., 2020)
        league_id: Fleaflicker league ID (e.g., "311194")
        skip_matchups: If True, only fetch standings (for draft-only seasons like 2025)

    Returns:
        True if successful, False otherwise.
    """
    print(f"\n{'='*60}")
    print(f"Ingesting Fleaflicker {season} (League: {league_id})")
    if skip_matchups:
        print("(Draft-only season - skipping matchups)")
    print(f"{'='*60}")

    client = FleaflickerClient(season, league_id)

    try:
        # Fetch standings (includes teams and records)
        print("\n[1/?] Fetching standings...")
        standings_data = client.fetch_standings()

        if skip_matchups:
            print(f"\n✓ Fleaflicker {season} ingestion complete (standings only)!")
            return True

        # Determine number of weeks from standings data
        # Fleaflicker typically has 14 weeks regular season + playoffs
        # We'll try fetching weeks 1-17 and stop when we get empty data
        print("\n[2/?] Fetching scoreboards...")

        weeks_fetched = 0
        for week in range(1, 18):  # Try up to week 17
            try:
                print(f"\n  Week {week}...")
                scoreboard = client.fetch_scoreboard(week)

                # Check if week has data
                if not scoreboard or not scoreboard.get("games"):
                    print(f"  [INFO] No games found for week {week}, stopping")
                    break

                weeks_fetched += 1

            except Exception as e:
                print(f"  [WARN] Could not fetch week {week}: {e}")
                # If we've already fetched some weeks, this might be the end of season
                if weeks_fetched > 0:
                    break
                # If we haven't fetched any weeks yet, this is a real error
                raise

        print(f"\n✓ Fleaflicker {season} ingestion complete! ({weeks_fetched} weeks)")
        return True

    except Exception as e:
        print(f"\n✗ Fleaflicker {season} ingestion failed: {e}")
        import traceback
        traceback.print_exc()
        return False


# Configuration for all WCHB Fleaflicker seasons
WCHB_FLEAFLICKER_LEAGUES = {
    # season: (league_id, skip_matchups)
    2024: ("311194", False),
    2023: ("311194", False),
    2022: ("311194", False),
    2021: ("311194", False),
    2020: ("311194", False),
    2025: ("311194", True),  # Draft-only season
}


def ingest_all_fleaflicker() -> Dict[int, bool]:
    """Ingest all WCHB Fleaflicker seasons."""
    results = {}
    # Process in reverse chronological order (most recent first)
    for season, (league_id, skip_matchups) in sorted(WCHB_FLEAFLICKER_LEAGUES.items(), reverse=True):
        if season == 2025:
            print(f"\n[INFO] Skipping Fleaflicker 2025 (draft-only season)")
            continue
        results[season] = ingest_fleaflicker_season(season, league_id, skip_matchups)
    return results


if __name__ == "__main__":
    print("=" * 60)
    print("FLEAFLICKER INGESTION TEST")
    print("=" * 60)

    # Test 2020 first
    success = ingest_fleaflicker_season(2020, "311194")

    if success:
        print("\n✓ 2020 test passed! Ready to ingest all seasons.")
    else:
        print("\n✗ 2020 test failed. Check the errors above.")
