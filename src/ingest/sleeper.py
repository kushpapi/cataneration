"""Sleeper API ingestion module with caching."""

import json
import time
import requests
from pathlib import Path
from typing import Optional, Dict

# Sleeper API is public - no auth required
BASE_URL = "https://api.sleeper.app/v1"

# Rate limiting: be respectful to Sleeper's servers
REQUEST_DELAY_SECONDS = 0.5


class SleeperClient:
    def __init__(self, season: int, league_id: str, username: str):
        self.season = season
        self.league_id = league_id
        self.username = username
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

    def fetch_user(self) -> dict:
        """Fetch user info for the league owner."""
        url = f"{BASE_URL}/user/{self.username}"
        return self._fetch(url, "user")

    def fetch_league(self) -> dict:
        """Fetch league info (settings, metadata)."""
        url = f"{BASE_URL}/league/{self.league_id}"
        return self._fetch(url, "league")

    def fetch_users(self) -> list:
        """Fetch all users (owners) in the league."""
        url = f"{BASE_URL}/league/{self.league_id}/users"
        return self._fetch(url, "users")

    def fetch_rosters(self) -> list:
        """Fetch all rosters (teams) in the league."""
        url = f"{BASE_URL}/league/{self.league_id}/rosters"
        return self._fetch(url, "rosters")

    def fetch_matchups(self, week: int) -> list:
        """Fetch matchups for a specific week.

        Args:
            week: Week number (1-18 for regular season + playoffs)
        """
        url = f"{BASE_URL}/league/{self.league_id}/matchups/{week}"
        return self._fetch(url, f"matchups_week{week:02d}")

    def fetch_winners_bracket(self) -> list:
        """Fetch playoff winners bracket."""
        url = f"{BASE_URL}/league/{self.league_id}/winners_bracket"
        return self._fetch(url, "winners_bracket")

    def fetch_losers_bracket(self) -> list:
        """Fetch playoff losers bracket (if configured)."""
        url = f"{BASE_URL}/league/{self.league_id}/losers_bracket"
        return self._fetch(url, "losers_bracket")

    def _fetch(self, url: str, cache_name: str) -> dict:
        """Fetch data from URL with caching and rate limiting."""
        cache_dir = Path(f"data/raw/sleeper/{self.season}")
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / f"{self.league_id}_{cache_name}.json"

        # Return cached if exists
        if cache_file.exists():
            print(f"  [CACHE HIT] {cache_file}")
            with open(cache_file) as f:
                return json.load(f)

        # Rate limit before making request
        self._rate_limit()

        print(f"  [FETCHING] {url}")

        try:
            response = self.session.get(url, timeout=30)

            # Check for rate limiting
            if response.status_code == 429:
                print(f"  [RATE LIMITED] Too many requests. Waiting 5s...")
                time.sleep(5)
                return self._fetch(url, cache_name)  # Retry

            response.raise_for_status()
            data = response.json()

            # Cache the response
            with open(cache_file, 'w') as f:
                json.dump(data, f, indent=2)

            print(f"  [CACHED] {cache_file}")
            return data

        except requests.exceptions.HTTPError as e:
            if response.status_code == 404:
                print(f"  [ERROR] 404 Not Found - Check league ID/username/week")
            elif response.status_code == 403:
                print(f"  [ERROR] 403 Forbidden")
            raise
        except requests.exceptions.JSONDecodeError:
            print(f"  [ERROR] Invalid JSON response")
            print(f"  Response text: {response.text[:200]}")
            raise


def ingest_sleeper_season(season: int, league_id: str, username: str) -> bool:
    """Ingest all data for a single Sleeper season.

    Args:
        season: Year (e.g., 2025)
        league_id: Sleeper league ID (e.g., "1269081387739136000")
        username: Sleeper username (e.g., "benmmccoy")

    Returns:
        True if successful, False otherwise.
    """
    print(f"\n{'='*60}")
    print(f"Ingesting Sleeper {season} (League: {league_id})")
    print(f"{'='*60}")

    client = SleeperClient(season, league_id, username)

    try:
        # Fetch user info
        print("\n[1/7] Fetching user info...")
        user_data = client.fetch_user()

        # Fetch league info
        print("\n[2/7] Fetching league info...")
        league_data = client.fetch_league()

        # Fetch users (all owners)
        print("\n[3/7] Fetching users...")
        users_data = client.fetch_users()

        # Fetch rosters (teams)
        print("\n[4/7] Fetching rosters...")
        rosters_data = client.fetch_rosters()

        # Fetch playoff brackets
        print("\n[5/7] Fetching winners bracket...")
        try:
            winners_bracket = client.fetch_winners_bracket()
        except Exception as e:
            print(f"  [WARN] Could not fetch winners bracket: {e}")
            winners_bracket = []

        print("\n[6/7] Fetching losers bracket...")
        try:
            losers_bracket = client.fetch_losers_bracket()
        except Exception as e:
            print(f"  [WARN] Could not fetch losers bracket: {e}")
            losers_bracket = []

        # Fetch matchups (weeks 1-18 or until empty)
        print("\n[7/7] Fetching matchups...")

        weeks_fetched = 0
        for week in range(1, 19):  # Try up to week 18
            try:
                print(f"\n  Week {week}...")
                matchups = client.fetch_matchups(week)

                # Check if week has data
                if not matchups or len(matchups) == 0:
                    print(f"  [INFO] No matchups found for week {week}, stopping")
                    break

                weeks_fetched += 1

            except Exception as e:
                print(f"  [WARN] Could not fetch week {week}: {e}")
                # If we've already fetched some weeks, this might be the end of season
                if weeks_fetched > 0:
                    break
                # If we haven't fetched any weeks yet, this is a real error
                raise

        print(f"\n✓ Sleeper {season} ingestion complete! ({weeks_fetched} weeks)")
        return True

    except Exception as e:
        print(f"\n✗ Sleeper {season} ingestion failed: {e}")
        import traceback
        traceback.print_exc()
        return False


# Configuration for all WCHB Sleeper seasons
WCHB_SLEEPER_LEAGUES = {
    # season: (league_id, username)
    2025: ("1269081387739136000", "benmmccoy"),
}


def ingest_all_sleeper() -> Dict[int, bool]:
    """Ingest all WCHB Sleeper seasons."""
    results = {}
    for season, (league_id, username) in sorted(WCHB_SLEEPER_LEAGUES.items(), reverse=True):
        results[season] = ingest_sleeper_season(season, league_id, username)
    return results


if __name__ == "__main__":
    print("=" * 60)
    print("SLEEPER INGESTION TEST")
    print("=" * 60)

    # Test 2025
    success = ingest_sleeper_season(2025, "1269081387739136000", "benmmccoy")

    if success:
        print("\n✓ 2025 test passed! Ready to ingest all seasons.")
    else:
        print("\n✗ 2025 test failed. Check the errors above.")
