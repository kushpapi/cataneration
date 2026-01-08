"""MFL API ingestion module with corrected URL format, host support, and rate limiting."""

import os
import json
import time
import requests
from pathlib import Path
from typing import Optional, Dict
from dotenv import load_dotenv

load_dotenv()

# Known hosts for WCHB leagues (discovered from API info pages)
KNOWN_HOSTS: Dict[int, str] = {
    # 2016-2019 all on www44 (league 19829)
    2019: "www44",
    2018: "www44",
    2017: "www44",
    2016: "www44",
    # Earlier seasons may be on different hosts - discover via API
}

# Fallback to API gateway if host unknown
DEFAULT_HOST = "api"

# Rate limiting: MFL docs say to space requests 1 second apart
REQUEST_DELAY_SECONDS = 1.0


class MFLClient:
    def __init__(
        self,
        season: int,
        league_id: str,
        host: Optional[str] = None,
        api_key: Optional[str] = None
    ):
        self.season = season
        self.league_id = league_id
        self.host = host or KNOWN_HOSTS.get(season, DEFAULT_HOST)
        self.api_key = api_key or os.getenv("MFL_APIKEY")
        self.session = requests.Session()
        # Set User-Agent for better rate limits (per MFL docs)
        self.session.headers.update({
            "User-Agent": "WCHB-Fantasy-History/1.0"
        })
        self._last_request_time = 0

    def _get_base_url(self) -> str:
        """Get the base URL for this league's host."""
        if self.host == "api":
            return "https://api.myfantasyleague.com"
        return f"https://{self.host}.myfantasyleague.com"

    def _build_url(self, export_type: str, **extra_params) -> str:
        """Build properly formatted MFL API URL."""
        base = self._get_base_url()
        url = f"{base}/{self.season}/export?TYPE={export_type}&L={self.league_id}&JSON=1"

        for key, value in extra_params.items():
            url += f"&{key}={value}"

        # Add API key for authenticated access (works for export only)
        if self.api_key:
            url += f"&APIKEY={self.api_key}"

        return url

    def _rate_limit(self):
        """Enforce rate limiting between requests."""
        elapsed = time.time() - self._last_request_time
        if elapsed < REQUEST_DELAY_SECONDS:
            sleep_time = REQUEST_DELAY_SECONDS - elapsed
            print(f"  [RATE LIMIT] Sleeping {sleep_time:.2f}s...")
            time.sleep(sleep_time)
        self._last_request_time = time.time()

    def fetch_league(self) -> dict:
        """Fetch league configuration (includes franchise list)."""
        url = self._build_url("league")
        return self._fetch(url, "league")

    def fetch_schedule(self) -> dict:
        """Fetch full season schedule with matchup results."""
        url = self._build_url("schedule")
        return self._fetch(url, "schedule")

    def fetch_weekly_results(self, week: str = "YTD") -> dict:
        """Fetch weekly matchup results.

        Args:
            week: Week number (1-17) or "YTD" for all weeks
        """
        url = self._build_url("weeklyResults", W=week)
        return self._fetch(url, f"weeklyResults_W{week}")

    def fetch_standings(self) -> dict:
        """Fetch league standings (W/L records)."""
        url = self._build_url("leagueStandings")
        return self._fetch(url, "standings")

    def fetch_playoff_brackets(self) -> dict:
        """Fetch playoff bracket list metadata."""
        url = self._build_url("playoffBrackets")
        return self._fetch(url, "playoffBrackets")

    def fetch_playoff_bracket(self, bracket_id: str) -> dict:
        """Fetch a specific playoff bracket by ID."""
        param_options = ["BRACKET_ID", "BRACKETID", "BRACKET"]
        last_error = None
        for param in param_options:
            url = self._build_url("playoffBracket", **{param: bracket_id})
            try:
                return self._fetch(url, f"playoffBracket_{bracket_id}")
            except ValueError as e:
                if "playoff bracket id" in str(e).lower():
                    last_error = e
                    continue
                raise
        if last_error:
            raise last_error
        raise ValueError("Unable to fetch playoff bracket")

    def _fetch(self, url: str, cache_name: str) -> dict:
        """Fetch data from URL with caching and rate limiting."""
        cache_dir = Path(f"data/raw/mfl/{self.season}")
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / f"{self.league_id}_{cache_name}.json"

        # Return cached if exists (MFL recommends caching)
        if cache_file.exists():
            print(f"  [CACHE HIT] {cache_file}")
            with open(cache_file) as f:
                return json.load(f)

        # Rate limit before making request
        self._rate_limit()

        # Build URL for logging (hide API key)
        log_url = url.split("&APIKEY=")[0] + ("&APIKEY=***" if self.api_key else "")
        print(f"  [FETCHING] {log_url}")

        try:
            response = self.session.get(url, timeout=30)

            # Check for rate limiting (HTTP 429)
            if response.status_code == 429:
                print(f"  [RATE LIMITED] Too many requests. Waiting 5s...")
                time.sleep(5)
                return self._fetch(url, cache_name)  # Retry

            # Check for redirect (MFL may redirect to correct host)
            if response.history:
                print(f"  [REDIRECTED] Final URL: {response.url.split('&APIKEY=')[0]}")

            response.raise_for_status()
            data = response.json()

            # Check for API error in response
            if "error" in data:
                error_msg = data['error']
                # If API key validation failed, retry without API key
                if isinstance(error_msg, dict) and error_msg.get('$t') == 'API Key Validation Failed':
                    if self.api_key:
                        print(f"  [WARN] API key invalid for this season, retrying without API key...")
                        # Temporarily remove API key and retry
                        old_key = self.api_key
                        self.api_key = None
                        result = self._fetch(url.replace(f"&APIKEY={old_key}", ""), cache_name)
                        self.api_key = old_key  # Restore for other requests
                        return result
                print(f"  [API ERROR] {error_msg}")
                raise ValueError(f"MFL API Error: {error_msg}")

            # Cache the response
            with open(cache_file, 'w') as f:
                json.dump(data, f, indent=2)

            print(f"  [CACHED] {cache_file}")
            return data

        except requests.exceptions.HTTPError as e:
            if response.status_code == 404:
                print(f"  [ERROR] 404 Not Found")
                print(f"  [TIP] Check league ID and host. Current: {self.host}")
            elif response.status_code == 403:
                print(f"  [ERROR] 403 Forbidden - Check APIKEY")
            raise
        except requests.exceptions.SSLError:
            print(f"  [WARN] SSL error, retrying without verification...")
            self.session.verify = False
            return self._fetch(url, cache_name)


def discover_host(season: int, league_id: str) -> Optional[str]:
    """Try to discover which host a league is on.

    MFL's API gateway will redirect to the correct host.
    """
    url = f"https://api.myfantasyleague.com/{season}/export?TYPE=league&L={league_id}&JSON=1"
    try:
        response = requests.get(url, timeout=10, allow_redirects=True)
        # Extract host from final URL
        if "myfantasyleague.com" in response.url:
            host = response.url.split("//")[1].split(".")[0]
            print(f"  [DISCOVERED] League {league_id} ({season}) is on host: {host}")
            return host
    except Exception as e:
        print(f"  [WARN] Could not discover host: {e}")
    return None


def extract_bracket_ids(playoff_list: dict) -> list[str]:
    """Extract playoff bracket IDs from MFL playoffBrackets response."""
    if not playoff_list:
        return []

    bracket_container = playoff_list.get("playoffBrackets") or playoff_list.get("playoffbrackets")
    if bracket_container and isinstance(bracket_container, dict):
        bracket_items = bracket_container.get("bracket") or bracket_container.get("playoffBracket")
        if isinstance(bracket_items, dict):
            bracket_items = [bracket_items]
        if isinstance(bracket_items, list):
            ids = []
            for bracket in bracket_items:
                if isinstance(bracket, dict) and "id" in bracket:
                    ids.append(str(bracket["id"]))
            if ids:
                return ids

    # Fallback: search for dicts with id fields
    ids = []

    def walk(node):
        if isinstance(node, dict):
            if "id" in node and isinstance(node["id"], (str, int)):
                ids.append(str(node["id"]))
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(playoff_list)
    return sorted(set(ids))


def ingest_mfl_season(season: int, league_id: str, host: Optional[str] = None) -> bool:
    """Ingest all data for a single MFL season.

    Args:
        season: Year (e.g., 2019)
        league_id: MFL league ID (e.g., "19829")
        host: Optional host override (e.g., "www44")

    Returns:
        True if successful, False otherwise.
    """
    print(f"\n{'='*60}")
    print(f"Ingesting MFL {season} (League: {league_id})")
    print(f"{'='*60}")

    # Try to discover host if not provided
    if not host and season not in KNOWN_HOSTS:
        host = discover_host(season, league_id)

    client = MFLClient(season, league_id, host=host)
    print(f"Using host: {client.host}")
    print(f"API key: {'configured' if client.api_key else 'not set'}")

    try:
        # Fetch league info (includes franchises)
        print("\n[1/5] Fetching league info...")
        league_data = client.fetch_league()

        # Fetch schedule (includes matchup results)
        print("\n[2/5] Fetching schedule...")
        schedule_data = client.fetch_schedule()

        # Fetch weekly results
        print("\n[3/5] Fetching weekly results...")
        results_data = client.fetch_weekly_results("YTD")

        # Fetch standings
        print("\n[4/5] Fetching standings...")
        standings_data = client.fetch_standings()

        # Fetch playoff brackets (for titles)
        print("\n[5/5] Fetching playoff brackets...")
        bracket_ids = []
        try:
            playoff_list = client.fetch_playoff_brackets()
            bracket_ids = extract_bracket_ids(playoff_list)
        except Exception as e:
            print(f"  [WARN] Could not fetch playoff bracket list: {e}")

        if not bracket_ids:
            bracket_ids = ["1"]

        for bracket_id in bracket_ids:
            try:
                client.fetch_playoff_bracket(str(bracket_id))
            except Exception as e:
                print(f"  [WARN] Could not fetch playoff bracket {bracket_id}: {e}")

        print(f"\n✓ MFL {season} ingestion complete!")
        return True

    except Exception as e:
        print(f"\n✗ MFL {season} ingestion failed: {e}")
        return False


# Configuration for all WCHB MFL seasons
WCHB_MFL_LEAGUES = {
    # season: (league_id, host)
    2019: ("19829", "www44"),
    2018: ("19829", "www44"),
    2017: ("19829", "www44"),
    2016: ("19829", "www44"),
    2015: ("43804", None),  # Host unknown - will discover
    2014: ("34454", None),
    2013: ("63294", None),
}


def ingest_all_mfl() -> Dict[int, bool]:
    """Ingest all WCHB MFL seasons."""
    results = {}
    for season, (league_id, host) in sorted(WCHB_MFL_LEAGUES.items(), reverse=True):
        results[season] = ingest_mfl_season(season, league_id, host)
    return results


if __name__ == "__main__":
    print("=" * 60)
    print("MFL INGESTION TEST")
    print("=" * 60)

    # Test 2019 first
    success = ingest_mfl_season(2019, "19829", "www44")

    if success:
        print("\n✓ 2019 test passed! Ready to ingest all seasons.")
    else:
        print("\n✗ 2019 test failed. Check the errors above.")
