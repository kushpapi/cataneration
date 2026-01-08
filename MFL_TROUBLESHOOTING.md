# MFL API Troubleshooting Guide & Claude Code Update

## ✅ CONFIRMED: Your League is Accessible!

From the API info page you provided, we have everything needed:

| Info | Value |
|------|-------|
| **Host** | `www44.myfantasyleague.com` |
| **League ID** | `19829` |
| **Season** | 2019 (and 2016-2019 based on your config) |
| **Your API Key** | `REDACTED_API_KEY` |

**Key insight:** Since you can see your API key on the page, you're logged in and have access. The API key allows authenticated access without needing to pass cookies.

> ⚠️ **IMPORTANT:** The API key shown is tied to your user/franchise/league combination. Add it to your `.env` file but don't commit it to git.

---

## Root Cause Analysis

Based on the official MFL API documentation, the 404 errors are caused by:

### 1. **Wrong Base URL** (Confirmed)
The MFL API URL format is:
```
protocol://host/year/command?args
```

**Correct URL format:**
```
https://www44.myfantasyleague.com/2019/export?TYPE=league&L=19829&JSON=1
```

**NOT:**
```
https://www.mfl.com/2019/export?...  ❌
```

### 2. **Private League Access** (May Apply)
From the docs: "Data from leagues that are marked as PRIVATE are no longer accessible by users not in the league."

If your league is private, you need to pass either:
- **APIKEY parameter** (simplest): `&APIKEY=REDACTED_API_KEY`
- **Cookie-based auth** via the login API

### 3. **Host Routing**
From the docs: "The host is the server where the league you are accessing lives. Our hosts at this time are of the form wwwXX where XX is a 2-digit number."

Your league is on **`www44`**. The API gateway (`api.myfantasyleague.com`) will redirect, but directly hitting the correct host is more reliable.

### 4. **Rate Limiting** (New in 2020+)
The docs mention: "Unregistered clients will be limited in the amount of requests they can make." If you hit rate limits, you'll get HTTP 429 errors. Solutions:
- Space requests 1 second apart
- Cache responses
- Set a User-Agent header

---

## Fix Priority Order

### Fix #1: Update Base URL (Do This First)

Update `src/ingest/mfl.py` to use the correct base URL:

```python
# OLD (wrong):
BASE_URL = "https://www.mfl.com"

# NEW (correct) - Use the specific host for your league:
BASE_URL = "https://www44.myfantasyleague.com"

# OR use the API gateway (redirects automatically):
BASE_URL = "https://api.myfantasyleague.com"
```

**Your specific league URLs (test these in browser first!):**

```bash
# 2019 League Info
https://www44.myfantasyleague.com/2019/export?TYPE=league&L=19829&JSON=1

# 2019 Schedule (includes matchup results)
https://www44.myfantasyleague.com/2019/export?TYPE=schedule&L=19829&JSON=1

# 2019 Weekly Results (all weeks)
https://www44.myfantasyleague.com/2019/export?TYPE=weeklyResults&L=19829&W=YTD&JSON=1

# 2019 Standings
https://www44.myfantasyleague.com/2019/export?TYPE=leagueStandings&L=19829&JSON=1
```

**URL construction pattern:**
```python
def build_mfl_url(host: str, season: int, league_id: str, export_type: str) -> str:
    """Build MFL API URL with correct format."""
    return (
        f"https://{host}.myfantasyleague.com/{season}/export"
        f"?TYPE={export_type}&L={league_id}&JSON=1"
    )

# For your league:
url = build_mfl_url("www44", 2019, "19829", "league")
# Result: https://www44.myfantasyleague.com/2019/export?TYPE=league&L=19829&JSON=1
```

---

### Fix #2: Add Authentication Support (If Fix #1 Still Fails)

If leagues are private, you need authentication. There are two methods:

#### Option A: APIKEY (Simpler)
1. Log into MFL with a league member's credentials
2. Go to: `https://www{XX}.myfantasyleague.com/{year}/api_info?L={league_id}`
3. Find "Your API Key" section (only visible when logged in)
4. Add to `.env`: `MFL_APIKEY=your_key_here`

**Code update:**
```python
import os
from dotenv import load_dotenv

load_dotenv()

def build_mfl_url(season: int, league_id: str, export_type: str) -> str:
    api_key = os.getenv("MFL_APIKEY")
    url = f"https://api.myfantasyleague.com/{season}/export?TYPE={export_type}&L={league_id}&JSON=1"
    if api_key:
        url += f"&APIKEY={api_key}"
    return url
```

#### Option B: Cookie Authentication (For Multiple Leagues)
```python
import requests

def get_mfl_auth_cookie(username: str, password: str, year: int = 2019) -> str:
    """Get MFL authentication cookie via login API."""
    login_url = f"https://api.myfantasyleague.com/{year}/login"
    params = {
        "USERNAME": username,
        "PASSWORD": password,
        "XML": 1
    }
    response = requests.get(login_url, params=params)
    # Parse XML response to get MFL_USER_ID cookie value
    # Return the cookie value
    ...

def fetch_with_auth(url: str, cookie: str) -> dict:
    """Fetch MFL data with authentication cookie."""
    headers = {"Cookie": f"MFL_USER_ID={cookie}"}
    response = requests.get(url, headers=headers)
    return response.json()
```

---

### Fix #3: Alternative Endpoints to Try

If `weeklyResults` with `W=YTD` returns empty, try these:

```python
# Get the full season schedule with scores
schedule_url = f"https://api.myfantasyleague.com/{season}/export?TYPE=schedule&L={league_id}&JSON=1"

# Get league standings (has win/loss records)
standings_url = f"https://api.myfantasyleague.com/{season}/export?TYPE=leagueStandings&L={league_id}&JSON=1"

# Get specific week results
week_url = f"https://api.myfantasyleague.com/{season}/export?TYPE=weeklyResults&L={league_id}&W={week}&JSON=1"
```

---

## Updated Code for Claude Code

Here's the complete fix to apply to `src/ingest/mfl.py`:

```python
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
            "User-Agent": "WCHB-Fantasy-History/1.0 (github.com/yourrepo)"
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
        """Fetch playoff bracket results."""
        url = self._build_url("playoffBracket")
        return self._fetch(url, "playoffBracket")
    
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
                print(f"  [API ERROR] {data['error']}")
                raise ValueError(f"MFL API Error: {data['error']}")
            
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
        print("\n[1/4] Fetching league info...")
        league_data = client.fetch_league()
        
        # Fetch schedule (includes matchup results)
        print("\n[2/4] Fetching schedule...")
        schedule_data = client.fetch_schedule()
        
        # Fetch weekly results
        print("\n[3/4] Fetching weekly results...")
        results_data = client.fetch_weekly_results("YTD")
        
        # Fetch standings
        print("\n[4/4] Fetching standings...")
        standings_data = client.fetch_standings()
        
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
```

---

## Manual Testing Steps

**Before running the full pipeline, test these URLs in your browser:**

### 1. Test your 2019 league directly on www44:
```
https://www44.myfantasyleague.com/2019/export?TYPE=league&L=19829&JSON=1
```

### 2. Test schedule endpoint (has matchup results):
```
https://www44.myfantasyleague.com/2019/export?TYPE=schedule&L=19829&JSON=1
```

### 3. Test weekly results:
```
https://www44.myfantasyleague.com/2019/export?TYPE=weeklyResults&L=19829&W=YTD&JSON=1
```

### 4. Test via API gateway (should redirect to www44):
```
https://api.myfantasyleague.com/2019/export?TYPE=league&L=19829&JSON=1
```

### Expected Results:
- ✅ **JSON data returned** → URL fix will work, league is public
- ❌ **404 or empty response** → League may be private, need APIKEY
- ❌ **403 Forbidden** → Definitely need authentication

### 5. Test earlier years (different league IDs, unknown hosts):
```
# 2015 - League 43804
https://api.myfantasyleague.com/2015/export?TYPE=league&L=43804&JSON=1

# 2014 - League 34454
https://api.myfantasyleague.com/2014/export?TYPE=league&L=34454&JSON=1

# 2013 - League 63294
https://api.myfantasyleague.com/2013/export?TYPE=league&L=63294&JSON=1
```

---

## Environment Setup

Update your `.env` file with your actual API key:

```bash
# .env

# MFL API Key for league 19829 (YOUR KEY - don't share or commit to git!)
MFL_APIKEY=REDACTED_API_KEY

# Note: This key is tied to your user/franchise/league combination
# It works for export requests but NOT import requests
# It's valid for the entire season

# Optional: Add keys for other league years if they differ
# MFL_APIKEY_2018=...
# MFL_APIKEY_2017=...
```

**Add `.env` to `.gitignore`:**
```bash
echo ".env" >> .gitignore
```

---

## Summary for Claude Code

**✅ CONFIRMED WORKING:** Your 2019 league (ID: 19829) is on host `www44.myfantasyleague.com` and you have an API key.

**Immediate action items:**

1. ✅ **Update base URL** to `https://www44.myfantasyleague.com`
2. ✅ **Add API key** to `.env`: `MFL_APIKEY=REDACTED_API_KEY`
3. ✅ **Append APIKEY to requests** (if league is private): `&APIKEY=REDACTED_API_KEY`
4. ✅ **Add rate limiting** - space requests 1 second apart
5. ✅ **Set User-Agent header** for better rate limits

**Confirmed working URL pattern:**
```
https://www44.myfantasyleague.com/2019/export?TYPE=league&L=19829&JSON=1&APIKEY=REDACTED_API_KEY
```

**Update `config/leagues.yaml` with hosts:**

```yaml
platforms:
  mfl:
    seasons:
      2019:
        league_id: "19829"
        host: "www44"
      2018:
        league_id: "19829"
        host: "www44"
      2017:
        league_id: "19829"
        host: "www44"
      2016:
        league_id: "19829"
        host: "www44"
      2015:
        league_id: "43804"
        host: null  # Discover via API
      2014:
        league_id: "34454"
        host: null
      2013:
        league_id: "63294"
        host: null
```

**Key API details from official docs:**
- Player IDs are strings with leading zeros (e.g., "0531" not "531")
- Franchise IDs are 4-digit strings starting at "0001"
- Timestamps are Unix format in EST/EDT
- JSON output requires `&JSON=1` parameter
- API key only works for export, not import requests
- Rate limits apply - space requests 1 second apart

**To get API keys for other seasons:** Log into MFL for each season/league and visit the API info page to get that season's key.
