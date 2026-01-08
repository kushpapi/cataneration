"""HTTP client with file-based caching for API responses."""

import json
import os
from pathlib import Path
from typing import Any, Optional
import requests
from dotenv import load_dotenv

load_dotenv()


class CachedHTTPClient:
    """HTTP client that caches responses to disk."""

    def __init__(self, cache_dir: str = "data/raw", verify_ssl: bool = True):
        """
        Initialize the cached HTTP client.

        Args:
            cache_dir: Base directory for cached responses
            verify_ssl: Whether to verify SSL certificates (default True)
        """
        self.cache_dir = Path(cache_dir)
        self.session = requests.Session()
        self.verify_ssl = verify_ssl

    def get_json(
        self,
        url: str,
        cache_path: str,
        params: Optional[dict] = None,
        force: bool = False
    ) -> dict[str, Any]:
        """
        Fetch JSON data from URL with file-based caching.

        Args:
            url: URL to fetch
            cache_path: Relative path within cache_dir to store response
            params: Optional query parameters
            force: If True, bypass cache and refetch

        Returns:
            Parsed JSON response as dict

        Raises:
            requests.HTTPError: If request fails
        """
        full_cache_path = self.cache_dir / cache_path

        # Return cached response if exists and not forcing refresh
        if not force and full_cache_path.exists():
            with open(full_cache_path, 'r') as f:
                return json.load(f)

        # Fetch from API
        response = self.session.get(url, params=params, verify=self.verify_ssl)
        response.raise_for_status()
        data = response.json()

        # Cache the response
        full_cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(full_cache_path, 'w') as f:
            json.dump(data, f, indent=2)

        return data

    def add_header(self, key: str, value: str) -> None:
        """Add a header to all requests."""
        self.session.headers[key] = value
