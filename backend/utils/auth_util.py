import os
import threading
import hashlib
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List, Tuple
from cachetools import TTLCache
# from logger import debug_logger


class AuthManager:

    def __init__(self, ttl_seconds: int = 3600, use_dynamic_config: bool = True):
        """
        Initialize the AuthManager.

        Args:
            ttl_seconds: Cache TTL in seconds (default: 1 hour)
            use_dynamic_config: Enable dynamic config fetching (default: True)
        """
        # Configuration
        self.ttl_seconds = ttl_seconds
        self.use_dynamic_config = use_dynamic_config

        # Thread safety
        self.lock = threading.Lock()

        # Caching - TTLCache provides automatic expiration
        self.config_cache = TTLCache(maxsize=100, ttl=ttl_seconds)
        self.filter_attributes_cache = TTLCache(maxsize=100, ttl=ttl_seconds)
        self.token_cache = {}  # Store {access_token, id_token, expires_at}

        # Lazy-initialized clients
        self._http_client = None

        # Environment variables (OAuth2 credentials and API endpoints)
        self.internal_api_base_url = os.getenv("INTERNAL_API_BASE_URL")
        self.oauth2_username = os.getenv("INTERNAL_API_USERNAME")
        self.oauth2_password = os.getenv("INTERNAL_API_PASSWORD")
        self.aws_region = os.getenv("AWS_REGION")

        # Logger
        # self.logger = debug_logger()

        # Validate configuration
        self._validate_configuration()

    def _validate_configuration(self):
        """Validate that required environment variables are set when dynamic config is enabled."""
        if self.use_dynamic_config:
            missing_vars = []
            if not self.oauth2_username:
                missing_vars.append("INTERNAL_API_USERNAME")
            if not self.oauth2_password:
                missing_vars.append("INTERNAL_API_PASSWORD")

            if missing_vars:
                error_msg = f"FLEET_CONFIG_USE_DYNAMIC=true but required environment variables are missing: {', '.join(missing_vars)}"


                raise ValueError(error_msg)

            # self.logger.info(f"FleetConfigManager initialized with dynamic config enabled (TTL: {self.ttl_seconds}s)")
        else:
            # self.logger.warning("FleetConfigManager initialized with dynamic config DISABLED (using minimal fallback)")
            pass


    # ========================================================================
    # OAuth2 Authentication & Token Management
    # ========================================================================

    def _get_access_token(self) -> Tuple[str, str]:
        """
        Get cached OAuth2 tokens or fetch new ones.

        Returns:
            Tuple of (access_token, id_token)

        Raises:
            Exception if token acquisition fails
        """
        import httpx

        # Check if we have valid cached token (thread-safe read)
        with self.lock:
            if self.token_cache.get("expires_at"):
                if datetime.now() < self.token_cache["expires_at"]:
                    return self.token_cache["access_token"], self.token_cache["id_token"]

        # Fetch new token
        # self.logger.info("OAuth2 token expired or missing, fetching new token")

        try:
            response = httpx.post(
                f"{self.internal_api_base_url}/v2/auth/oauth2/token",
                json={
                    "grant_type": "password",
                    "username": self.oauth2_username,
                    "password": self.oauth2_password
                },
                headers={"Content-Type": "application/json"},
                timeout=10
            )
            response.raise_for_status()
            token_data = response.json()

            # Cache token with expiry (refresh 60s before actual expiry)
            expires_in = token_data.get("expires_in", 3600)
            expires_at = datetime.now() + timedelta(seconds=expires_in - 60)

            with self.lock:
                self.token_cache = {
                    "access_token": token_data["access_token"],
                    "id_token": token_data["id_token"],
                    "expires_at": expires_at
                }

            # self.logger.info(f"OAuth2 token acquired, expires at {expires_at}")
            return token_data["access_token"], token_data["id_token"]

        except Exception as e:
            # self.logger.exception(f"Failed to acquire OAuth2 token: {e}")
            raise

    def make_api_request(self, client_id: str, endpoint: str, params: Optional[Dict] = None, retries: int = 3, method: str = "GET") -> Dict[str, Any]:
        """
        Make authenticated API request with OAuth2 tokens and client context.

        Args:
            client_id: Client ID for x-lm-desired-account header
            endpoint: API endpoint path (e.g., "/v2/fleets/123")
            params: Optional query parameters
            retries: Number of retry attempts (default: 3)
            method: HTTP method (default: GET)

        Returns:
            JSON response as dict

        Raises:
            httpx.HTTPError if request fails after all retries
        """
        import httpx
        from time import sleep

        url = f"{self.internal_api_base_url}{endpoint}"

        for attempt in range(retries):
            try:
                # Get fresh tokens (cached or refresh)
                access_token, id_token = self._get_access_token()

                headers = {
                    "x-lm-desired-account": client_id,
                    "id-token": id_token,
                    "Authorization": f"Bearer {access_token}"
                }

                response = httpx.request(method, url, params=params, headers=headers, timeout=10)

                if response.status_code == 429:  # Rate limit
                    retry_after = int(response.headers.get("Retry-After", 2**attempt))
                    # self.logger.warning(f"Rate limited, retrying after {retry_after}s")
                    sleep(retry_after)
                    continue

                if response.status_code == 401:  # Token expired, force refresh
                    # self.logger.warning("Got 401, forcing token refresh")
                    with self.lock:
                        self.token_cache = {}
                    if attempt < retries - 1:
                        continue

                response.raise_for_status()
                return response.json()

            except httpx.HTTPError as e:
                if attempt < retries - 1:
                    backoff = 2**attempt
                    # self.logger.warning(f"API request failed (attempt {attempt+1}/{retries}), retrying in {backoff}s: {e}")
                    sleep(backoff)  # Exponential backoff
                else:
                    # self.logger.exception(f"API request failed after {retries} attempts: {e}")
                    resp = getattr(e, "response", None)
                    if resp is not None:
                        # self.logger.error(f"API error response body: {resp.text}")
                        pass
                    raise

        # Only reachable when every attempt was rate limited
        raise httpx.HTTPError(f"API request to {endpoint} still rate limited after {retries} attempts")