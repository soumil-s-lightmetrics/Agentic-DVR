"""
Dynamic Fleet Configuration Manager

Fetches fleet configurations from multiple internal data sources:
- DynamoDB (client_configs table) for fleet portal version
- REST APIs for APK versions, camera models, fleet plans, and disabled events

Features:
- OAuth2 password grant authentication with token caching
- TTL-based caching for configs and filter attributes
- Thread-safe operations
- Graceful degradation: fresh data → cached → minimal fallback
- Feature flag for emergency rollback
"""

import os
import threading
import hashlib
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List, Tuple
from cachetools import TTLCache
from logger import debug_logger


class FleetConfigManager:
    """
    Manages fleet configurations with dynamic fetching from internal APIs and DynamoDB.

    Configuration is fetched on-demand and cached with TTL. Supports graceful degradation
    when APIs are unavailable (falls back to cached or minimal default config).
    """

    def __init__(self, ttl_seconds: int = 3600, use_dynamic_config: bool = True):
        """
        Initialize the FleetConfigManager.

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
        self._dynamodb_client = None
        self._http_client = None

        # Environment variables (OAuth2 credentials and API endpoints)
        self.internal_api_base_url = os.getenv("INTERNAL_API_BASE_URL")
        self.oauth2_username = os.getenv("INTERNAL_API_USERNAME")
        self.oauth2_password = os.getenv("INTERNAL_API_PASSWORD")
        self.aws_region = os.getenv("AWS_REGION")
        self.dynamodb_table_name = os.getenv("DYNAMODB_CLIENT_CONFIGS_TABLE")

        # Logger
        self.logger = debug_logger()

        # Minimal default fallback (test configs have no prod meaning)
        self._initialize_minimal_fallback()

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
            if not self.dynamodb_table_name:
                missing_vars.append("DYNAMODB_CLIENT_CONFIGS_TABLE")

            if missing_vars:
                error_msg = f"FLEET_CONFIG_USE_DYNAMIC=true but required environment variables are missing: {', '.join(missing_vars)}"
                self.logger.error(error_msg)
                raise ValueError(error_msg)

            self.logger.info(f"FleetConfigManager initialized with dynamic config enabled (TTL: {self.ttl_seconds}s)")
        else:
            self.logger.warning("FleetConfigManager initialized with dynamic config DISABLED (using minimal fallback)")

    def _initialize_minimal_fallback(self):
        """Initialize minimal fallback config (test configs have no production meaning)."""
        self.minimal_fallback = {
            "fleet_portal_version": "v10.9.0",
            "device_apk_version": "v1.23.1",
            "camera_models": [
                "mitac-gemini",
                "mitac-sprint-k220",
                "mitac-evo-k265",
                "jimi-jc261",
                "jimi-jc261p",
                "jimi-jc450",
                "jimi-jc400",
                "jimi-jc400p"
            ],
            "disabled_standard_events": [],
            "plan": "NON-SHIELD",
        }

    def _get_minimal_fallback(self) -> Dict[str, Any]:
        """Get minimal fallback config."""
        return self.minimal_fallback.copy()

    # ========================================================================
    # Public API Methods
    # ========================================================================

    def get_fleet_config(self, client_id: Optional[str], fleet_id: Optional[str]) -> Dict[str, Any]:
        """
        Get fleet configuration for the specified client and fleet.

        BREAKING CHANGE: Now requires both client_id and fleet_id (was only fleet_id before).

        Fetches from cache or APIs, with graceful fallback to stale cache or minimal config.

        Args:
            client_id: Client ID (TSP/account identifier)
            fleet_id: Fleet ID within the client

        Returns:
            Fleet configuration dict with keys:
            - fleet_portal_version: e.g., "v10.9.0"
            - device_apk_version: e.g., "v1.23.1"
            - camera_models: list of model strings
            - disabled_standard_events: list of event type strings
            - plan: "SHIELD" or "NON-SHIELD"
        """
        # If dynamic config is disabled, return minimal fallback (emergency kill switch)
        if not self.use_dynamic_config:
            self.logger.warning(f"Dynamic config disabled, returning minimal fallback for client={client_id}, fleet={fleet_id}")
            return self._get_minimal_fallback()

        # Validate inputs
        if not client_id or not fleet_id:
            self.logger.warning(f"Missing client_id or fleet_id (client={client_id}, fleet={fleet_id}), returning minimal fallback")
            return self._get_minimal_fallback()

        # Build cache key (composite key since fleet_id repeats across clients)
        cache_key = f"{client_id}:{fleet_id}"

        # Check cache (thread-safe read)
        with self.lock:
            if cache_key in self.config_cache:
                self.logger.debug(f"Cache HIT for {cache_key}")
                return self.config_cache[cache_key].copy()

        # Cache miss - fetch from APIs
        self.logger.debug(f"Cache MISS for {cache_key}, fetching from APIs")

        try:
            # Aggregate config from all data sources
            config = self._aggregate_fleet_config(client_id, fleet_id)

            # Cache the result (thread-safe write)
            with self.lock:
                self.config_cache[cache_key] = config

            self.logger.info(f"Successfully fetched and cached config for {cache_key}")
            return config.copy()

        except Exception as e:
            self.logger.exception(f"Failed to fetch config for {cache_key}: {e}")

            # Try returning stale cache (even if expired)
            with self.lock:
                if cache_key in self.config_cache:
                    self.logger.warning(f"Returning stale cached config for {cache_key}")
                    return self.config_cache[cache_key].copy()

            # No cache available - return minimal fallback
            self.logger.error(f"No cached config available for {cache_key}, returning minimal fallback")
            return self._get_minimal_fallback()

    def get_filter_attributes(self, fleet_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse fleet config into filter attributes for vector store queries.

        Caches results to avoid redundant parsing (eliminates 9 calls per request).

        Args:
            fleet_config: Fleet configuration dict

        Returns:
            Dict with parsed version numbers and filter lists:
            - fleet_portal_version_major/minor/patch: int
            - device_apk_version_major/minor: int
            - device_models_in: list of strings
            - plans_in: list with single plan string
            - event_type_in: list of disabled event strings
            - required_features: empty list
        """
        # Generate cache key from config hash
        cache_key = self._hash_config(fleet_config)
        self.logger.debug(f"Computed filter attributes cache key: {cache_key}")
        # Check cache (thread-safe read)
        with self.lock:
            if cache_key in self.filter_attributes_cache:
                self.logger.debug(f"Filter attributes cache HIT for hash={cache_key[:8]}")
                return self.filter_attributes_cache[cache_key].copy()

        # Cache miss - parse config
        self.logger.debug(f"Filter attributes cache MISS for hash={cache_key[:8]}, parsing")

        try:
            # Parse version strings
            portal_version_parts = fleet_config["fleet_portal_version"].lstrip('v').split('.')
            apk_version_parts = fleet_config["device_apk_version"].lstrip('v').split('.')

            filter_attrs = {
                "fleet_portal_version_major": int(portal_version_parts[0]),
                "fleet_portal_version_minor": int(portal_version_parts[1]),
                "fleet_portal_version_patch": int(portal_version_parts[2]),
                "device_apk_version_major": int(apk_version_parts[0]),
                "device_apk_version_minor": int(apk_version_parts[1]),
                "device_models_in": fleet_config["camera_models"],
                "plans_in": [fleet_config["plan"]],
                "event_type_in": fleet_config["disabled_standard_events"],
                "required_features": []
            }

            # Cache result (thread-safe write)
            with self.lock:
                self.filter_attributes_cache[cache_key] = filter_attrs

            return filter_attrs.copy()

        except Exception as e:
            self.logger.exception(f"Failed to parse filter attributes: {e}")
            # Return minimal structure to avoid breaking downstream code
            return {
                "fleet_portal_version_major": 0,
                "fleet_portal_version_minor": 0,
                "fleet_portal_version_patch": 0,
                "device_apk_version_major": 0,
                "device_apk_version_minor": 0,
                "device_models_in": [],
                "plans_in": [],
                "event_type_in": [],
                "required_features": []
            }

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
        self.logger.info("OAuth2 token expired or missing, fetching new token")

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

            self.logger.info(f"OAuth2 token acquired, expires at {expires_at}")
            return token_data["access_token"], token_data["id_token"]

        except Exception as e:
            self.logger.exception(f"Failed to acquire OAuth2 token: {e}")
            raise

    def _make_api_request(self, client_id: str, endpoint: str, params: Optional[Dict] = None, retries: int = 3) -> Dict[str, Any]:
        """
        Make authenticated API request with OAuth2 tokens and client context.

        Args:
            client_id: Client ID for x-lm-desired-account header
            endpoint: API endpoint path (e.g., "/v2/fleets/123")
            params: Optional query parameters
            retries: Number of retry attempts (default: 3)

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

                response = httpx.get(url, params=params, headers=headers, timeout=10)

                if response.status_code == 429:  # Rate limit
                    retry_after = int(response.headers.get("Retry-After", 2**attempt))
                    self.logger.warning(f"Rate limited, retrying after {retry_after}s")
                    sleep(retry_after)
                    continue

                if response.status_code == 401:  # Token expired, force refresh
                    self.logger.warning("Got 401, forcing token refresh")
                    with self.lock:
                        self.token_cache = {}
                    if attempt < retries - 1:
                        continue

                response.raise_for_status()
                return response.json()

            except httpx.HTTPError as e:
                if attempt < retries - 1:
                    backoff = 2**attempt
                    self.logger.warning(f"API request failed (attempt {attempt+1}/{retries}), retrying in {backoff}s: {e}")
                    sleep(backoff)  # Exponential backoff
                else:
                    self.logger.exception(f"API request failed after {retries} attempts: {e}")
                    raise

    # ========================================================================
    # DynamoDB Client
    # ========================================================================

    def _get_dynamodb_client(self):
        """Lazy initialization of DynamoDB client with double-checked locking."""
        if self._dynamodb_client is None:
            with self.lock:
                if self._dynamodb_client is None:
                    import boto3
                    self._dynamodb_client = boto3.client('dynamodb', region_name=self.aws_region)
                    self.logger.info(f"Initialized DynamoDB client for region {self.aws_region}")
        return self._dynamodb_client

    # ========================================================================
    # Private Fetch Methods (One per Data Source)
    # ========================================================================

    def _fetch_fleet_portal_version(self, client_id: str, fleet_id: str) -> Optional[str]:
        """
        Fetch fleet portal version from DynamoDB client_configs table.

        Args:
            client_id: Client ID (used as DynamoDB partition key)
            fleet_id: Fleet ID (not used in DynamoDB query, but kept for consistency)

        Returns:
            Version string (e.g., "v10.9.0") or None if not found/error
        """
        try:
            dynamodb = self._get_dynamodb_client()

            response = dynamodb.get_item(
                TableName=self.dynamodb_table_name,
                Key={'clientId': {'S': client_id}}
            )

            if 'Item' not in response:
                self.logger.warning(f"No DynamoDB item found for client_id={client_id}")
                return None

            # Navigate nested structure: Item.fleetRebranding.M.fleetPortalVersion.S
            item = response['Item']
            if 'fleetRebranding' in item and 'M' in item['fleetRebranding']:
                fleet_rebranding = item['fleetRebranding']['M']
                if 'fleetPortalVersion' in fleet_rebranding and 'S' in fleet_rebranding['fleetPortalVersion']:
                    version = fleet_rebranding['fleetPortalVersion']['S']
                    self.logger.debug(f"Fetched fleet portal version from DynamoDB: {version}")
                    return version

            self.logger.warning(f"fleetPortalVersion not found in DynamoDB item for client_id={client_id}")
            return None

        except Exception as e:
            self.logger.exception(f"Failed to fetch fleet portal version from DynamoDB: {e}")
            return None

    def _fetch_apk_versions(self, client_id: str, fleet_id: str) -> Optional[str]:
        """
        Fetch latest APK version from diagnostics API.

        Args:
            client_id: Client ID for authentication
            fleet_id: Fleet ID for query parameter

        Returns:
            Latest version string (e.g., "v1.23.1") or None if not found/error
        """
        try:
            response = self._make_api_request(
                client_id,
                "/v2/diagnostics/apk-versions/aggregate",
                params={"fleetId": fleet_id}
            )

            apk_versions = response.get("apkVersions", {})
            if not apk_versions:
                self.logger.warning(f"No APK versions found for fleet_id={fleet_id}")
                return None

            # Find latest version
            latest_version = max(apk_versions.keys(), key=self._parse_version)
            self.logger.debug(f"Fetched latest APK version: {latest_version}")
            return latest_version

        except Exception as e:
            self.logger.exception(f"Failed to fetch APK versions: {e}")
            return None

    def _fetch_camera_models(self, client_id: str) -> Optional[List[str]]:
        """
        Fetch valid camera models from device management API.

        Args:
            client_id: Client ID for authentication

        Returns:
            List of camera model strings or None if error
        """
        try:
            response = self._make_api_request(
                client_id,
                "/v2/device-management/valid-vendors"
            )

            vendors = response.get("vendors", [])
            if not vendors:
                self.logger.warning(f"No vendors found for client_id={client_id}")
                return None

            # Map vendor IDs to camera models
            camera_models = []
            vendor_mapping = {
                "mitac_v": ["mitac-gemini", "mitac-sprint-k220", "mitac-evo-k265"],
                "jimi_v": ["jimi-jc261", "jimi-jc261p", "jimi-jc450", "jimi-jc400", "jimi-jc400p"]
            }

            for vendor in vendors:
                vendor_id = vendor.get("vendorId", "")
                if vendor_id in vendor_mapping:
                    camera_models.extend(vendor_mapping[vendor_id])

            self.logger.debug(f"Fetched camera models: {camera_models}")
            return camera_models if camera_models else None

        except Exception as e:
            self.logger.exception(f"Failed to fetch camera models: {e}")
            return None

    def _fetch_fleet_plan(self, client_id: str, fleet_id: str) -> Optional[str]:
        """
        Fetch fleet plan (SHIELD vs NON-SHIELD) from fleets API.

        Args:
            client_id: Client ID for authentication
            fleet_id: Fleet ID for API path

        Returns:
            "SHIELD" or "NON-SHIELD" or None if error
        """
        try:
            response = self._make_api_request(
                client_id,
                f"/v2/fleets/{fleet_id}"
            )

            # Check preferences.fleetRidecamPlusPlan (primary)
            preferences = response.get("preferences", {})
            plan = preferences.get("fleetRidecamPlusPlan")

            if not plan:
                return "NON-SHIELD"  # Default to NON-SHIELD if not set

            if plan:
                # Normalize to uppercase and ensure SHIELD/NON-SHIELD format
                plan_upper = plan.upper()
                if "SHIELD" in plan_upper:
                    result = "SHIELD" if plan_upper == "SHIELD" else "NON-SHIELD"
                    self.logger.debug(f"Fetched fleet plan: {result}")
                    return result

            self.logger.warning(f"No fleet plan found for fleet_id={fleet_id}")
            return "NON-SHIELD"

        except Exception as e:
            self.logger.exception(f"Failed to fetch fleet plan: {e}")
            return "NON-SHIELD"

    def _fetch_disabled_events(self, client_id: str, fleet_id: str) -> Optional[List[str]]:
        """
        Fetch disabled standard events from configuration API.

        Args:
            client_id: Client ID for authentication
            fleet_id: Fleet ID for query parameter

        Returns:
            List of disabled event type strings or None if error
        """
        try:
            response = self._make_api_request(
                client_id,
                "/v2/configuration/events",
                params={"fleetId": fleet_id}
            )

            standard_events = response.get("standardEvents", [])
            disabled_events = [
                event["eventType"]
                for event in standard_events
                if event.get("state") == "DISABLED"
            ]

            self.logger.debug(f"Fetched disabled events: {disabled_events}")
            return disabled_events

        except Exception as e:
            self.logger.exception(f"Failed to fetch disabled events: {e}")
            return None

    # ========================================================================
    # Aggregation Logic
    # ========================================================================

    def _aggregate_fleet_config(self, client_id: str, fleet_id: str) -> Dict[str, Any]:
        """
        Fetch all attributes from data sources and combine with per-attribute fallbacks.

        Each fetch method handles errors internally and returns None on failure.
        We use minimal fallback values for any attributes that fail to fetch.

        Args:
            client_id: Client ID
            fleet_id: Fleet ID

        Returns:
            Complete fleet config dict (never raises, always returns valid config)
        """
        self.logger.info(f"Aggregating fleet config for client={client_id}, fleet={fleet_id}")

        # Fetch from all sources (methods return None on failure)
        portal_version = self._fetch_fleet_portal_version(client_id, fleet_id)
        apk_version = self._fetch_apk_versions(client_id, fleet_id)
        camera_models = self._fetch_camera_models(client_id)
        plan = self._fetch_fleet_plan(client_id, fleet_id)
        disabled_events = self._fetch_disabled_events(client_id, fleet_id)

        # Get minimal fallback for missing attributes
        fallback = self._get_minimal_fallback()

        # Build config with per-attribute fallbacks
        config = {
            "fleet_portal_version": portal_version or fallback["fleet_portal_version"],
            "device_apk_version": apk_version or fallback["device_apk_version"],
            "camera_models": camera_models or fallback["camera_models"],
            "disabled_standard_events": disabled_events if disabled_events is not None else fallback["disabled_standard_events"],
            "plan": plan or fallback["plan"],
        }

        # Log which attributes used fallback
        fallback_used = []
        if portal_version is None:
            fallback_used.append("fleet_portal_version")
        if apk_version is None:
            fallback_used.append("device_apk_version")
        if camera_models is None:
            fallback_used.append("camera_models")
        if disabled_events is None:
            fallback_used.append("disabled_standard_events")
        if plan is None:
            fallback_used.append("plan")

        if fallback_used:
            self.logger.warning(f"Used fallback for attributes: {fallback_used}")

        return config

    # ========================================================================
    # Helper Methods
    # ========================================================================

    def _hash_config(self, config: Dict[str, Any]) -> str:
        """Generate stable hash of config for caching."""
        # Convert to sorted tuple for stable hashing
        config_str = str(sorted(config.items()))
        return hashlib.md5(config_str.encode()).hexdigest()

    def _parse_version(self, version: str) -> Tuple[int, int, int]:
        """
        Parse version string into tuple for comparison.

        Args:
            version: Version string (e.g., "v1.23.1" or "1.23.1")

        Returns:
            Tuple of (major, minor, patch)
        """
        try:
            parts = version.lstrip('v').split('.')
            return (int(parts[0]), int(parts[1]), int(parts[2]) if len(parts) > 2 else 0)
        except Exception:
            return (0, 0, 0)
