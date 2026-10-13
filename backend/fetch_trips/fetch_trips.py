"""
Fetch trips: pulls a fleet's trips from the trips API, filters them by driver,
asset, events and time window, and adds how long each trip's DVR footage stays
available (footage is kept for 17.5 days from the trip start).
"""
from datetime import datetime, timedelta, timezone

from dotenv import load_dotenv

# utils.auth creates the AuthManager on import, which needs the API credentials from .env
load_dotenv()

from logger import debug_logger
from utils.auth import auth_manager

logger = debug_logger()

PAGE_SIZE = 500
DEFAULT_MAX_TRIPS = 1000
DVR_RETENTION_DAYS = 17.5
DVR_EXPIRING_DAYS = 3     # footage with less than this many days left is 'expiring'


def parse_utc(value):
    """ISO 8601 string ('...Z', or naive = UTC) -> aware UTC datetime."""
    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def fetch_all_trips(fleet_id, client_id, base_params, skip=0, max_trips=DEFAULT_MAX_TRIPS):
    """
    Pages through /v2/fleets/{fleet_id}/trips, newest first, starting at `skip`, until
    `max_trips` trips are fetched or the API has no more.
    """
    all_trips = []
    while len(all_trips) < max_trips:
        limit = min(PAGE_SIZE, max_trips - len(all_trips))
        params = {**base_params, "key": "startTimeUTC", "sort": "desc", "skip": skip, "limit": limit}
        try:
            body = auth_manager.make_api_request(
                client_id=client_id,
                endpoint=f"/v2/fleets/{fleet_id}/trips",
                params=params,
            )
        except Exception as e:
            # Raise instead of returning a partial/empty list that reads as "No trips found"
            # (e.g. a 403 for an account without access to this fleet)
            logger.error(f"fetch_all_trips: failed at skip={skip}: {e}")
            raise

        rows = body.get("rows", [])
        logger.debug(f"fetch_all_trips: skip={skip} limit={limit} got={len(rows)}")
        if not rows:
            break
        all_trips.extend(rows)
        if len(rows) < limit:
            break
        skip += limit

    logger.info(f"fetch_all_trips: total trips fetched {len(all_trips)}")
    return all_trips


def dvr_availability(start_time_utc, now):
    """Returns (status, available_until, days_left) for a trip's DVR footage."""
    if not start_time_utc:
        return "unknown", None, None

    expiry = parse_utc(start_time_utc) + timedelta(days=DVR_RETENTION_DAYS)
    days_left = (expiry - now).total_seconds() / 86400
    if days_left > DVR_EXPIRING_DAYS:
        status = "available"
    elif days_left > 0:
        status = "expiring"
    else:
        status = "expired"
    return status, expiry.strftime("%d %b %Y %H:%M"), round(days_left, 1)


def enrich_trip(trip, now):
    """Simplifies an API trip to the fields shown in the trips table, plus DVR availability."""
    start = trip.get("startTimeUTC", "")
    end = trip.get("endTimeUTC", "")
    dvr_status, dvr_until, dvr_days = dvr_availability(start, now)

    events = [
        {"type": event_type, "count": count}
        for event_type, count in (trip.get("eventCount") or {}).items()
        if event_type != "total" and count > 0
    ]

    return {
        "tripId": trip.get("tripId"),
        "driverId": trip.get("driverId"),
        "driverName": trip.get("driverName", trip.get("driverId", "—")),
        "assetId": trip.get("asset", {}).get("assetId", "—"),
        "startTimeUTC": start,
        "endTimeUTC": end,
        "lastPinged": end if end else trip.get("lastPingedAt", start),
        "lastPingedLabel": "Ended" if end else "Ongoing",
        "dvr_status": dvr_status,
        "dvr_until": dvr_until,
        "dvr_days": dvr_days,
        "events": events,
        "totalEvents": sum(e["count"] for e in events),
    }


def has_events(trip, events):
    """
    True if the enriched trip has every event type in `events` (case-insensitive),
    with exactly the given count where one is given.
    """
    trip_counts = {e["type"].lower(): e["count"] for e in trip["events"]}
    for event_type, count in events.items():
        trip_count = trip_counts.get(event_type.lower())
        if trip_count is None or (count is not None and trip_count != count):
            return False
    return True


def fetch_trips_with_expiry(
        fleet_id,
        client_id,
        start_time=None,
        end_time=None,
        driver_ids=None,
        asset_ids=None,
        events=None,
        limit_to_latest=None,
        skip=0):
    """
    Fetches the fleet's trips and returns them filtered, enriched and sorted newest first.

    Args:
        fleet_id (str): Fleet to fetch trips for.
        client_id (str): Account sent as the x-lm-desired-account header.
        start_time, end_time (str | None): ISO 8601 window (UTC) the trip must start in.
            Required unless limit_to_latest is set.
        driver_ids (list[str] | None): Only trips by these drivers.
        asset_ids (list[str] | None): Only trips on these assets.
        events (dict[str, int | None] | None): Event type -> count the trip must have.
            None means the trip just has to have that event type, e.g.
            {"trafficSpeedViolated": 2, "harshBraking": None}.
        limit_to_latest (int | None): Fetch the latest N trips instead of a time window.
        skip (int): Trips to skip, for loading more results.

    Returns:
        list[dict]: Enriched trips (see enrich_trip), newest first.

    Raises:
        ValueError: No time window and no limit_to_latest, or the window is in the future.
        httpx.HTTPError: The trips API request failed.
    """
    if limit_to_latest:
        params = {}
        max_trips = limit_to_latest
    else:
        if not (start_time and end_time):
            raise ValueError("start_time and end_time are required unless limit_to_latest is set.")

        window_start = parse_utc(start_time)
        window_end = parse_utc(end_time)
        if window_start.date() > datetime.now(timezone.utc).date():
            raise ValueError(
                "That date range is in the future - trips haven't happened yet. "
                "Please choose a date range up to today."
            )

        # The API only takes whole-date bounds, and 'before' is exclusive
        params = {
            "after": window_start.date().isoformat(),
            "before": (window_end.date() + timedelta(days=1)).isoformat(),
        }
        max_trips = DEFAULT_MAX_TRIPS

    if driver_ids:
        params["driverId"] = ",".join(driver_ids)

    trips = fetch_all_trips(fleet_id, client_id, params, skip=skip, max_trips=max_trips)
    logger.info(f"fetch_trips_with_expiry: fetched {len(trips)} trips for fleet {fleet_id}")

    # The endpoint can't filter by asset, so filter here
    if asset_ids:
        trips = [t for t in trips if t.get("asset", {}).get("assetId") in asset_ids]
    if driver_ids:
        trips = [t for t in trips if t.get("driverId", "") in driver_ids]

    now = datetime.now(timezone.utc)
    enriched = [enrich_trip(trip, now) for trip in trips]

    if events:
        enriched = [t for t in enriched if has_events(t, events)]

    # The API returns whole days, so narrow down to the requested time window
    # (e.g. "around 5PM" -> 16:57:30-17:07:30)
    if not limit_to_latest:
        enriched = [
            t for t in enriched
            if t["startTimeUTC"] and window_start <= parse_utc(t["startTimeUTC"]) <= window_end
        ]

    enriched.sort(key=lambda t: t["startTimeUTC"] or "", reverse=True)
    return enriched[:limit_to_latest] if limit_to_latest else enriched


def fetch_trip_details(fleet_id, client_id, driver_id, trip_id):
    """Full trip response (violations, route, upload requests) for the selected trip."""
    return auth_manager.make_api_request(
        client_id=client_id,
        endpoint=f"/v1/fleets/{fleet_id}/drivers/{driver_id}/trips/{trip_id}",
        params={"includeViolations": True, "includePathInfo": True, "includeUploadRequests": True},
    )

# guide for the end orchestrator
# the returned trip has to be one if the returned trip is more than 10 ask the user to specify a single date
# if heven't already and same for driver if haven't already.

# if all of these conditions are satisfied then ask the user to select one of the trips from the given if the user has asked
# for a request of DVR or timelapse, if the user has simply asked to show trips then the above guidelines will not be applicable
