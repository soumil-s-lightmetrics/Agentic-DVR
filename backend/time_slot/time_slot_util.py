"""
Common helpers for the time-slot tools: trip parsing, geocoding, route geometry,
time parsing, time-slot building and violation fuzzy matching.
"""
import math
import os
from datetime import datetime, timezone, timedelta

import numpy as np
import requests
from dateutil import parser, tz
from dotenv import load_dotenv
from openai import OpenAI
from rapidfuzz import process, fuzz

from logger import debug_logger

load_dotenv()

logger = debug_logger()

GEOAPIFY_URL = "https://api.geoapify.com/v1/geocode/search"
AREA_TYPES = {"city", "suburb", "district", "county", "locality"}

M_PER_DEG_LAT = 111_320          # metres in 1° of latitude
EARTH_RADIUS_M = 6_371_000

TIME_SLOT_FORMAT = "%Y-%m-%d %H:%M:%S.%f"
MAX_THRESHOLD_MIN = {"dvr_clip": 1.5, "timelapse": 30}   # per side: 3 min DVR clip, 60 min timelapse

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def simplify_response(trip):
    """
    Simplifies the trip response from the API to a more manageable format.
    Includes trip details, asset information, driver information, and violations.
    """
    asset = trip.get('asset', {})
    return {
        'tripId': trip.get('tripId'),
        'startTime': trip.get('startTime'),
        'endTime': trip.get('endTime'),
        'startTimeUTC': trip.get('startTimeUTC'),
        'endTimeUTC': trip.get('endTimeUTC'),
        'timeZoneId': trip.get('timeZoneId'),
        'duration': trip.get('duration'),
        'distance': trip.get('distance'),
        'averageSpeed': trip.get('averageSpeed'),
        'maxSpeed': trip.get('maxSpeed'),
        'asset_info': {
            'assetId': asset.get('assetId'),
            'assetName': asset.get('assetName'),
            'cameraType': asset.get('cameraType'),
            'dutyType': asset.get('dutyType')},
        'fleetId': trip.get('fleetId'),
        'driverId': trip.get('driverId'),
        'driverName': trip.get('driverName'),
        'eventCount': trip.get('eventCount'),
        'violations': [{
            'event_index': violation.get('eventIndex'),
            'event_type': violation.get('eventType'),
            'event_timeUTC': violation.get('timestampUTC'),
            'event_location': {
                'latitude': violation.get('latitude'),
                'longitude': violation.get('longitude')},
            } for violation in trip.get('violations', [])],
    }


def get_trip_route(trip):
    route = []
    for point in trip.get('pathInfo', []):
        route.append({
            'timeOfDay': point.get('timeOfDay'),
            'timeUTC': point.get('timestampUTC'),
            'latitude': point.get('latitude'),
            'longitude': point.get('longitude')
        })
    summarised_route = route[::100]
    return {
        'route': route,
        'summarised_route': summarised_route
    }


def get_trip_upload_requests(trip):
    upload_requests = []
    for request in trip.get('uploadRequests', []):
        upload_requests.append({
            'requestId': request.get('requestId'),
            'status': request.get('status'),
            'timestamp': request.get('createdAt')})
    return upload_requests



def geocode_place(place_name):
    """Returns up to 5 candidate places [{'name', 'lat', 'lon', 'types'}, ...] for place_name."""
    params = {
        "text": place_name,
        "format": "json",
        "limit": 5,
        "bias": "countrycode:none",
        "apiKey": os.environ["GEOAPIFY_API_KEY"],
    }
    resp = requests.get(GEOAPIFY_URL, params=params, timeout=5)
    resp.raise_for_status()

    return [
        {
            "name": r.get("formatted", ""),
            "lat": r["lat"],
            "lon": r["lon"],
            "types": ["locality"] if r.get("result_type") in AREA_TYPES
                     else [r.get("result_type", "")],
        }
        for r in resp.json().get("results", [])
    ]


def route_bbox(route, margin_m=750):
    """
    route: [{'timeOfDay': 'HH:MM:SS', 'latitude': float, 'longitude': float}, ...]
    Returns (min_lat, max_lat, min_lon, max_lon) padded by margin_m metres, or None if route is empty.
    """
    if not route:
        return None

    lats = [p["latitude"] for p in route]
    lons = [p["longitude"] for p in route]
    min_lat, max_lat = min(lats), max(lats)
    min_lon, max_lon = min(lons), max(lons)

    lat_pad = margin_m / M_PER_DEG_LAT
    mid_lat = (min_lat + max_lat) / 2
    lon_pad = margin_m / (M_PER_DEG_LAT * math.cos(math.radians(mid_lat)))

    return (min_lat - lat_pad, max_lat + lat_pad,
            min_lon - lon_pad, max_lon + lon_pad)


def inside_bbox(lat, lon, bbox):
    min_lat, max_lat, min_lon, max_lon = bbox
    return min_lat <= lat <= max_lat and min_lon <= lon <= max_lon


def check_candidates_in_bbox(candidates, route):
    """Filters the candidates to only those that are within the bounding box of the route."""
    bbox = route_bbox(route)
    if bbox is None:
        return []
    return [c for c in candidates if inside_bbox(c["lat"], c["lon"], bbox)]


def haversine_m(lat1, lon1, lat2, lon2):
    """Distance in metres between two lat/lon points along the Earth's surface."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * EARTH_RADIUS_M * math.asin(math.sqrt(a))


def nearest_route_point(route, lat, lon, max_dist_m=1150):
    """
    route: [{'timeOfDay': 'HH:MM:SS', 'timeUTC': int, 'latitude': float, 'longitude': float}, ...]
    Returns {"time", "timeUTC", "dist_m", "lat", "lon"} of the closest route point,
    or None if even the closest is farther than max_dist_m.
    """
    best = None
    for p in route:
        d = haversine_m(lat, lon, p["latitude"], p["longitude"])
        if best is None or d < best["dist_m"]:
            best = {"time": p["timeOfDay"], "timeUTC": p["timeUTC"], "dist_m": d,
                    "lat": p["latitude"], "lon": p["longitude"]}

    if best is None or best["dist_m"] > max_dist_m:
        return None
    return best


def closest_point_candidate(route, candidates, threshold_m=1150):
    """
    Returns {"place": candidate, "point": route_point} for the candidate closest
    to the route within threshold_m, or None if none are close enough.
    """
    closest = None
    for c in candidates:
        point = nearest_route_point(route, c["lat"], c["lon"], max_dist_m=threshold_m)
        if point and (closest is None or point["dist_m"] < closest["point"]["dist_m"]):
            closest = {"place": c, "point": point}
    return closest



def to_utc_string(time_str: str):
    """Parses a free-form time string and returns it as 'YYYY-MM-DDTHH:MM:SSZ' in UTC (naive input is taken as UTC)."""
    dt = parser.parse(time_str)

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=tz.UTC)

    dt = dt.astimezone(tz.UTC)

    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_utc(value):
    """Converts an epoch-milliseconds int or an ISO 8601 string ('...Z') to an aware UTC datetime."""
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(value / 1000, tz=timezone.utc)
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def get_time_slot(time_utc, threshold_min=1.5):
    """
    Returns a time slot of threshold_min minutes on each side of time_utc
    (epoch milliseconds or ISO 8601 string), as 'YYYY-MM-DD HH:MM:SS.mmm' strings in UTC.
    """
    dt = parse_utc(time_utc)
    delta = timedelta(minutes=threshold_min)
    return {
        'startTimeUTC': (dt - delta).strftime(TIME_SLOT_FORMAT)[:-3],
        'endTimeUTC': (dt + delta).strftime(TIME_SLOT_FORMAT)[:-3]
    }


def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)

    return np.dot(a, b) / (
        np.linalg.norm(a) * np.linalg.norm(b)
    )


def get_embeddings(texts):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts
    )

    return [item.embedding for item in response.data]


def fuzzy_match_violation(violation, trip_info, threshold=0.5):
    """
    Matches the user's phrasing of a violation (e.g. 'speed violation') to one of the
    trip's event types (e.g. 'Traffic-Speed-Violated'). Returns None if nothing matches.
    """
    event_count = trip_info.get("eventCount") or {}
    violation_types = list(event_count.keys())

    if not violation_types:
        return None

    # Embed the user's violation and all violation types in ONE API call
    embeddings = get_embeddings([violation] + violation_types)
    violation_embedding, candidate_embeddings = embeddings[0], embeddings[1:]

    best_match = None
    best_score = -1

    for violation_type, candidate_embedding in zip(violation_types, candidate_embeddings):
        score = cosine_similarity(violation_embedding, candidate_embedding)

        if score > best_score:
            best_score = score
            best_match = violation_type

    logger.debug(f"fuzzy_match_violation: '{violation}' -> {best_match} ({best_score:.3f})")

    # Semantic match
    if best_score >= threshold:
        return best_match

    # Fuzzy/lexical fallback
    result = process.extractOne(
        violation,
        violation_types,
        scorer=fuzz.WRatio
    )

    if result and result[1] >= 85:
        return result[0]

    return None


def occurred_violation_types(trip_info):
    """Event types that occurred at least once in the trip (excluding 'total')."""
    event_count = trip_info.get("eventCount") or {}
    return [k for k, v in event_count.items() if k != "total" and v]
