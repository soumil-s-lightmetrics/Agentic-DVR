"""
Time-slot tool for when the user identifies the footage by a location,
optionally with a violation/event that happened there.
"""
from typing import Annotated, Literal

from langchain_core.tools import InjectedToolArg, tool

from logger import debug_logger

from time_slot.time_slot_util import (
    MAX_THRESHOLD_MIN,
    check_candidates_in_bbox,
    closest_point_candidate,
    fuzzy_match_violation,
    geocode_place,
    get_time_slot,
    haversine_m,
)

logger = debug_logger()

def closest_point_event(trip_info, violation_name, candidates, threshold_m=500):
    """
    trip_info["violations"] -> [{'event_type', 'event_timeUTC',
                'event_location': {'latitude', 'longitude'}}, ...]
    violation_name: matched event type (e.g. 'Traffic-Speed-Violated') or None for any event.
    candidates: geocoded places [{'name', 'lat', 'lon', ...}, ...]

    Returns (closest, message):
      - closest = {"place": candidate, "event": violation, "dist_m": float} for the event
        nearest to any candidate within threshold_m, message = None
      - (None, message) if no matching event occurred within threshold_m of any candidate.
    """
    violations = trip_info.get('violations', [])
    events = (violations if violation_name is None
              else [v for v in violations if v.get("event_type") == violation_name])
    label = violation_name or "violation"

    closest = None
    for c in candidates:
        for v in events:
            loc = v.get("event_location", {})
            lat, lon = loc.get("latitude"), loc.get("longitude")
            if not isinstance(lat, (int, float)) or not isinstance(lon, (int, float)):
                continue
            d = haversine_m(c["lat"], c["lon"], lat, lon)
            if d <= threshold_m and (closest is None or d < closest["dist_m"]):
                closest = {"place": c, "event": v, "dist_m": d}

    if closest is None:
        return None, f"No {label} event occurred within {threshold_m} m of this location."
    return closest, None


# when only place is mentioned, place with event is mentioned
@tool
def get_time_slot_from_place(trip_info: Annotated[dict, InjectedToolArg], route: Annotated[list, InjectedToolArg],
                             place_name: str, violation_mentioned: bool = False,
                             violation_name: str | None = None, threshold_min: float = 1.5,
                             clip_type: Literal["dvr_clip", "timelapse"] = "dvr_clip"):
    """
    Find the footage time slot for when the selected trip was at a place the user mentions.

    Purpose:
        Geocodes the place the user mentioned, checks it lies on the trip's
        route, and returns the time slot (start and end time) for the DVR
        request. If the user also mentions that a violation/event happened at
        that place, the tool looks for that violation near the place and centres
        the slot on it. Otherwise it centres the slot on the moment the vehicle
        passed closest to the place.

    When to use:
        Use when the user identifies the moment by a LOCATION, with or without
        a violation:
        - Place only:
            "get me the footage from when the truck was near Hosur"
            "DVR clip of the trip near Silk Board"
        - Place with an unnamed event:
            "an event occurred near Bhuvanahalli"
        - Place with a named violation:
            "the overspeeding near Bhuvanahalli"

    When NOT to use:
        - The user mentions a violation/event with no location, with or without
          a time ("harsh braking at 4:20 pm", "a violation occurred"). Use
          get_time_slot_from_violation instead.
        - No trip is selected yet. Fetch and select a trip first.

    Args:
        trip_info (dict): Injected by the agent, not chosen by the model.
            Simplified trip details for the selected trip.
        route (list): Injected by the agent, not chosen by the model. Route
            points of the selected trip.
        place_name (str): The place as the user said it, e.g. "Bhuvanahalli".
            Misspellings are tolerated, since geocoding resolves to the closest
            real place.
        violation_mentioned (bool): True if the user said a violation/event
            happened at the place, whether or not they named it. False if they
            only mentioned the place.
        violation_name (str | None): Violation as the user phrased it, e.g.
            "traffic speed violation". It is fuzzy-matched to the trip's
            violation types. Pass None if no violation was named.
        threshold_min (float): Minutes of footage on EACH side of the matched
            moment. Default 1.5, which gives a 3-minute clip. Capped at 1.5 for
            a DVR clip and 30 for a timelapse.
        clip_type (str): "timelapse" if the user asked for a timelapse,
            otherwise "dvr_clip".

    Returns:
        dict with a "status" key:
        - "success": the place was matched on the route. Includes "place"
          (resolved place name), "timeSlot" (start and end time of the clip)
          and "basis":
            - "place_event": centred on "violation", the matching event
              "distM" metres from the place.
            - "place": centred on "routePoint" ("timeUTC", "distM"), the
              moment the vehicle passed closest to the place. If a violation
              was mentioned but none was found near the place, "message"
              says so.
        - "not_found": the place could not be geocoded. Includes "query". Ask
          the user to rephrase or name a nearby landmark.
        - "not_on_route": the place exists but the trip did not pass near it.
          Includes "closest_guess" (the best geocoded match). Tell the user the
          trip didn't go there and confirm the place.
    """
    threshold_min = min(threshold_min, MAX_THRESHOLD_MIN[clip_type])

    candidates = geocode_place(place_name)
    if not candidates:
        return {"status": "not_found", "query": place_name}

    candidates_in_area = check_candidates_in_bbox(candidates, route)
    if not candidates_in_area:
        return {"status": "not_on_route", "closest_guess": candidates[0]["name"]}

    no_event_message = None
    if violation_mentioned:
        matched_violation = None
        if violation_name:
            matched_violation = fuzzy_match_violation(violation_name, trip_info)

        if violation_name and matched_violation is None:
            no_event_message = f"{violation_name} did not occur in the trip."
        else:
            closest, no_event_message = closest_point_event(trip_info, matched_violation, candidates_in_area, threshold_m=1000)
            if closest:
                return {
                    "status": "success",
                    "basis": "place_event",
                    "place": closest["place"]["name"],
                    "violation": closest["event"],
                    "distM": round(closest["dist_m"]),
                    "timeSlot": get_time_slot(closest["event"].get("event_timeUTC"), threshold_min)
                }
        logger.debug(f"get_time_slot_from_place: {no_event_message} Falling back to the place.")

    closest = closest_point_candidate(route, candidates_in_area)
    if not closest:
        return {"status": "not_on_route", "closest_guess": candidates[0]["name"]}

    result = {
        "status": "success",
        "basis": "place",
        "place": closest["place"]["name"],
        "routePoint": {"timeUTC": closest["point"]["timeUTC"], "distM": round(closest["point"]["dist_m"])},
        "timeSlot": get_time_slot(closest["point"]["timeUTC"], threshold_min)
    }
    if no_event_message:
        result["message"] = no_event_message
    return result
