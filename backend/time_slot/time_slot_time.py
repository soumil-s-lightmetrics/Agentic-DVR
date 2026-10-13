"""
Time-slot tool for when the user identifies the footage only by a time, with no place or violation/event.
"""
from datetime import datetime, timezone
from typing import Annotated, Literal

from langchain_core.tools import InjectedToolArg, tool

from logger import debug_logger
from time_slot.time_slot_util import MAX_THRESHOLD_MIN, get_time_slot, parse_utc

logger = debug_logger()


def as_utc(value):
    """parse_utc, treating a time without a timezone as UTC."""
    dt = parse_utc(value)
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


@tool
def get_time_slot_from_time(trip_info: Annotated[dict, InjectedToolArg], time_utc: str,
                            threshold_min: float = 1.5,
                            clip_type: Literal["dvr_clip", "timelapse"] = "dvr_clip"):
    """
    Find the footage time slot around a time of the selected trip the user mentions.

    Purpose:
        Centres the slot on the time the user gave, after checking the trip was
        running at that time.

    When to use:
        Use when the user identifies the moment ONLY by a time, without a place
        or a violation/event:
            "a 2 minute clip at 10:15 AM"
            "get me the footage from 4:20 pm"
            "timelapse of the trip from 9 to 9:30" (the middle of the range, with
            threshold_min half of its length)

    When NOT to use:
        - The user mentions a violation/event, with or without a time. Use
          get_time_slot_from_violation.
        - The user mentions a place. Use get_time_slot_from_place.

    Args:
        trip_info (dict): Injected by the agent, not chosen by the model.
            Simplified trip details for the selected trip.
        time_utc (str): The time the user gave, as an ISO 8601 string in UTC on
            the trip's date, e.g. "2026-10-08T10:15:00Z".
        threshold_min (float): Minutes of footage on EACH side of the time.
            Default 1.5, which gives a 3-minute clip. Capped at 1.5 for a DVR
            clip and 30 for a timelapse.
        clip_type (str): "timelapse" if the user asked for a timelapse,
            otherwise "dvr_clip".

    Returns:
        dict with a "status" key:
        - "success": "time" (the moment used) and "timeSlot" (start and end
          time of the clip).
        - "not_in_trip": the trip wasn't running at that time. Includes
          "tripStart" and "tripEnd"; ask the user for a time between them.
    """
    threshold_min = min(threshold_min, MAX_THRESHOLD_MIN[clip_type])

    moment = as_utc(time_utc)
    trip_start = as_utc(trip_info["startTimeUTC"])
    # an ongoing trip has no end yet
    trip_end = as_utc(trip_info["endTimeUTC"]) if trip_info.get("endTimeUTC") else datetime.now(timezone.utc)

    if not trip_start <= moment <= trip_end:
        logger.debug(f"get_time_slot_from_time: {moment} is outside the trip ({trip_start} - {trip_end})")
        return {
            "status": "not_in_trip",
            "time": moment.isoformat(),
            "tripStart": trip_start.isoformat(),
            "tripEnd": trip_end.isoformat(),
        }

    return {
        "status": "success",
        "basis": "time",
        "time": moment.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "timeSlot": get_time_slot(moment.strftime("%Y-%m-%dT%H:%M:%SZ"), threshold_min),
    }
