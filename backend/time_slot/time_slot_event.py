"""
Time-slot tool for when the user identifies the footage by a violation/event,
with or without a time.
"""
from datetime import timedelta
from typing import Annotated, Literal

from langchain_core.tools import InjectedToolArg, tool

from logger import debug_logger
from time_slot.time_slot_util import (
    MAX_THRESHOLD_MIN,
    fuzzy_match_violation,
    get_time_slot,
    occurred_violation_types,
    parse_utc,
    to_utc_string,
)

logger = debug_logger()


def find_closest_violation_time(target_time, violation_list, max_diff=timedelta(hours=1)):
    """
    Returns the violation in violation_list whose time is closest to target_time
    (ISO 8601 string), or None if the closest is more than max_diff away.
    """
    target = parse_utc(target_time)

    closest = None
    smallest_diff = None

    for v in violation_list:
        diff = abs(target - parse_utc(v.get('event_timeUTC')))

        if smallest_diff is None or diff < smallest_diff:
            smallest_diff = diff
            closest = v

    if smallest_diff is None or smallest_diff > max_diff:
        return None

    return closest


def _success(violation, threshold_min, matched_by):
    """matched_by: 'only_occurrence' or 'closest_to_time', how the violation was picked."""
    return {
        "status": "success",
        "basis": "event",
        "matchedBy": matched_by,
        "violation": violation,
        "timeSlot": get_time_slot(violation.get('event_timeUTC'), threshold_min),
    }


def get_time_slot_info(trip_info, violation_name: str = None, violation_date_time: str = None, threshold_min: float = 1.5):
    """
    violation_name: matched event type (e.g. 'Traffic-Speed-Violated'), or None if the
    user only said that an event occurred.
    """
    event_count = trip_info.get('eventCount') or {}

    # if user hasn't mentioned any particular event, just mentioned that an event occurred then pull the list of the occurring events
    if violation_name is None:
        violation_count = event_count.get('total', 0)
    else:
        violation_count = event_count.get(violation_name, 0)

    logger.debug(f"get_time_slot_info: {violation_name} occurred {violation_count} times")

    if violation_count == 0:
        return {"status": "no_violation",
                "message": f"{violation_name or 'No violation'} did not occur in the trip.",
                "options": occurred_violation_types(trip_info)}

    violations = trip_info.get("violations", [])
    possible_violation_references = (
        violations if violation_name is None
        else [v for v in violations if v.get("event_type") == violation_name])

    if violation_count == 1 and possible_violation_references:
        return _success(possible_violation_references[0], threshold_min, "only_occurrence")

    if violation_date_time is None:
        return {"status": "multiple_violations",
                "message": {"mssg": f"multiple violations for {violation_name}",
                            "options": possible_violation_references}}

    # the planner node will extract the violation + time from the user selected option and call this function again with the violation_date_time argument.
    violation_date_time = to_utc_string(time_str=violation_date_time)

    closest_violation = find_closest_violation_time(
        target_time=violation_date_time,
        violation_list=possible_violation_references
    )

    if closest_violation is None:
        return {"status": "multiple_violations",
                "message": {"mssg": f"there was no {violation_name} around {violation_date_time}. Select from the rest of the occurences of the {violation_name}",
                            "options": possible_violation_references}}

    return _success(closest_violation, threshold_min, "closest_to_time")


# when event_name + time, only event_name, event_name is mentioned but event occurred mentioned along with or without time
@tool
def get_time_slot_from_violation(trip_info: Annotated[dict, InjectedToolArg], violation_name: str | None = None,
                                 violation_date_time: str | None = None, threshold_min: float = 1.5,
                                 clip_type: Literal["dvr_clip", "timelapse"] = "dvr_clip"):
    """
    Find the footage time slot for a violation/event that occurred in the selected trip.

    Purpose:
        Identifies which violation in the trip the user means and returns its
        time slot. The violation name is fuzzy-matched against the violation
        types present in the trip. If more than one violation could match,
        returns a list of options for the user to choose from.

    When to use:
        Use when the user says a violation or event occurred. Any of these
        cases counts, with or without a time:
        - Event named, no time:    "get the harsh braking clip"
        - Event named, with time:  "the overspeeding around 4:20 pm"
        - Event not named, no time:    "a violation occurred, get me the footage"
        - Event not named, with time:  "an event happened at 10 am, pull that video"
        Also use when the user picks one violation from options this tool
        returned earlier. Call it again with that option's name and time.

    When NOT to use:
        Do not use when the user identifies the moment by a LOCATION. This
        applies even if a violation is also mentioned:
        - "an event occurred near Silk Board"
        - "get me the DVR footage from when the truck was near Hosur"
        Use get_time_slot_from_place for these instead.

    Args:
        trip_info (dict): Injected by the agent, not chosen by the model.
            Simplified trip details for the selected trip.
        violation_name (str | None): Violation as the user phrased it, e.g.
            "speed violation". Pass None if the user only said that "an event"
            or "a violation" occurred without naming it.
        violation_date_time (str | None): When the violation happened, as an ISO
            8601 string in UTC, e.g. "2026-09-25T04:20:32.038Z". Pass None if
            the user gave no time.
        threshold_min (float): Minutes of footage on EACH side of the violation.
            Default 1.5, which gives a 3-minute clip. Capped at 1.5 for a DVR
            clip and 30 for a timelapse.
        clip_type (str): "timelapse" if the user asked for a timelapse,
            otherwise "dvr_clip".

    Returns:
        dict with a "status" key:
        - "success": a single violation was identified. Includes "violation",
          "matchedBy" ("only_occurrence" or "closest_to_time") and "timeSlot" ("startTimeUTC", "endTimeUTC" as
          "YYYY-MM-DD HH:MM:SS.mmm").
        - "multiple_violations": several candidates matched and no time (or no
          close-enough time) was given. "message" contains "mssg" and "options"
          (list of candidate violations). Show the options to the user, then
          call this tool again with their choice.
        - "no_violation": the named violation did not occur in this trip.
          "options" lists the violation types that did occur; offer these to
          the user.
    """
    threshold_min = min(threshold_min, MAX_THRESHOLD_MIN[clip_type])

    if violation_name is None:
        return get_time_slot_info(trip_info, None, violation_date_time, threshold_min)

    matched_violation = fuzzy_match_violation(violation_name, trip_info)
    if matched_violation is None:
        return {"status": "no_violation",
                "message": f"{violation_name} did not occur in the trip.",
                "options": occurred_violation_types(trip_info)}

    return get_time_slot_info(trip_info, matched_violation, violation_date_time, threshold_min)
