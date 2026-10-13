"""
Extract node: pulls the parameters (driver, asset, date time, trip, event, clip settings...) out of the
user's message, resolves the driver / asset IDs, and carries the filters over from the earlier messages.
"""
from collections import Counter
from datetime import datetime, timezone
from typing import Literal

from langchain_core.messages import HumanMessage
from pydantic import BaseModel

from agent.llm import llm
from agent.lookups import get_asset, get_drivers
from state import AgentState, AssetInfo, DateTimeRange, DriverInfo

EXTRACT_PROMPT = """
You are an information extraction system for a fleet management DVR assistant.
Users ask for DVR clips or timelapses of a vehicle's camera footage.

Your task is to extract structured information from the user's query.
A query usually contains only some of the fields below. Extract only the fields that are
present in the query and leave every other field as None.

Current date/time (UTC): {now}

Extract the following fields:

1. driver_name
   * The driver's name as written in the query, e.g. "John", "Sunil Kumar".
   * Return None if no driver is mentioned.

2. asset_name
   * The vehicle / asset the user refers to: its name, number or plate,
     e.g. "Truck 12", "Bus-042", "KA01AB1234".
   * Copy it as written. Return None if no vehicle is mentioned.

3. trip_id
   * A trip identifier, copied exactly character for character,
     e.g. "t_2LNEWUFYS4I7DAAAZ4JIDYDS6A_8CT1z".
   * Trip IDs do not follow one fixed pattern. They are usually a single long token of
     letters, digits, underscores or hyphens, and often follow words like "trip" or
     "trip id", or appear inside "[Trip: ...]".
   * Never shorten, re-case or correct it, and never treat a driver or asset name as a trip ID.
   * Return None if no trip ID is present.

4. event_name and violation_mentioned
   * violation_mentioned: true when the user says a safety event / violation occurred, whether
     or not they name it, e.g. "the harsh braking", "a violation happened at 10am",
     "something happened near Hosur". Otherwise false.
   * event_name: the specific event the user names, in the user's own words,
     e.g. "harsh braking", "overspeeding", "using the phone while driving".
     Do not map it to a standard event type, that is checked later.
   * If the user only refers to an event in general ("an event", "a violation", "an incident",
     "something happened"), set violation_mentioned = true and event_name = None.
   * Return event_name None if no specific event is named.

5. location
   * The place the footage should be from: an address, street, area, city, landmark or
     business, e.g. "the Walmart on 5th Street", "Downtown Chicago", "the toll plaza on I-90".
   * Copy the place as written. Return None if no place is mentioned.

6. clip_type
   * "dvr_clip" when the user wants a video clip, footage, recording or DVR.
   * "timelapse" when the user asks for a timelapse.
   * Return None if the query doesn't say which one.

7. request_duration
   * The length of footage the user asks for, in minutes,
     e.g. "a 2 minute clip" -> 2, "30 seconds of video" -> 0.5, "one and a half minutes" -> 1.5.
   * Return None if no length is given. Don't derive it from the start and end times.

8. video_format
   * The camera view the user wants, exactly one of:
       road              - road-facing / front / outward camera, dashcam view
       driver            - driver-facing / cabin / inward camera
       sideBySide        - both views next to each other
       pictureInPicture  - one view shown inside the other
       separate          - both views as separate videos
   * Return the exact value above. Return None if no view is mentioned.

9. video_resolution
   * Exactly one of: 320x180, 640x360, 1280x720, 1920x1080.
   * Map common names: "180p" -> 320x180, "360p" -> 640x360, "720p" / "HD" -> 1280x720,
     "1080p" / "Full HD" -> 1920x1080.
   * Return None if no resolution is mentioned.

10. available_duration
   * How much recorded footage the user says is available, as written, e.g. "only 5 minutes".
   * Return None unless the user states it explicitly.

11. date_time (startTimeUTC, endTimeUTC)
   * The time window the footage / trip should be from, in UTC, formatted as
     YYYY-MM-DDTHH:MM:SS. Times in the query are UTC unless the user gives a timezone,
     in which case convert them to UTC.
   * Explicit range ("5th June between 9 and 10 AM") -> its start and end.
   * Date with no time ("5th June") -> 00:00:00 to 23:59:59 of that date.
   * Date AND a single clock time with no window ("5th June 9AM") -> 1 hour before to
     1 hour after that time (08:00:00 to 10:00:00).
   * Clock time with no date ("at 9AM") -> apply it to today.
   * Date without a year ("16th June") -> the current year, or the previous year if that
     date would be in the future.
   * Relative references are resolved against the current date/time above:
     "today" / "yesterday" -> that whole day, "last 3 days" -> now minus 3 days to now,
     "last week" -> now minus 7 days to now, "last month" -> now minus 30 days to now.
   * If only one bound is given ("after 5PM yesterday"), the other bound is the start or
     end of that same day.
   * Return None for both if the query has no date or time information.

   Time of day (when the query names a part of the day instead of a clock time, e.g.
   "in the morning", "after sunrise", "during rush hour", "late at night"):
   * Use your general understanding of what time of day the phrase conventionally refers
     to, the way a person would in everyday conversation. Reference scale (24-hour clock):
       midnight ~ 00:00   early morning ~ 04:00-06:00   sunrise/dawn ~ 06:00
       morning ~ 06:00-12:00   noon ~ 12:00   afternoon ~ 12:00-17:00
       evening ~ 17:00-20:00   sunset/dusk ~ 18:00-19:00   night ~ 20:00-24:00
     Interpret other phrases (rush hour, lunch hour, post-midnight, before dawn...) by
     placing them on the same scale.
   * Build a window of roughly 1-4 hours around your interpretation, unless the wording
     clearly means a single moment ("right after sunrise").
   * "before <phrase>" -> start of the day to the start of the phrase's window.
     "after <phrase>"  -> end of the phrase's window to the end of the day, unless another
     bound is given. "during <phrase>" or no preposition -> the window itself.
   * Apply the window to the date that is otherwise resolved, or today if there is none.
   * If an explicit clock time is also given ("morning, around 9AM"), use the clock time.
   * If the phrase is ambiguous, don't guess: treat it as if no time of day was given.

12. event_time
   * The exact moment the user points to with a single clock time, in UTC, formatted as
     YYYY-MM-DDTHH:MM:SSZ, e.g. "the overspeeding at 4:20 PM on 5th June" -> "2026-06-05T16:20:00Z".
   * Resolve its date the same way as date_time (today if no date is given).
   * It is set alongside date_time, not instead of it: "5th June 9AM" gives date_time
     08:00:00 to 10:00:00 AND event_time "2026-06-05T09:00:00Z".
   * Return None if the query has only a date, a range or a part of the day, and no single clock time.

13. limit_to_latest
   * Set when the user asks for the latest / most recent trips instead of a time window:
       "the latest trip", "the most recent trip", "last trip"        -> 1
       "the last 3 trips", "most recent 10 trips"                    -> that number
       "latest trips", "recent trips" (plural, no count)             -> 50
   * "last" + a unit of time ("last week", "last 5 days", "past month") is a time window,
     not a count: use date_time for it and leave limit_to_latest None.
     "last" + trip/trips or a bare count ("last 5 trips") is a count.
   * limit_to_latest and date_time are never both set. If the query also gives a date, a range,
     a relative duration or a part of the day, use date_time and leave limit_to_latest None.
   * Return None if the query doesn't ask for the latest trips.

Clarifying questions
* The query may be followed by a clarifying question asked to the user and the option they
  selected. Use the selected option, exactly as written, as the value of the field named in
  its purpose (e.g. purpose "driver_name" -> driver_name = the selected option), and keep
  extracting every other field from the original query.
* If the purpose names more than one field, put the answer in the field it belongs to: a name
  from the "drivers" options -> driver_name, from the "assets" options -> asset_name, a date or
  time -> date_time. The answer may be typed by the user instead of picked from the options.

General Rules
1. Extract only information present in the query (or the user's answer to a clarifying
   question). Never invent values.
2. Return None for every field that is not present.
3. Copy names, IDs and places as written, without correcting their spelling.
4. video_format, video_resolution and clip_type must be one of their listed values or None.
"""

class Extract(BaseModel):

    driver_name : str | None = None

    asset_name : str | None = None

    date_time : DateTimeRange

    event_name : str | None = None

    violation_mentioned : bool = False

    event_time : str | None = None

    clip_type : Literal["dvr_clip", "timelapse"] | None = None

    request_duration : float | None = None

    video_format : str | None = None

    available_duration : str | None = None

    video_resolution : str | None = None

    location : str | None = None

    trip_id : str | None = None

    limit_to_latest : int | None = None


EVENT_COUNTS = {'total': 15, 'Traffic-Speed-Violated': 3, 'Cornering': 0, 'Traffic-STOP-Sign-Violated': 0, 'Harsh-Braking': 0, 'Harsh-Acceleration': 0, 'Tail-Gating-Detected': 1, 'Lane-Drift-Found': 0, 'Distracted-Driving': 1, 'MaxSpeedExceeded': 0, 'Drowsy-Driving-Detected': 1, 'Forward-Collision-Warning': 5, 'Cellphone-Distracted-Driving': 0, 'Smoking-Distracted-Driving': 0, 'Drinking-Distracted-Driving': 0, 'Unbuckled-Seat-Belt': 0, 'Lizard-Eye-Distracted-Driving': 0, 'Roll-Over-Detected': 0, 'Texting-Distracted-Driving': 0, 'Traffic-Light-Violated': 0, 'Driver-Fatigue-Detected': 4}

EVENT_TYPES = [e for e in EVENT_COUNTS if e != "total"]

MAX_CLIP_MIN = {"dvr_clip": 3, "timelapse": 60}     # longest clip allowed for each clip_type

MAP_EVENT_PROMPT = """
Map the safety event the user mentioned to one of the supported event types.

Supported event types:
{event_types}

* Pick the event type that means the same as the user's words, including synonyms and
  paraphrases, e.g. "tailgating" -> Tail-Gating-Detected, "on the phone" -> Cellphone-Distracted-Driving,
  "ran a red light" -> Traffic-Light-Violated.
* If more than one fits, pick the most specific one.
* Return the event type exactly as written in the list above.
* Return None if none of them match. Don't treat general words like "accident" as an event.

User's event: {event_name}
"""


class EventMatch(BaseModel):

    event_type : Literal[tuple(EVENT_TYPES)] | None = None


def map_event(event_name: str):
    """Maps the event in the user's words to one of EVENT_TYPES, None if none of them match."""
    prompt = MAP_EVENT_PROMPT.format(event_types="\n".join(EVENT_TYPES), event_name=event_name)
    return llm.with_structured_output(EventMatch).invoke(prompt).event_type


def driver_options(drivers):
    """
    Drivers as options that carry their ID, so picking one of two drivers with the same name picks
    that driver. The label shows the ID only when the name isn't unique.
    """
    counts = Counter(d["driverName"] for d in drivers)
    return [{"driverId": d["driverId"], "name": d["driverName"],
             "label": d["driverName"] if counts[d["driverName"]] == 1 else f"{d['driverName']} ({d['driverId']})"}
            for d in drivers if d["driverName"]]


def asset_options(assets):
    """Assets as options that carry their ID, like driver_options."""
    counts = Counter(a["assetName"] for a in assets)
    return [{"assetId": a["assetId"], "name": a["assetName"],
             "label": a["assetName"] if counts[a["assetName"]] == 1 else f"{a['assetName']} ({a['assetId']})"}
            for a in assets if a["assetName"]]


def picked_option(cross_question, id_key):
    """The driver / asset option (by id_key) the user picked for the pending question, None if they didn't pick one."""
    if not cross_question or not isinstance(cross_question.get("user_feedback"), str):
        return None
    options = cross_question.get("options") or []
    groups = options.values() if isinstance(options, dict) else [options]
    return next((o for group in groups for o in group
                 if isinstance(o, dict) and o.get(id_key) == cross_question["user_feedback"]), None)


def resolve_one(matches: list[dict], searched: str, kind: str, name_key: str, id_key: str):
    """
    Returns (id, None) when exactly one match is found, otherwise
    (None, response) where response is what the node should return.
    """
    # an exact name match wins, so picking "John" from the options
    # doesn't match "John Smith" again on the next pass
    exact = [m for m in matches if searched.lower() in ((m[name_key] or "").lower(), (m[id_key] or "").lower())]
    if len(exact) == 1:
        return exact[0][id_key], None

    if not matches:
        return None, {
            "next_node" : "END",
            "reason" : f"No Matching {kind.title()}",
            "message" : f"No {kind} found matching '{searched}'. Please check the name and try again.",
            "options" : []
        }

    if len(matches) > 1:
        options = driver_options(matches) if kind == "driver" else asset_options(matches)
        question = f"Multiple {kind}s match '{searched}'. Which one did you mean?"
        return None, {
            "next_node" : "END",
            "reason" : f"Multiple Matching {kind.title()}s",
            "message" : f"{question} Please Select an option",
            "options" : options,
            "cross_question" : {
                "node" : "extract",
                "question" : question,
                "options" : options,
                "purpose" : f"{kind}_name",
                "user_feedback" : None
            }
        }

    return matches[0][id_key], None


def extract(state : AgentState):

    query = state["query"]

    cross_question = state.get("cross_question")

    cross_question_feedback = ""

    if cross_question:
        cq_node = cross_question["node"]
        question = cross_question["question"]
        options = cross_question["options"]
        purpose = cross_question["purpose"] # to fill information in which field of the state
        user_response = cross_question["user_feedback"]

        # no option picked: the user typed a message instead, which may answer the question
        # or ask for something else, the planner decides which
        if user_response is None:
            cross_question_feedback = f"""
            Node {cq_node} asked a clarifying question {question} from the user
            for the purpose {purpose}. The options provided to the user were {options}.
            The user didn't pick an option and replied with the query below instead, which may
            answer the question or ask for something else"""
        else:
            cross_question_feedback = f"""
            Node {cq_node} asked a clarifying question {question} from the user
            for the purpose {purpose}. The options provided to the user were {options}
            and the user selected the option {user_response}"""


    final_prompt = f"""
    {EXTRACT_PROMPT.format(now=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S"))}
    user's query : {query}
    {cross_question_feedback}"""

    structured_llm = llm.with_structured_output(Extract)

    output = structured_llm.invoke(final_prompt)

    event_name = output.event_name
    location = output.location
    # the clip settings carry over too, a "confirm" reply doesn't repeat them
    clip_type = output.clip_type or state.get("clip_type")
    video_format = output.video_format or state.get("video_format")
    video_resolution = output.video_resolution or state.get("video_resolution")
    request_duration = output.request_duration
    event_time = output.event_time
    violation_mentioned = output.violation_mentioned or bool(event_name)

    # a follow-up only names what it changes, the other filters carry over from the earlier messages
    prev_driver = state.get("driver") or DriverInfo()
    prev_asset = state.get("asset") or AssetInfo()
    prev_date_time = state.get("date_time")
    prev_limit = state.get("limit_to_latest")

    driver_name = output.driver_name or prev_driver.driver_name
    asset_name = output.asset_name or prev_asset.asset_name

    # a driver / asset picked from the options comes with its ID, used as it is
    picked_driver = picked_option(cross_question, "driverId")
    picked_asset = picked_option(cross_question, "assetId")
    if picked_driver:
        driver_name = picked_driver["name"]
    if picked_asset:
        asset_name = picked_asset["name"]
    trip_id = output.trip_id or state.get("trip_id")

    # a trip picked from the options: only its ID is taken, the driver / asset / time in its label
    # describe that trip, they don't change the filters
    picked_trip = picked_option(cross_question, "tripId")
    if picked_trip:
        driver_name, asset_name, trip_id = prev_driver.driver_name, prev_asset.asset_name, picked_trip["tripId"]

    # a time window and limit_to_latest are never both set, the one in this message wins
    if picked_trip:
        date_time, limit_to_latest = prev_date_time, prev_limit
    elif output.date_time is not None and output.date_time.startTimeUTC is not None:
        date_time, limit_to_latest = output.date_time, None
    elif output.limit_to_latest:
        date_time, limit_to_latest = None, output.limit_to_latest
    else:
        date_time, limit_to_latest = prev_date_time, prev_limit

    # a trip_id is one trip on its own; otherwise fetch_trips needs a time window or limit_to_latest,
    # and a driver or asset so it doesn't return too many trips to pick from
    has_window = date_time is not None and date_time.startTimeUTC is not None
    has_trips_range = has_window or limit_to_latest
    has_driver_or_asset = driver_name or asset_name
    if not trip_id and not (has_trips_range and has_driver_or_asset):
        missing, purposes, options = [], [], []
        if not has_trips_range:
            missing.append("a date or date range (or ask for the latest trips)")
            purposes.append("date_time")
        if not has_driver_or_asset:
            missing.append("a driver or an asset")
            purposes.append("driver_name or asset_name")
            options = {
                "drivers" : driver_options(get_drivers(fleet_id = state['fleet_id'], client_id = state['client_id'])),
                "assets" : asset_options(get_asset(fleet_id = state['fleet_id'], client_id = state['client_id']))
            }
        question = f"Please provide {' and '.join(missing)}, or a trip ID."
        return {
            "next_node" : "END",
            "reason" : "Missing Params",
            "message" : question,
            "options" : options,
            "cross_question" : {
                "node" : "extract",
                "question" : question,
                "options" : options,
                "purpose" : ", ".join(purposes),
                "user_feedback" : None
            }
        }

    # map the event in the user's words to a supported event type
    event_type = None
    if event_name:
        event_type = map_event(event_name)
        if event_type is None:
            question = f"'{event_name}' doesn't match any supported event. Which event did you mean?"
            return {
                "next_node" : "END",
                "reason" : "No Matching Event",
                "message" : f"{question} Please Select an option",
                "options" : EVENT_TYPES,
                "cross_question" : {
                    "node" : "extract",
                    "question" : question,
                    "options" : EVENT_TYPES,
                    "purpose" : "event_name",
                    "user_feedback" : None
                }
            }

    # only look up what the user actually mentioned, a None name would match everything
    # reuse the ID found earlier when the driver didn't change
    if picked_driver:
        driver_id = picked_driver["driverId"]
    else:
        driver_id = prev_driver.driver_id if driver_name == prev_driver.driver_name else None
    if driver_name and driver_id is None:
        drivers = get_drivers(
            fleet_id = state['fleet_id'], client_id = state['client_id'], driver_name = driver_name
        )
        driver_id, response = resolve_one(drivers, driver_name, "driver", "driverName", "driverId")
        if response:
            return response

    if picked_asset:
        asset_id = picked_asset["assetId"]
    else:
        asset_id = prev_asset.asset_id if asset_name == prev_asset.asset_name else None
    if asset_name and asset_id is None:
        assets = get_asset(
            fleet_id = state['fleet_id'], client_id = state['client_id'], asset_name = asset_name
        )
        asset_id, response = resolve_one(assets, asset_name, "asset", "assetName", "assetId")
        if response:
            return response

    # the time slot tools take half the clip length on each side of the moment
    threshold_min = None
    if request_duration:
        threshold_min = min(request_duration, MAX_CLIP_MIN[clip_type or "dvr_clip"]) / 2

    # the planner compares these with the filters after this message to decide if the user changed them
    previous_filters = {
        "driver_name" : prev_driver.driver_name,
        "driver_id" : prev_driver.driver_id,
        "asset_name" : prev_asset.asset_name,
        "asset_id" : prev_asset.asset_id,
        "start_time" : prev_date_time.startTimeUTC if prev_date_time else None,
        "end_time" : prev_date_time.endTimeUTC if prev_date_time else None,
        "limit_to_latest" : prev_limit,
        "trip_id" : state.get("trip_id")
    }

    history = list(state.get("history") or [])
    history.append(HumanMessage(query))

    result = {
        "driver" : DriverInfo(driver_id = driver_id, driver_name=driver_name),
        "asset" : AssetInfo(asset_id = asset_id, asset_name= asset_name),
        "date_time" : date_time,
        "location" : location,
        "event_mentioned" : event_name,
        "event_type" : event_type,
        "violation_mentioned" : violation_mentioned,
        "event_time" : event_time,
        "trip_id" : trip_id,
        "limit_to_latest" : limit_to_latest,
        "clip_type" : clip_type,
        "video_format" : video_format,
        "video_resolution" : video_resolution,
        "request_duration" : request_duration,
        "threshold_min" : threshold_min,
        "history" : history,
        "previous_filters" : previous_filters,
        "new_message" : True
    }

    # only clear a cross question this node asked, one asked by another
    # node stays in the state so that node can read the user's answer
    if cross_question and cross_question["node"] == "extract":
        result["cross_question"] = None

    return result
