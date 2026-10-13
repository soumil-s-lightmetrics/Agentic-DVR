"""
Task nodes of the graph: fetch_trip, time_slot, trip_qna, prev_upload_requests and dvr.
Each one writes a NodeResult to last_result for the end orchestrator, removes itself from the plan when
done, and stays in it (storing its question) when it needs the user's answer.
"""
from datetime import timedelta
from functools import wraps
import json

import httpx
import openai

from agent.extract import asset_options, driver_options
from agent.lookups import get_asset, get_drivers
from agent.planner import TRIP_NODES
from fetch_trips.fetch_trips import fetch_trip_details, fetch_trips_with_expiry, parse_utc
from logger import debug_logger
from prev_upload_requests.get_prev_requests import get_trip_upload_requests
from question_trips.QnA import answer_trip_query
from state import AgentState
from time_slot.time_slot_agent import run_time_slot_agent
from time_slot.time_slot_util import simplify_response
from upload_request.create_request import REQUEST_TYPES, create_dvr_request


# ── node helpers ──────────────────────────────────────────────────────────────────────────────

logger = debug_logger()


def is_recoverable(error: Exception):
    """Network problems, timeouts, rate limits and server errors may work on a retry; the rest won't."""
    if isinstance(error, httpx.HTTPStatusError):
        return error.response.status_code == 429 or error.response.status_code >= 500
    return isinstance(error, (httpx.TransportError, openai.APIConnectionError,
                              openai.RateLimitError, openai.InternalServerError))


def graph_node(name):
    """Runs the node, turning an exception into an error result for the end orchestrator."""
    def wrap(fn):
        @wraps(fn)
        def node(state: AgentState):
            try:
                return fn(state)
            except Exception as e:
                logger.error(f"{name} failed: {e}", exc_info=True)
                return {"last_result": {"node": name, "status": "error", "error": str(e),
                                        "recoverable": is_recoverable(e)}}
        return node
    return wrap


def own_answer(state: AgentState, node):
    """The user's answer to the question this node asked: the option picked, or the message typed. None if no question."""
    question = state.get("cross_question")
    if not question or question["node"] != node:
        return None
    answer = question.get("user_feedback")
    if answer is None:
        return state["query"]
    return answer if isinstance(answer, str) else json.dumps(answer, default=str)


def clear_own_question(state: AgentState, node):
    question = state.get("cross_question")
    return {"cross_question": None} if question and question["node"] == node else {}


def done(state: AgentState, node, message=None, data=None, **update):
    """The node finished: it removes itself from the plan and hands its result to the end orchestrator."""
    plan = list(state.get("plan") or [])
    if plan and plan[0] == node:
        plan = plan[1:]
    return {
        "plan" : plan,
        "last_result" : {"node": node, "status": "done", "message": message, "data": data},
        **clear_own_question(state, node),
        **update
    }


def ask(state: AgentState, node, question, options=None, purpose="", **update):
    """The node needs the user to pick / answer something. It stays in the plan and runs again with the answer."""
    cross_question = {"node": node, "question": question, "options": options or [],
                      "purpose": purpose, "user_feedback": None}
    return {
        "cross_question" : cross_question,
        "last_result" : {"node": node, "status": "needs_input", "message": question,
                         "options": options or [], "cross_question": cross_question},
        **update
    }


def needs_input(state: AgentState, node, message, **update):
    """The node can't go on with the current parameters; the user's next message changes them. Stays in the plan."""
    return {
        "last_result" : {"node": node, "status": "needs_input", "message": message},
        **clear_own_question(state, node),
        **update
    }


def failed(node, error):
    """A problem running the node again won't fix."""
    return {"last_result": {"node": node, "status": "error", "error": error, "recoverable": False}}


def earlier_history(state: AgentState):
    """History without the latest user message, which the nodes pass as the query."""
    return list(state.get("history") or [])[:-1]


# ── fetch_trip ────────────────────────────────────────────────────────────────────────────────

MAX_TRIPS_TO_PICK = 10          # more trips than this: ask the user to narrow them down first
MAX_PICK_WINDOW_DAYS = 2


def trip_label(trip):
    return f"{trip['driverName']} · {trip['assetId']} · {trip['startTimeUTC']}"


def select_trip(state: AgentState, trip_id, driver_id, trips):
    """Loads the full details of the chosen trip for the nodes after this one."""
    details = fetch_trip_details(state["fleet_id"], state["client_id"], driver_id, trip_id)
    return done(state, "fetch_trip", message=f"Selected trip {trip_id}.", data=details,
                trip_id=trip_id, selected_trip=details, trips=trips)


def narrow_down(state: AgentState, trips, driver_id, asset_id, has_window):
    """Too many trips to pick one: asks for a driver / asset and / or a shorter time range, None if both are set."""
    date_time = state.get("date_time")
    asks, purposes, options = [], [], {}

    if not (driver_id or asset_id):
        asks.append("pick a driver or an asset")
        purposes.append("driver_name or asset_name")
        # trips only have the asset ID, the user picks by name
        asset_names = {a["assetId"]: a["assetName"] for a in get_asset(fleet_id=state["fleet_id"], client_id=state["client_id"])}
        drivers = {t["driverId"]: t["driverName"] for t in trips if t["driverId"] and t["driverName"] not in (None, "—")}
        assets = {t["assetId"]: asset_names.get(t["assetId"]) or t["assetId"] for t in trips if t["assetId"] != "—"}
        options = {
            "drivers" : driver_options([{"driverId": i, "driverName": n} for i, n in sorted(drivers.items(), key=lambda d: d[1])]),
            "assets" : asset_options([{"assetId": i, "assetName": n} for i, n in sorted(assets.items(), key=lambda a: a[1])])
        }

    window_too_long = not has_window or (
        parse_utc(date_time.endTimeUTC) - parse_utc(date_time.startTimeUTC) > timedelta(days=MAX_PICK_WINDOW_DAYS))
    if window_too_long:
        asks.append(f"limit the time range to {MAX_PICK_WINDOW_DAYS} days")
        purposes.append("date_time")

    if not asks:
        return None
    question = f"I found {len(trips)} trips, too many to pick from. Please {' or '.join(asks)}."
    return ask(state, "fetch_trip", question, options=options, purpose=", ".join(purposes), trips=trips)


@graph_node("fetch_trip")
def fetch_trip_node(state: AgentState):
    """
    Fetches the trips matching the filters. When a later node needs one trip (DVR, QnA, previous
    requests), makes sure exactly one is selected: picks it if only one matches, asks the user to
    pick one if 2-10 match, and to narrow the filters down first if more than 10 match.
    """
    node = "fetch_trip"
    plan = list(state.get("plan") or [])
    needs_one_trip = any(n in TRIP_NODES for n in plan[1:])
    driver, asset, date_time = state.get("driver"), state.get("asset"), state.get("date_time")
    driver_id = driver.driver_id if driver else None
    asset_id = asset.asset_id if asset else None
    has_window = bool(date_time and date_time.startTimeUTC and date_time.endTimeUTC)
    limit_to_latest = state.get("limit_to_latest")
    trip_id = state.get("trip_id")

    # the user picked one of the trips this node offered (extract normally copies it to trip_id too)
    question, answer = state.get("cross_question"), own_answer(state, node)
    if answer and question.get("purpose") == "trip_id":
        picked = next((t for t in state.get("trips") or [] if answer in (t["tripId"], trip_label(t))), None)
        if picked:
            trip_id = picked["tripId"]

    # the user gave a trip ID, or picked one of the trips offered earlier
    if trip_id:
        known = next((t for t in state.get("trips") or [] if t["tripId"] == trip_id), None)
        trip_driver = known["driverId"] if known else driver_id
        if trip_driver:
            return select_trip(state, trip_id, trip_driver, state.get("trips") or [])
        if not (has_window or limit_to_latest):
            drivers = driver_options(get_drivers(fleet_id=state["fleet_id"], client_id=state["client_id"]))
            return ask(state, node, f"Which driver drove trip {trip_id}?", options=drivers, purpose="driver_name")
        # otherwise look for it in the trips matching the filters

    if not (has_window or limit_to_latest):
        return needs_input(state, node, "Which trips should I look at? Give a date or date range, or ask for the latest trips.")

    event_type = state.get("event_type")
    try:
        trips = fetch_trips_with_expiry(
            state["fleet_id"], state["client_id"],
            start_time = date_time.startTimeUTC if has_window else None,
            end_time = date_time.endTimeUTC if has_window else None,
            driver_ids = [driver_id] if driver_id else None,
            asset_ids = [asset_id] if asset_id else None,
            events = {event_type: None} if event_type else None,
            limit_to_latest = None if has_window else limit_to_latest
        )
    except ValueError as e:
        # a missing or future date range, the user has to change it
        return needs_input(state, node, str(e))

    if trip_id:
        trip = next((t for t in trips if t["tripId"] == trip_id), None)
        if trip:
            return select_trip(state, trip_id, trip["driverId"], trips)
        return needs_input(state, node, f"I couldn't find trip {trip_id} for the selected filters. "
                                        "Which driver or date is it from?", trips=trips)

    if not trips:
        return needs_input(state, node, "No trips found for the selected filters. Try a different date range, driver or asset.", trips=[])

    # only listing trips
    if not needs_one_trip:
        return done(state, node, message=f"Found {len(trips)} trips.", data=trips, trips=trips)

    if len(trips) == 1:
        return select_trip(state, trips[0]["tripId"], trips[0]["driverId"], trips)

    if len(trips) > MAX_TRIPS_TO_PICK:
        narrow = narrow_down(state, trips, driver_id, asset_id, has_window)
        if narrow:
            return narrow

    options = [{"tripId": t["tripId"], "label": trip_label(t)} for t in trips]
    return ask(state, node, f"I found {len(trips)} trips. Which one do you mean?",
               options=options, purpose="trip_id", trips=trips)


# ── time_slot ─────────────────────────────────────────────────────────────────────────────────

@graph_node("time_slot")
def time_slot_node(state: AgentState):
    """Finds the moment of the selected trip the footage should cover, asking the user when it's unclear."""
    node = "time_slot"
    trip = state.get("selected_trip")
    if not trip:
        return failed(node, "No trip is selected to find the time slot in.")

    query = state["query"]
    question = state.get("cross_question")
    if own_answer(state, node) and question.get("user_feedback") is not None:
        query = f"{query}\nThe user picked: {own_answer(state, node)}"

    result = run_time_slot_agent(query, trip, history=earlier_history(state))

    if result["status"] == "success":
        update = done(state, node, message=result["thinking"], data=result["time_slot"], time_slot=result["time_slot"])
        update["last_result"]["thinking"] = result["thinking"]
        return update

    update = ask(state, node, result["message"] or result["thinking"], options=result["options"],
                 purpose="the moment of the trip the footage should cover")
    update["last_result"]["thinking"] = result["thinking"]
    return update


# ── trip_qna ──────────────────────────────────────────────────────────────────────────────────

@graph_node("trip_qna")
def trip_qna_node(state: AgentState):
    """Answers the user's question about the selected trip."""
    node = "trip_qna"
    trip = state.get("selected_trip")
    if not trip:
        return failed(node, "No trip is selected to answer the question about.")

    result = answer_trip_query(state["query"], simplify_response(trip), history=earlier_history(state))
    return done(state, node, message=result["answer"], data={"category": result["category"]})


# ── prev_upload_requests ──────────────────────────────────────────────────────────────────────

@graph_node("prev_upload_requests")
def prev_upload_requests_node(state: AgentState):
    """Lists the DVR clip / timelapse requests already made for the selected trip."""
    node = "prev_upload_requests"
    trip = state.get("selected_trip")
    if not trip:
        return failed(node, "No trip is selected to list the upload requests of.")

    upload_requests, message = get_trip_upload_requests(trip)
    return done(state, node, message=message, data=upload_requests or [])


# ── dvr ───────────────────────────────────────────────────────────────────────────────────────

CLIP_TYPES = {"dvr_clip": "clip", "timelapse": "timelapse"}     # extract / time slot names -> create_request names


def iso_utc(time_utc):
    """'2026-06-05 16:20:00' (time slot) -> '2026-06-05T16:20:00Z' (create_request)."""
    time_utc = time_utc.replace(" ", "T")
    return time_utc if time_utc.endswith("Z") else time_utc + "Z"


@graph_node("dvr")
def dvr_node(state: AgentState):
    """
    Shows the DVR clip / timelapse params for the user to confirm, then places the request.
    The user can change the video format / resolution instead of confirming, it asks again with them.
    """
    node = "dvr"
    trip, time_slot = state.get("selected_trip"), state.get("time_slot")
    if not trip or not time_slot:
        return failed(node, "A DVR request needs a selected trip and its time slot.")

    params = {
        "tripId" : trip["tripId"],
        "driverId" : trip["driverId"],
        "assetId" : trip.get("asset", {}).get("assetId"),
        "type" : CLIP_TYPES[state.get("clip_type") or "dvr_clip"],
        "clipStart" : iso_utc(time_slot["startTimeUTC"]),
        "clipEnd" : iso_utc(time_slot["endTimeUTC"])
    }
    if state.get("video_format"):
        params["videoFormat"] = state["video_format"]
    if state.get("video_resolution"):
        params["videoResolution"] = state["video_resolution"]

    result = create_dvr_request(state["fleet_id"], state["client_id"], params, user_response=own_answer(state, node))

    if result["status"] == "submitted":
        request_type = REQUEST_TYPES[result["params"]["type"]]
        return done(state, node, message=f"{request_type} request placed. Request ID: {result['uploadRequestId']}",
                    data=result, dvr_params=result["params"])

    if result["status"] == "invalid":
        # the clip times come from the time slot, so it runs again with the user's correction
        plan = list(state.get("plan") or [])
        return needs_input(state, node, result["message"], plan=["time_slot", *plan], dvr_params=result["params"])

    return ask(state, node, result["message"], options=["confirm"],
               purpose="confirm the DVR request params", dvr_params=result["params"])
