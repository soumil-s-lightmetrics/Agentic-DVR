"""
Planner: extracts the parameters from the user's query, decides which nodes run and in what
order, makes sure the trip nodes get exactly one selected trip, and passes each node's output
on to the next node.
"""
import json
import os
from datetime import datetime, timezone
from typing import Literal

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from fetch_trips.fetch_trips import fetch_trip_details, parse_utc
from logger import debug_logger
from state import AgentState, DvrParams, DvrType, NodeResult, VideoFormat, VideoResolution

load_dotenv()

logger = debug_logger()

llm = ChatOpenAI(
    model="gpt-5.4",
    temperature=0,
    api_key=os.environ.get("OPENAI_API_KEY")
)

Node = Literal["fetch_trips", "find_time_slot", "trip_qna", "prev_upload_requests", "create_dvr_request"]
TRIP_NODES = {"find_time_slot", "trip_qna", "prev_upload_requests", "create_dvr_request"}   # need one selected trip
MAX_TRIPS_TO_PICK = 10    # more trips than this: ask for a single date / driver before asking to pick one

SYSTEM_PROMPT = """You plan how to handle the user's message in a DVR assistant for a fleet's trips.

Nodes:
- fetch_trips: lists the fleet's trips for a date range or the latest N trips, filtered by driver, asset and events.
- find_time_slot: finds the moment of the selected trip the user wants footage of (a place, an event or a time).
- trip_qna: answers a question about the selected trip (distance, speed, events, driver, times...).
- prev_upload_requests: lists the DVR clip / timelapse requests already made for the selected trip.
- create_dvr_request: confirms and places a DVR clip or timelapse request for the selected trip.

Current state:
{context}

Current UTC time: {now}

"plan" - the nodes to run, in order:
- Show / list trips: ["fetch_trips"]
- DVR clip or timelapse request: ["find_time_slot", "create_dvr_request"]
- Question about a trip: ["trip_qna"]
- Previous upload requests of a trip: ["prev_upload_requests"]
- If the user is answering the pending question (picking an option, giving a date / driver / trip, confirming or
  changing the DVR request params), return the current plan unchanged.
- Only add fetch_trips when the user wants to see trips; trips needed by other nodes are fetched automatically.
- Return [] if the message needs none of these nodes.

Parameters - only what THIS message says, leave the rest empty:
- drivers: driver names or IDs. asset_ids: asset IDs. trip_id: a trip ID, e.g. from "[Trip: t_...]".
- events: event types, with the exact count if given ("2 traffic speed violations" -> count 2). Use the event
  type names seen in the trips when one matches.
- start_time / end_time: ISO 8601 UTC, e.g. "2026-09-28T00:00:00Z". A single day -> 00:00:00 to 23:59:59.
- limit_to_latest: "last 5 trips" -> 5.
- dvr_type: "clip" for a DVR clip / video, "timelapse" for a timelapse.
- video_format, video_resolution, clip_start, clip_end (ISO 8601 UTC): only when the user changes the params of
  the DVR request shown to them.
- confirmed: true only if the user agrees to the DVR request params shown to them."""


class EventFilter(BaseModel):
    type: str
    count: int | None = None                           # exact count, None for any count


class PlannerOutput(BaseModel):
    plan: list[Node]
    drivers: list[str] = []
    asset_ids: list[str] = []
    events: list[EventFilter] = []
    start_time: str | None = None
    end_time: str | None = None
    limit_to_latest: int | None = None
    trip_id: str | None = None
    dvr_type: DvrType | None = None
    video_format: VideoFormat | None = None
    video_resolution: VideoResolution | None = None
    clip_start: str | None = None
    clip_end: str | None = None
    confirmed: bool = False


def ask(s: AgentState, message, options=None):
    """The Planner's own question to the user, shown by the End Orchestrator."""
    s.last_result = NodeResult(node="planner", status="needs_input", message=message, options=options or [])


def has_window(s: AgentState):
    return bool(s.limit_to_latest or (s.start_time and s.end_time))


def single_day(s: AgentState):
    return bool(s.start_time and s.end_time) and parse_utc(s.start_time).date() == parse_utc(s.end_time).date()


def context(s: AgentState):
    """What the LLM needs to know about the state to plan the user's message."""
    return json.dumps({
        "filters": {"start_time": s.start_time, "end_time": s.end_time, "limit_to_latest": s.limit_to_latest,
                    "driver_ids": s.driver_ids, "asset_ids": s.asset_ids, "events": s.events},
        "trips": [
            {"tripId": t["tripId"], "driver": t["driverName"], "assetId": t["assetId"],
             "start": t["startTimeUTC"], "events": [e["type"] for e in t["events"]]}
            for t in s.trips[:20]
        ],
        "trips_count": len(s.trips),
        "selected_trip_id": s.selected_trip_id,
        "current_plan": s.plan,
        "pending_question": s.pending_question.model_dump(include={"node", "message", "options"})
        if s.pending_question else None,
        "dvr_params": s.dvr_params.model_dump() if s.dvr_params else None,
    }, default=str)


def take_node_output(s: AgentState):
    """
    Passes the output of the node that just finished on to the state the next node reads.
    Returns True if it was fetch_trips.
    """
    result, s.last_result = s.last_result, None
    if not result or result.status != "done":
        return False

    if result.node == "fetch_trips":
        s.trips = result.data or []
        if s.selected_trip_id not in {t["tripId"] for t in s.trips}:
            s.selected_trip_id, s.selected_trip = None, None
        return True

    if result.node == "find_time_slot" and s.selected_trip:
        trip = s.selected_trip
        s.dvr_params = DvrParams(
            tripId=trip["tripId"],
            driverId=trip["driverId"],
            assetId=trip.get("asset", {}).get("assetId"),
            type=s.dvr_type or "clip",
            clipStart=result.data["startTimeUTC"].replace(" ", "T") + "Z",
            clipEnd=result.data["endTimeUTC"].replace(" ", "T") + "Z",
        )
    return False


def resolve_driver_ids(drivers, trips):
    """Driver names -> driver IDs, using the drivers in the fetched trips; unknown names are used as IDs."""
    ids = []
    for driver in drivers:
        name = driver.lower()
        match = next((t["driverId"] for t in trips
                      if name == (t["driverId"] or "").lower() or name in (t["driverName"] or "").lower()), None)
        ids.append(match or driver)
    return ids


def merge_filters(s: AgentState, out: PlannerOutput):
    """Applies the filters in the user's message to the state. Returns True if any of them changed."""
    new = {}
    if out.drivers:
        new["driver_ids"] = resolve_driver_ids(out.drivers, s.trips)
    if out.asset_ids:
        new["asset_ids"] = out.asset_ids
    if out.events:
        new["events"] = {e.type: e.count for e in out.events}
    if out.start_time and out.end_time:
        new.update(start_time=out.start_time, end_time=out.end_time, limit_to_latest=None)
    elif out.limit_to_latest:
        new.update(limit_to_latest=out.limit_to_latest, start_time=None, end_time=None)

    changed = any(getattr(s, k) != v for k, v in new.items())
    for k, v in new.items():
        setattr(s, k, v)
    return changed


def plan_turn(s: AgentState):
    """New user message: extracts its parameters, updates the state and builds the plan."""
    out = llm.with_structured_output(PlannerOutput).invoke([
        SystemMessage(SYSTEM_PROMPT.format(context=context(s), now=datetime.now(timezone.utc).isoformat())),
        HumanMessage(s.user_query or ""),
    ])
    logger.info(f"planner: {out}")

    filters_changed = merge_filters(s, out)
    if filters_changed:
        s.selected_trip_id, s.selected_trip = None, None
    if out.trip_id and out.trip_id != s.selected_trip_id:
        s.selected_trip_id, s.selected_trip = out.trip_id, None

    if out.dvr_type:
        s.dvr_type = out.dvr_type
    if s.dvr_params:
        changes = {"type": out.dvr_type, "videoFormat": out.video_format, "videoResolution": out.video_resolution,
                   "clipStart": out.clip_start, "clipEnd": out.clip_end}
        s.dvr_params = s.dvr_params.model_copy(update={k: v for k, v in changes.items() if v})
    s.dvr_user_response = "confirm" if out.confirmed else None

    if out.plan != s.plan:
        s.results = []          # a new plan, so earlier results don't belong in its answer
    s.plan = list(out.plan)

    # Fetch trips only when the trips in the state don't match the user's filters
    if filters_changed or not s.trips:
        if s.plan and "fetch_trips" not in s.plan:
            s.plan.insert(0, "fetch_trips")
    elif s.plan == ["fetch_trips"]:
        s.last_result = NodeResult(node="fetch_trips", status="done", data=s.trips,
                                   message=f"Showing the {len(s.trips)} trips already fetched.")
    elif "fetch_trips" in s.plan:
        s.plan.remove("fetch_trips")


def select_trip(s: AgentState, just_fetched):
    """Makes sure exactly one trip is selected, asking the user to narrow down / pick one if needed."""
    if s.selected_trip_id:
        trip = next((t for t in s.trips if t["tripId"] == s.selected_trip_id), None)
        if not trip:
            ask(s, f"I couldn't find trip {s.selected_trip_id} in the fetched trips. "
                   "Which date and driver is it from?")
            return
    elif len(s.trips) == 1:
        trip = s.trips[0]
    elif not s.trips:
        if just_fetched:
            ask(s, "No trips found for the selected filters. Try a different date range, driver or asset.")
        elif has_window(s):
            s.plan.insert(0, "fetch_trips")
        else:
            ask(s, "Which trips should I look at? Give a date or date range, or ask for the latest trips.")
        return
    elif len(s.trips) > MAX_TRIPS_TO_PICK and not single_day(s):
        ask(s, f"I found {len(s.trips)} trips. Which date is the trip from?")
        return
    elif len(s.trips) > MAX_TRIPS_TO_PICK and not s.driver_ids:
        ask(s, f"I found {len(s.trips)} trips. Which driver was driving?")
        return
    else:
        ask(s, "Which trip do you mean? Pick one from the list.", options=[
            {"tripId": t["tripId"], "label": f"{t['driverName']} · {t['assetId']} · {t['startTimeUTC']}"}
            for t in s.trips
        ])
        return

    s.selected_trip_id = trip["tripId"]
    s.selected_trip = fetch_trip_details(s.fleet_id, s.client_id, trip["driverId"], trip["tripId"])


def prepare_next_node(s: AgentState, just_fetched):
    """Checks plan[0] has what it needs; otherwise asks the user or adds the node that provides it."""
    if s.last_result:
        return
    if not s.plan:
        ask(s, "I can show trips, answer questions about a trip, list its previous upload requests, "
               "or request a DVR clip or timelapse. What would you like to do?")
        return

    if s.plan[0] == "fetch_trips":
        if not has_window(s):
            ask(s, "Which trips should I look at? Give a date or date range, or ask for the latest trips.")
        return

    if not s.selected_trip:
        select_trip(s, just_fetched)
        if s.last_result or s.plan[0] == "fetch_trips":
            return

    if s.plan[0] == "create_dvr_request" and (not s.dvr_params or s.dvr_params.tripId != s.selected_trip_id):
        s.plan.insert(0, "find_time_slot")


def planner(state: AgentState):
    """
    New user message: extracts its parameters and builds the plan.
    Back from the End Orchestrator with steps remaining: passes the finished node's output on.
    Then makes sure plan[0] has what it needs; route_after_planner sends control to it.
    """
    continuing = bool(state.last_result and state.last_result.status == "done" and state.plan)
    s = state.model_copy(deep=True)
    s.response = None

    try:
        just_fetched = take_node_output(s)
        if not continuing:
            plan_turn(s)
        prepare_next_node(s, just_fetched)
    except Exception as e:
        logger.error(f"planner failed: {e}", exc_info=True)
        s.last_result = NodeResult(node="planner", status="error",
                                   message="Something went wrong while handling your request. Please try again.")

    logger.info(f"planner: plan={s.plan} selected_trip={s.selected_trip_id} "
                f"next={'end_orchestrator' if s.last_result else s.plan[0]}")
    return {field: getattr(s, field) for field in AgentState.model_fields}


def route_after_planner(state: AgentState):
    """To the End Orchestrator when the Planner asked the user something itself, else to the next node."""
    return "end_orchestrator" if state.last_result else state.plan[0]
