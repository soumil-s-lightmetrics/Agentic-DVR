"""
Planner node: decides if the user's message continues the ongoing request or starts a new one, and if the
trip filters changed, then builds / keeps the ordered plan of nodes and routes to the first one.
"""
import json
from typing import Literal

from pydantic import BaseModel

from agent.llm import llm
from state import AgentState

Action = Literal["fetch_trip", "dvr", "trip_qna", "prev_upload_requests", "none"]

# nodes each action runs, when the trip is already known
ACTION_PLANS = {
    "fetch_trip" : ["fetch_trip"],
    "dvr" : ["time_slot", "dvr"],
    "trip_qna" : ["trip_qna"],
    "prev_upload_requests" : ["prev_upload_requests"],
    "none" : []
}

TRIP_NODES = {"time_slot", "dvr", "trip_qna", "prev_upload_requests"}     # need one trip to work on

MAX_HISTORY_TURNS = 10

NOTHING_TO_DO_MESSAGE = ("I can show trips, answer questions about a trip, list its previous upload requests, "
                         "or request a DVR clip or timelapse. What would you like to do?")

PLANNER_PROMPT = """
You plan how a fleet management DVR assistant handles the user's latest message.

Actions:
- fetch_trip: show / list / find trips matching filters (driver, asset, date, events, latest N trips).
- dvr: get a DVR clip or a timelapse of a trip's camera footage.
- trip_qna: answer a question about one trip (distance, speed, events, driver, start / end time...).
- prev_upload_requests: list the DVR clip / timelapse requests already made for a trip.
- none: the message needs none of these (a greeting, off-topic, or too unclear to act on).

Ongoing request:
- Pending plan (nodes still to run): {plan}
- Intent: {intent}
- Waiting for the user's answer to: {pending_question}

Trip filters before this message: {previous_filters}
Trip filters after this message:  {current_filters}
(Filters the message doesn't mention carry over from the earlier messages.)

Decide filters_changed, by comparing the filters before and after this message together with
the message and the conversation:
- true when the user now wants DIFFERENT trips: another driver, another asset, or a date / time
  range that points to other trips than before ("actually Ramesh's trip", "no, the 6th June
  trips", "show truck 12 instead").
- false when the filters are the same, or the message only narrows down or refines the same
  trips: e.g. giving the time of the moment within the selected trip ("around 4 PM") or
  answering a question about them.
- false when there were no filters before this message.

Decide query_type. A new message is not by itself a new query, it is often a follow-up:
- cross_query: the message is a follow-up to the ongoing request. It answers the pending question,
  picks an option, gives or changes a parameter of it (date, driver, asset, place, event, clip
  length, video format...), or confirms it. Set action to the ongoing request's action.
- new_query: the user has moved on from the ongoing request and asks for something else: a
  different action (e.g. a question about the trip in the middle of a DVR request), or the same
  action for an unrelated request. Set action to what the new request needs. Also new_query when
  there is no ongoing request.

user_intent: one sentence stating the requested action, its filters and the objective, built from
the message, the filters and the conversation. For a cross_query, state the updated intent of the
whole request, not just the follow-up.
e.g. "Create a 2 minute DVR clip of John's trip on 2026-06-05 at the harsh braking near Hosur."

Other extracted parameters: {params}

Conversation history:
{history}

User's message: {query}
"""


class PlannerDecision(BaseModel):

    filters_changed : bool

    query_type : Literal["new_query", "cross_query"]

    action : Action

    user_intent : str


def current_filters(state: AgentState):
    """The trip filters after the latest message, in the same shape as previous_filters."""
    driver, asset, date_time = state.get("driver"), state.get("asset"), state.get("date_time")
    return {
        "driver_name" : driver.driver_name if driver else None,
        "driver_id" : driver.driver_id if driver else None,
        "asset_name" : asset.asset_name if asset else None,
        "asset_id" : asset.asset_id if asset else None,
        "start_time" : date_time.startTimeUTC if date_time else None,
        "end_time" : date_time.endTimeUTC if date_time else None,
        "limit_to_latest" : state.get("limit_to_latest"),
        "trip_id" : state.get("trip_id")
    }


def other_params(state: AgentState):
    """The rest of what the Extract node put in the state, for the planner prompt."""
    return {
        "event" : state.get("event_type") or state.get("event_mentioned"),
        "event_time" : state.get("event_time"),
        "location" : state.get("location"),
        "clip_type" : state.get("clip_type"),
        "request_duration" : state.get("request_duration")
    }


def format_history(history):
    """Last MAX_HISTORY_TURNS turns as 'role: content' lines."""
    lines = []
    for turn in (history or [])[-MAX_HISTORY_TURNS:]:
        if isinstance(turn, dict):
            lines.append(f"{turn.get('role')}: {turn.get('content')}")
        else:
            lines.append(f"{turn.type}: {turn.content}")
    return "\n".join(lines) or "(none)"


def ensure_trip(plan, trip_ready):
    """Puts fetch_trip first when the first node needs a trip whose details aren't loaded yet."""
    if plan and plan[0] in TRIP_NODES and not trip_ready:
        return ["fetch_trip", *plan]
    return plan


def planner(state: AgentState):
    """
    Back from the end orchestrator with no new user message: continues the pending plan.
    New user message: the LLM compares the filters before and after it to decide if the user
    changed them, and decides if the message is a follow-up (cross_query, the pending plan carries on)
    or the user moved on to something else (new_query, a whole new plan).
    """
    plan = list(state.get("plan") or [])

    # a node finished and steps remain, nothing new from the user to plan for
    if not state.get("new_message"):
        return {
            "plan" : plan,
            "planner_query_type" : "ongoing_execution",
            "next_node" : plan[0] if plan else "end_orchestrator"
        }

    previous = state.get("previous_filters") or {}
    current = current_filters(state)
    cross_question = state.get("cross_question")
    prompt = PLANNER_PROMPT.format(
        plan = plan or "(none)",
        intent = state.get("user_intent") or "(none)",
        pending_question = cross_question["question"] if cross_question else "(none)",
        previous_filters = json.dumps(previous, default=str),
        current_filters = json.dumps(current, default=str),
        params = json.dumps(other_params(state), default=str),
        history = format_history(state.get("history")),
        query = state["query"]
    )
    decision = llm.with_structured_output(PlannerDecision).invoke(prompt)

    query_type = decision.query_type if plan else "new_query"
    filters_changed = decision.filters_changed and any(previous.values())

    result = {
        "planner_query_type" : query_type,
        "user_intent" : decision.user_intent,
        "filters_changed" : filters_changed,
        "new_message" : False       # planned for this message, the loop back from the end orchestrator only continues
    }

    trip_id = state.get("trip_id")
    if filters_changed:
        # the trips and the selected trip were for the old filters, and so was the conversation
        if trip_id == previous.get("trip_id"):
            trip_id = None              # unless this message named a new trip
        result.update(trips = [], trip_id = trip_id, selected_trip = None, time_slot = None, dvr_params = None,
                      history = (state.get("history") or [])[-1:])

    # fetch_trip loads the trip details, also for a trip ID the user typed
    selected_trip = None if filters_changed else state.get("selected_trip")
    trip_ready = bool(trip_id and selected_trip and selected_trip.get("tripId") == trip_id)

    # completed nodes already removed themselves, so continuing the plan never repeats them;
    # changed filters rebuild it, as the nodes already done worked on the old trips
    if query_type == "cross_query" and not filters_changed:
        new_plan = plan
    else:
        new_plan = list(ACTION_PLANS[decision.action])
    new_plan = ensure_trip(new_plan, trip_ready)

    result.update(plan = new_plan, next_node = new_plan[0] if new_plan else "end_orchestrator")

    # the user moved on, a question a node asked about the old request no longer needs an answer
    if query_type == "new_query" and cross_question and cross_question["node"] != "extract":
        result["cross_question"] = None
    if not new_plan:
        result.update(reason = "Nothing To Do", message = NOTHING_TO_DO_MESSAGE)
    return result


def route_after_planner(state: AgentState):
    """To the first pending node, or to the end orchestrator when nothing is pending."""
    plan = state.get("plan") or []
    return plan[0] if plan else "end_orchestrator"
