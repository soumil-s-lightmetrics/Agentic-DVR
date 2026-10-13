"""
Graph state shared by the nodes.
"""
from typing import Any, Literal, TypedDict

from pydantic import BaseModel


class DateTimeRange(BaseModel):
    startTimeUTC: str | None = None
    endTimeUTC: str | None = None


class CrossQuestion(TypedDict):
    """A clarifying question a node asked the user."""
    node: str                           # node that asked it, and that clears it once answered
    question: str
    options: list[Any] | dict[str, list[Any]]     # strings or dicts (trips, violations); a dict when grouped, e.g. {"drivers": [...], "assets": [...]}
    purpose: str                       # state field the answer fills
    user_feedback: Any                  # the option the user picked, None until they answer


class NodeResult(TypedDict, total=False):
    """What every task node writes to last_result for the end orchestrator."""
    node: str
    status: Literal["done", "needs_input", "error"]
    message: str | None                 # reply, or the question for the user
    options: list[Any] | dict[str, Any]
    data: Any                           # trips, time slot, upload requests, created request...
    thinking: str | None                # time_slot: the place / event / clip time it found, for the frontend
    cross_question: CrossQuestion       # the node's question, also stored in the state by the node
    error: str | None
    recoverable: bool                   # error only: True if running the node again may work


class DriverInfo(BaseModel):
    driver_id: str | None = None
    driver_name: str | None = None

class AssetInfo(BaseModel):
    asset_id: str | None = None
    asset_name: str | None = None

class AgentState(TypedDict, total=False):
    # Request
    fleet_id: str
    client_id: str
    query: str
    pending_query: str | None                # request extract stopped on to ask something, the answer is added to it

    # Clarifying question waiting for / carrying the user's answer
    cross_question: CrossQuestion | None

    # Extracted parameters
    driver: DriverInfo
    asset: AssetInfo
    date_time: DateTimeRange | None

    # Routing and the reply to the user
    next_node: str
    reason: str
    message: str
    options: list[Any] | dict[str, Any]

    # location to find the time slot
    location: str | None = None

    # event mentioned by the user
    event_mentioned: str | None = None

    # actual event user was refering to, cross verified
    event_type: str | None = None

    # user said an event occurred, named or not ("a violation happened near Hosur")
    violation_mentioned: bool

    # exact moment the user pointed to (ISO UTC with Z), separate from the date_time search window
    event_time: str | None

    # trip the user referred to
    trip_id: str | None

    # "last 5 trips" -> 5, used by fetch_trips instead of a time window
    limit_to_latest: int | None

    # clip the user asked for
    clip_type: Literal["dvr_clip", "timelapse"] | None
    request_duration: float | None          # minutes
    threshold_min: float | None             # half the clip length, capped by the max for its clip_type

    # camera view and resolution of the clip ("road", "1280x720"...), None for the defaults
    video_format: str | None
    video_resolution: str | None

    # trips fetch_trip found for the current filters
    trips: list[dict]

    # full trip response (violations, route, upload requests) of trip_id, set by fetch_trip
    selected_trip: dict | None

    # set by time_slot: {"startTimeUTC", "endTimeUTC"} of the footage
    time_slot: dict | None

    # params last shown to the user for confirmation by the dvr node
    dvr_params: dict | None

    # filters before the latest message (set by extract), the planner compares them with the current ones
    previous_filters: dict[str, Any]

    # the planner decided the user changed the date time / driver / asset, so the trips are fetched again
    filters_changed: bool

    # Planner
    history: list[Any]                      # conversation since the filters last changed (BaseMessages)
    plan: list[str]                         # pending nodes in order, each node removes itself when done
    new_message: bool                       # set by extract, the planner hasn't planned for this user message yet
    planner_query_type: Literal["new_query", "ongoing_execution", "cross_query"] | None   # how the planner handled its last run
    user_intent: str | None                 # one sentence summary of what the user wants

    # End orchestrator
    last_result: NodeResult | None          # outcome of the node that just ran
    results: list[NodeResult]               # results of the nodes done so far in the current plan
    errors: list[dict[str, Any]]            # every node failure: {"node", "error", "recoverable"}
    retries: dict[str, int]                 # recoverable failures per node in the current plan
    status: Literal["success", "needs_input", "failed"] | None
    final_response: dict[str, Any] | None   # what is shown to the user when the workflow ends
