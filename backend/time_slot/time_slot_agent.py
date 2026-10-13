"""
Time-slot agent: reads the user's request, picks the location or event tool,
and returns the DVR time slot with a "thinking" message stating the place, event and clip time.
"""
import json
import os

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI

from logger import debug_logger
from time_slot.time_slot_event import get_time_slot_from_violation
from time_slot.time_slot_location import get_time_slot_from_place
from time_slot.time_slot_time import get_time_slot_from_time
from time_slot.time_slot_util import (
    get_trip_route,
    occurred_violation_types,
    parse_utc,
    simplify_response,
)

load_dotenv()

logger = debug_logger()

TOOLS = {t.name: t for t in (get_time_slot_from_place, get_time_slot_from_violation, get_time_slot_from_time)}

llm = ChatOpenAI(
    model="gpt-5.4-mini",
    temperature=0,
    api_key=os.environ.get("OPENAI_API_KEY")
)

SYSTEM_PROMPT = """You find the time slot of DVR footage the user wants from the selected trip, using the tools.

Trip context:
- Trip start: {start_utc} (UTC)
- Trip end:   {end_utc} (UTC)
- Violations in this trip (UTC):
{violations}

Rules:
- If the user identifies the moment by a PLACE, call get_time_slot_from_place, even if they also mention a violation.
- Otherwise, if they mention a violation/event (named or not), call get_time_slot_from_violation.
- Otherwise, if they give only a time (no place, no violation/event), call get_time_slot_from_time with that time.
  Only a time the user gave: never use the trip's start / end or any other time they didn't mention.
- Times the user gives are in UTC. Pass them as ISO 8601 in UTC, on the trip's date, e.g. "{example_time}".
- Set clip_type to "timelapse" if the user asked for a timelapse, otherwise "dvr_clip".
- If the user gives a clip length, set threshold_min to half of it in minutes. Otherwise leave the default.
- If the user asked for a DVR clip the threshold_min cannot go beyond 1.5 min (max DVR clip duration allowed is 3 min)
- If the user asked for a timelapse the threshold_min cannot go beyond 30 min (max timelapse duration allowed is 60 min)
- If the request has no place, no violation/event and no time, do not call a tool: ask the user briefly what moment of
  the trip they want.
- If a tool returns options or says the place/violation was not found, reply to the user in one or two short
  sentences asking them to choose or rephrase. Never invent times."""


def format_time(time_utc):
    """Epoch ms(1790309432038) / ISO string(2026-09-25T04:20:32Z") -> 'HH:MM:SS AM' in UTC."""
    if time_utc is None:
        return None
    return parse_utc(time_utc).strftime("%I:%M:%S %p")


def build_system_prompt(trip_info):
    start_dt = parse_utc(trip_info.get("startTimeUTC")) # start time of the trip
    end_dt = parse_utc(trip_info.get("endTimeUTC")) # end time of the trip
    violations = "\n".join(
        f"  - {v['event_type']} at {format_time(v['event_timeUTC'])}"
        for v in trip_info.get("violations", [])
    ) or "  (none)"
    return SYSTEM_PROMPT.format(
        start_utc=start_dt.strftime("%Y-%m-%d %I:%M:%S %p"),
        end_utc=end_dt.strftime("%Y-%m-%d %I:%M:%S %p"),
        violations=violations,
        example_time=start_dt.replace(hour=16, minute=20, second=0, microsecond=0).strftime("%Y-%m-%dT%H:%M:%SZ"),
    )


def success_thinking(args, result):
    """Place (if any), event name and time (if any), and the clip time, one per line."""
    lines = []
    if result.get("place"):
        lines.append(f"Place: {result['place']}")
    if result.get("violation"):
        event = result["violation"]
        lines.append(f"Event: {event['event_type']} at {format_time(event['event_timeUTC'])}")
    if result.get("basis") == "time":
        lines.append(f"Time: {format_time(result['time'])}")

    clip = "Timelapse" if args.get("clip_type") == "timelapse" else "DVR clip"
    start = format_time(result["timeSlot"]["startTimeUTC"].replace(" ", "T") + "Z")
    end = format_time(result["timeSlot"]["endTimeUTC"].replace(" ", "T") + "Z")
    lines.append(f"{clip}: {start} to {end} (UTC)")
    return "\n".join(lines)


def needs_input_thinking(tool_name, args, result):
    status = result.get("status")
    if status == "not_found":
        return f"I couldn't find a place called '{args.get('place_name')}'."
    if status == "not_on_route":
        return f"'{args.get('place_name')}' resolves to {result.get('closest_guess')}, but this trip didn't pass near it."
    if status == "no_violation":
        return f"{result.get('message')} Violations that did occur: {', '.join(result.get('options', [])) or 'none'}."
    if status == "multiple_violations":
        options = result["message"]["options"]
        name = args.get("violation_name") or "an event"
        return f"You mentioned {name}, which matches {len(options)} events in this trip, so I need you to pick one."
    if status == "not_in_trip":
        return (f"The trip wasn't running at {format_time(args.get('time_utc'))}; it ran from "
                f"{format_time(result['tripStart'])} to {format_time(result['tripEnd'])} (UTC).")
    if status == "error":
        return f"Looking up the time slot failed: {result.get('message')}"
    return "I couldn't tell which moment of the trip you meant."


def options(result):
    if result.get("status") == "multiple_violations":
        return result["message"]["options"]
    return result.get("options", [])


def run_time_slot_agent(user_query, trip , history=None, max_steps=3):
    """
    Finds the DVR time slot the user is asking for in the selected trip.

    Args:
        user_query (str): The user's latest message.
        trip (dict): Raw trip API response (with includeViolations and includePathInfo).
        history (list[BaseMessage] | None): Earlier turns of this conversation, e.g. when
            the user is picking from options the agent returned before.
        max_steps (int): Maximum number of tool calls before giving up.

    Returns:
        dict:
          - "status": "success" or "needs_input"
          - "time_slot": {"startTimeUTC", "endTimeUTC"} on success, else None
          - "thinking": on success, the place, event and clip time (DVR clip or timelapse); else why no
            slot was found. For the frontend
          - "message": question for the user when status is "needs_input", else None
          - "options": choices to show the user (violations or violation types), may be empty
          - "tool_result": the raw result of the last tool call, or None
    """

    trip_info = simplify_response(trip)
    route = get_trip_route(trip)["route"]

    # to avoid llm choosing certain arguments of the tools
    injected = {
        get_time_slot_from_place.name: {"trip_info": trip_info, "route": route},
        get_time_slot_from_violation.name: {"trip_info": trip_info},
        get_time_slot_from_time.name: {"trip_info": trip_info},
    }

    messages = [SystemMessage(build_system_prompt(trip_info)), *(history or []), HumanMessage(user_query)]
    llm_with_tools = llm.bind_tools(list(TOOLS.values()))

    last = None   # (tool_name, args, result) of the last tool call
    for _ in range(max_steps):
        ai_msg = llm_with_tools.invoke(messages)
        messages.append(ai_msg)

        if not ai_msg.tool_calls:
            break

        for call in ai_msg.tool_calls:
            tool_name, args = call["name"], call["args"]
            logger.debug(f"time_slot_agent: calling {tool_name} with {args}")
            try:
                result = TOOLS[tool_name].invoke({**args, **injected[tool_name]})
            except Exception as e:
                logger.error(f"time_slot_agent: {tool_name} failed: {e}")
                result = {"status": "error", "message": str(e)}
            logger.debug(f"time_slot_agent: {tool_name} returned {result}")
            messages.append(ToolMessage(json.dumps(result, default=str), tool_call_id=call["id"]))
            last = (tool_name, args, result)

            if result.get("status") == "success":
                return {
                    "status": "success",
                    "time_slot": result["timeSlot"],
                    "thinking": success_thinking(args, result),
                    "message": None,
                    "options": [],
                    "tool_result": result,
                }
    else:
        # ran out of steps on a tool call; ask the model for a reply to the user without tools
        ai_msg = llm.invoke(messages)
        messages.append(ai_msg)

    reply = ai_msg.content if isinstance(ai_msg, AIMessage) and ai_msg.content else None
    if last is None:
        return {
            "status": "needs_input",
            "time_slot": None,
            "thinking": "Your request didn't mention a place, a violation/event or a time, so I couldn't pick a moment of the trip.",
            "message": reply or "Which moment of the trip do you want footage of? You can name a place, a violation or a time.",
            "options": occurred_violation_types(trip_info),
            "tool_result": None,
        }

    tool_name, args, result = last
    return {
        "status": "needs_input",
        "time_slot": None,
        "thinking": needs_input_thinking(tool_name, args, result),
        "message": reply,
        "options": options(result),
        "tool_result": result,
    }
