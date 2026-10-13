"""
Trip Q&A: answers the user's questions about the selected trip, using only the
simplified trip details (trip_info from simplify_response) as the source.
"""
import json
import os
from typing import Literal

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from datetime import datetime, timezone


from logger import debug_logger

load_dotenv()

logger = debug_logger()

llm = ChatOpenAI(
    model="gpt-5.4",
    temperature=0,
    api_key=os.environ.get("OPENAI_API_KEY")
)


def parse_utc(value):
    """Converts an epoch-milliseconds int or an ISO 8601 string ('...Z') to an aware UTC datetime."""
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(value / 1000, tz=timezone.utc)
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


OFF_TOPIC_ANSWER = "I can only answer questions about the selected trip."

SYSTEM_PROMPT = """You answer the user's questions about one vehicle trip, using ONLY the trip context below.

Trip context (JSON):
{trip_context}

Notes on the context:
- All times are in UTC.
- "duration" is in seconds. Units for "distance", "averageSpeed" and "maxSpeed" are not given, so do not add any.
- "eventCount" is the number of events of each type in the trip ("total" is all of them). A count of 0 means that
  event did not occur.
- Violation locations are only latitude/longitude. Do not turn them into place or road names.
- A null value means that detail is not available.

Classify the question and answer it:
- "answered": the question is about this trip and the answer is in the context. Answer briefly and only with facts
  from the context. You may count, compare or do simple arithmetic on the values.
- "not_in_context": the question is about this trip (or its driver, vehicle, route, events or footage) but the
  context does not have the answer, or the value is null. Tell the user briefly that you don't have that information
  for this trip. Do not guess or use general knowledge.
- "off_topic": the question is not about this trip. Do not answer it."""


class TripAnswer(BaseModel):
    category: Literal["answered", "not_in_context", "off_topic"] = Field(
        description="answered | not_in_context | off_topic")
    answer: str = Field(description="Reply to the user.")


def format_time(time_utc):
    """Epoch ms(1790309432038) / ISO string(2026-09-25T04:20:32Z") -> 'HH:MM:SS AM' in UTC."""
    if time_utc is None:
        return None
    return parse_utc(time_utc).strftime("%Y-%m-%d %I:%M:%S %p UTC")


def trip_context(trip_info):
    """
    trip_info with readable UTC times. startTime/endTime/timeZoneId are dropped because
    they are local times, and all times shown to the user are UTC.
    """
    context = {k: v for k, v in trip_info.items() if k not in ("startTime", "endTime", "timeZoneId")}
    context["startTimeUTC"] = format_time(trip_info.get("startTimeUTC"))
    context["endTimeUTC"] = format_time(trip_info.get("endTimeUTC"))
    context["violations"] = [
        {**v, "event_timeUTC": format_time(v.get("event_timeUTC"))}
        for v in trip_info.get("violations", [])
    ]
    return json.dumps(context, indent=2, default=str)


def answer_trip_query(user_query, trip_info, history=None):
    """
    Answers the user's question about the selected trip from trip_info only.

    Args:
        user_query (str): The user's latest message.
        trip_info (dict): Simplified trip details (output of simplify_response).
        history (list[BaseMessage] | None): Earlier turns of this conversation.

    Returns:
        dict:
          - "category": "answered", "not_in_context" (about the trip but not in trip_info)
            or "off_topic" (not about the trip)
          - "answer": reply to show the user
    """
    messages = [
        SystemMessage(SYSTEM_PROMPT.format(trip_context=trip_context(trip_info))),
        *(history or []),
        HumanMessage(user_query),
    ]
    result = llm.with_structured_output(TripAnswer).invoke(messages)
    logger.debug(f"answer_trip_query: {result.category} -> {result.answer}")

    answer = OFF_TOPIC_ANSWER if result.category == "off_topic" else result.answer
    return {"category": result.category, "answer": answer}
