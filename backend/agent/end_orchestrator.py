"""
End orchestrator: reads the outcome of the node that just ran (last_result) and either sends control back
to the planner (nodes still pending, or a recoverable error to retry) or ends the turn with what the user
sees: the final result of the plan, the question a node asked, or the error that stopped it.
"""
from langchain_core.messages import AIMessage
from langgraph.graph import END

from state import AgentState

MAX_RETRIES = 2                                     # recoverable failures of one node before giving up on it
MAX_CLIP_MINUTES = {"clip": 3, "timelapse": 60}     # create_request's limits, for the confirmation card


def trip_summary(trip):
    """The parts of a full trip response the frontend shows for the selected trip."""
    return {
        "tripId" : trip.get("tripId"),
        "driverId" : trip.get("driverId"),
        "driverName" : trip.get("driverName"),
        "assetId" : trip.get("asset", {}).get("assetId"),
        "startTimeUTC" : trip.get("startTimeUTC"),
        "endTimeUTC" : trip.get("endTimeUTC")
    }


def node_payload(state: AgentState, result):
    """What the frontend needs to show a node's result, besides its message."""
    node, data, status = result.get("node"), result.get("data"), result.get("status")

    if node == "fetch_trip":
        payload = {"trips": state.get("trips") or []}
        if state.get("selected_trip"):
            payload["selected_trip"] = trip_summary(state["selected_trip"])
        return payload

    if node == "time_slot":
        return {"time_slot": state.get("time_slot"), "thinking": result.get("thinking")}

    if node == "trip_qna":
        return {"category": (data or {}).get("category")}

    if node == "prev_upload_requests":
        return {"upload_requests": data or []}

    if node == "dvr":
        if status == "done":
            return {"dvr_request": data}
        params = state.get("dvr_params")
        if params and status == "needs_input" and result.get("cross_question"):
            trip = state.get("selected_trip") or {}
            return {"dvr_confirmation": {**params, "driverName": trip.get("driverName"),
                                         "maxMinutes": MAX_CLIP_MINUTES.get(params.get("type"), 3)}}
    return {}


def finish(state: AgentState, status, message, payload=None, **update):
    """Ends the turn with the response for the user, which is also kept in the history."""
    history = list(state.get("history") or [])
    if message:
        history.append(AIMessage(message))
    return {
        "status" : status,
        "final_response" : {"status": status, "message": message, **(payload or {})},
        "history" : history,
        "last_result" : None,
        "next_node" : "END",
        **update
    }


def end_orchestrator(state: AgentState):
    """
    done        - nodes pending: back to the planner. Plan finished: stores the final result and ends.
    needs_input - the question reaches the user: stored in cross_question unless the node already stored it,
                  and the turn ends. The node stays in the plan and runs again with the answer.
    error       - recoverable and the node is next in the plan: back to the planner, which runs it again
                  (up to MAX_RETRIES). Otherwise: stores the error with a failed status, ends and shows it.
    """
    result = state.get("last_result")
    plan = list(state.get("plan") or [])
    results = list(state.get("results") or [])
    retries = dict(state.get("retries") or {})

    if result is None:
        # the planner had nothing to run, its own reply is the answer
        if not plan:
            return finish(state, "success", state.get("message"), results = [])
        result = {"node": plan[0], "status": "error", "recoverable": False,
                  "error": "The step finished without reporting a result."}

    node, status = result.get("node"), result.get("status")

    if status == "needs_input":
        update = {}
        question = result.get("cross_question")
        if question and not state.get("cross_question"):
            update["cross_question"] = question
        question = question or {}
        payload = {
            **node_payload(state, result),
            "options" : result.get("options") or question.get("options") or [],
            "question" : {"node": node, "purpose": question.get("purpose"), "question": question.get("question")}
                         if question else None
        }
        message = result.get("message") or question.get("question")
        # the user's answer comes as the next message, the request it answers is kept with it
        return finish(state, "needs_input", message, payload, pending_query = state.get("query"), **update)

    if status == "done":
        retries.pop(node, None)
        if plan and plan[0] == node:
            plan = plan[1:]             # a node that didn't remove itself would run forever
        results.append(result)
        if plan:
            return {"plan": plan, "results": results, "retries": retries, "last_result": None, "next_node": "planner"}

        payload = {"results": [{"node": r["node"], "message": r.get("message")} for r in results]}
        for r in results:
            payload.update(node_payload(state, r))
        # "Selected trip ..." only matters when picking the trip was all there was to do
        message = "\n\n".join(r["message"] for r in results
                               if r.get("message") and (r["node"] != "fetch_trip" or len(results) == 1))
        return finish(state, "success", message, payload, plan = plan, results = [], retries = retries,
                      pending_query = None)

    # failed, or a status the orchestrator doesn't know: never treated as done, the error is kept
    error = {
        "node" : node,
        "error" : result.get("error") or result.get("message") or f"Unknown status {status!r}",
        "recoverable" : status == "error" and bool(result.get("recoverable"))
    }
    errors = [*(state.get("errors") or []), error]
    retries[node] = retries.get(node, 0) + 1

    # only a node in the plan can be run again; extract / planner failures end the turn
    if error["recoverable"] and plan and plan[0] == node and retries[node] <= MAX_RETRIES:
        return {"errors": errors, "retries": retries, "last_result": None, "next_node": "planner"}

    retries.pop(node, None)
    message = f"Sorry, I couldn't complete your request. {error['error']}"
    return finish(state, "failed", message, {"error": error}, errors = errors, retries = retries)


def route_after_end_orchestrator(state: AgentState):
    """Back to the planner while nodes are pending or a node is retried, otherwise the turn ends."""
    return "planner" if state.get("next_node") == "planner" else END
