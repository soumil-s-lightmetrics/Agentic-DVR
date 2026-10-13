"""
DVR assistant graph.

START -> extract -> planner -> fetch_trip | time_slot | trip_qna | prev_upload_requests | dvr -> end_orchestrator
end_orchestrator -> planner while nodes are pending (or a node is retried), otherwise END.
extract and planner hand their questions / errors straight to the end orchestrator, which delivers them.

There are no interrupts: every user message is one run on the thread's checkpointed state, and a question a
node asked is answered by the next message.
"""
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import END, START, StateGraph

from agent.end_orchestrator import end_orchestrator, route_after_end_orchestrator
from agent.extract import extract
from agent.nodes import (
    dvr_node,
    fetch_trip_node,
    graph_node,
    prev_upload_requests_node,
    time_slot_node,
    trip_qna_node,
)
from agent.planner import planner, route_after_planner
from state import AgentState

# state classes stored in the checkpoint, allowed to be loaded back
CHECKPOINT_TYPES = [("state", "DriverInfo"), ("state", "AssetInfo"), ("state", "DateTimeRange")]

TASK_NODES = {
    "fetch_trip" : fetch_trip_node,
    "time_slot" : time_slot_node,
    "trip_qna" : trip_qna_node,
    "prev_upload_requests" : prev_upload_requests_node,
    "dvr" : dvr_node
}


@graph_node("extract")
def extract_node(state: AgentState):
    # the last turn ended on a question (from extract or a node), the message answers it: both together
    # are the request, for extract and for the nodes after it that read the query (time slot, QnA)
    if state.get("pending_query"):
        state = {**state, "query": f"{state['pending_query']}\n{state['query']}"}

    update = extract(state)

    # extract stopped to ask / tell the user something, the end orchestrator delivers it
    if update.get("next_node") == "END":
        update.update(query = state["query"], pending_query = state["query"])
        update["last_result"] = {"node": "extract", "status": "needs_input", "message": update.get("message"),
                                 "options": update.get("options") or []}
        if update.get("cross_question"):
            update["last_result"]["cross_question"] = update["cross_question"]
    else:
        update.update(query = state["query"], pending_query = None)
    return update


planner_node = graph_node("planner")(planner)


def route_after_extract(state: AgentState):
    return "end_orchestrator" if state.get("last_result") else "planner"


def route_from_planner(state: AgentState):
    # a planner failure goes to the end orchestrator instead of the first pending node
    return "end_orchestrator" if state.get("last_result") else route_after_planner(state)


def create_graph(checkpointer=None):
    builder = StateGraph(AgentState)

    builder.add_node("extract", extract_node)
    builder.add_node("planner", planner_node)
    builder.add_node("end_orchestrator", end_orchestrator)
    for name, node in TASK_NODES.items():
        builder.add_node(name, node)
        builder.add_edge(name, "end_orchestrator")

    builder.add_edge(START, "extract")
    builder.add_conditional_edges("extract", route_after_extract, ["planner", "end_orchestrator"])
    builder.add_conditional_edges("planner", route_from_planner, [*TASK_NODES, "end_orchestrator"])
    builder.add_conditional_edges("end_orchestrator", route_after_end_orchestrator, ["planner", END])

    checkpointer = checkpointer or MemorySaver(serde=JsonPlusSerializer(allowed_msgpack_modules=CHECKPOINT_TYPES))
    return builder.compile(checkpointer=checkpointer)
