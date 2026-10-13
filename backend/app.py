"""
DVR assistant backend: serves the frontend and runs the graph over a WebSocket (/chat).

No interrupts / Commands: every message from the user is one graph run on the thread's checkpointed state.
When a node asks the user something, the run ends with that question, and the user's answer is the next
message (the option they picked goes into cross_question.user_feedback).

Client -> server
  {"type": "load_fleet", "fleet_id", "client_id"}
  {"type": "query", "thread_id", "query"}
  {"type": "answer", "thread_id", "value", "label"}               picked option of the pending question
  {"type": "dvr_confirm", "thread_id", "video_format", "video_resolution", "clip_start", "clip_end"}
  {"type": "dvr_cancel", "thread_id"}
Server -> client
  {"type": "fleet_loaded", "fleet_id", "client_id", "drivers", "assets", "events"}
  {"type": "progress", "thread_id", "node", "label"}              a step of the run finished
  {"type": "response", "thread_id", "status", "message", ...}     final_response of the end orchestrator
  {"type": "error", "message"}
"""
import json
import os

import httpx
from dotenv import load_dotenv

load_dotenv()

from flask import Flask, jsonify, send_from_directory
from flask_cors import CORS
from flask_sock import Sock
from langchain_core.messages import AIMessage

from agent.extract import EVENT_TYPES
from agent.lookups import get_asset, get_drivers
from graph import create_graph
from logger import access_logger, debug_logger, log_request, websocket_logger

ws_logger = websocket_logger()
d_logger = debug_logger()
a_logger = access_logger()

BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
RECURSION_LIMIT = 50

# one graph with an in-memory checkpointer, shared by every connection; threads are kept apart by thread_id
graph = create_graph()

STEP_LABELS = {
    "extract" : "Read your request",
    "planner" : "Planned the steps",
    "fetch_trip" : "Looked up the trips",
    "time_slot" : "Found the time slot",
    "trip_qna" : "Looked into the trip",
    "prev_upload_requests" : "Checked the previous requests",
    "dvr" : "Prepared the DVR request"
}

app = Flask(__name__, static_folder=os.path.join(BACKEND_DIR, "images"), static_url_path="/images")
CORS(app)
sock = Sock(app)


@app.after_request
def log_http_request(response):
    log_request(response, a_logger)
    return response


@app.route("/")
def get_ui():
    return send_from_directory(BACKEND_DIR, "DVR_frontend.html")


@app.route("/health")
def health():
    return jsonify({"status": "running"})


def send(ws, payload):
    ws.send(json.dumps(payload, default=str))


def thread_config(thread_id):
    return {"configurable": {"thread_id": thread_id}, "recursion_limit": RECURSION_LIMIT}


def thread_state(thread_id):
    snapshot = graph.get_state(thread_config(thread_id))
    return snapshot.values if snapshot else {}


def load_fleet(fleet_id, client_id):
    """Drivers, assets and event types of the fleet, for the @ filters in the frontend."""
    drivers = get_drivers(fleet_id=fleet_id, client_id=client_id)
    assets = get_asset(fleet_id=fleet_id, client_id=client_id)
    return {
        "drivers" : [{"driverId": d["driverId"], "driverName": d["driverName"] or d["driverId"]} for d in drivers],
        "assets" : [{"assetId": a["assetId"], "assetName": a["assetName"] or a["assetId"]} for a in assets],
        "events" : EVENT_TYPES
    }


def run_turn(ws, thread_id, fleet, update):
    """One graph run for a user message, streaming its steps, then sending the end orchestrator's response."""
    config = thread_config(thread_id)
    turn = {
        "fleet_id" : fleet["fleet_id"],
        "client_id" : fleet["client_id"],
        "last_result" : None,
        "final_response" : None,
        **update
    }
    for chunk in graph.stream(turn, config, stream_mode="updates"):
        for node in chunk:
            if node in STEP_LABELS:
                send(ws, {"type": "progress", "thread_id": thread_id, "node": node, "label": STEP_LABELS[node]})

    response = thread_state(thread_id).get("final_response") or {
        "status": "failed", "message": "Something went wrong while handling your request. Please try again."}
    send(ws, {"type": "response", "thread_id": thread_id, **response})


def answer_update(thread_id, value, label):
    """The user picked an option of the pending question: it goes to cross_question.user_feedback."""
    question = thread_state(thread_id).get("cross_question")
    query = label or (value if isinstance(value, str) else json.dumps(value))
    # a trip label (driver · asset · start time) would read as filters and a clip time, the ID is enough
    if question and question.get("purpose") == "trip_id" and isinstance(value, str):
        query = f"Trip {value}"
    if not question:
        return {"query": query}
    return {"query": query, "cross_question": {**question, "user_feedback": value}}


def dvr_confirm_update(thread_id, payload):
    """
    The user submitted the confirmation card: their format / resolution / clip times replace the shown ones,
    and the dvr node places the request with them.
    """
    state = thread_state(thread_id)
    question = state.get("cross_question")
    if not question or question["node"] != "dvr":
        return None

    update = {"query": "confirm", "cross_question": {**question, "user_feedback": "confirm"}}
    if payload.get("video_format"):
        update["video_format"] = payload["video_format"]
    if payload.get("video_resolution"):
        update["video_resolution"] = payload["video_resolution"]
    if payload.get("clip_start") and payload.get("clip_end"):
        update["time_slot"] = {"startTimeUTC": payload["clip_start"], "endTimeUTC": payload["clip_end"]}
    return update


@sock.route("/chat")
def chat_socket(ws):
    fleet = {"fleet_id": None, "client_id": None}
    ws_logger.info("Client connected")

    try:
        while True:
            raw = ws.receive()
            if raw is None:
                break
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                send(ws, {"type": "error", "message": "Invalid JSON payload."})
                continue

            msg_type, thread_id = payload.get("type"), payload.get("thread_id")
            ws_logger.info(f"ws message: {msg_type} | thread: {thread_id}")

            try:
                if msg_type == "load_fleet":
                    fleet_id = (payload.get("fleet_id") or "").strip()
                    client_id = (payload.get("client_id") or "").strip()
                    if not fleet_id or not client_id:
                        send(ws, {"type": "error", "message": "Enter both the fleet ID and the account."})
                        continue
                    try:
                        data = load_fleet(fleet_id, client_id)
                    except httpx.HTTPStatusError as e:
                        d_logger.error(f"Failed to load fleet {fleet_id} (client {client_id}): {e.response.status_code}")
                        send(ws, {"type": "error", "message": f'Failed to load fleet "{fleet_id}" under account "{client_id}".'})
                        continue
                    except httpx.HTTPError as e:
                        d_logger.error(f"Failed to reach the API for fleet {fleet_id}: {e}")
                        send(ws, {"type": "error", "message": "Could not reach the fleet API. Please try again."})
                        continue
                    fleet = {"fleet_id": fleet_id, "client_id": client_id}
                    ws_logger.info(f"Fleet loaded: {fleet_id} | client: {client_id} | "
                                   f"{len(data['drivers'])} drivers | {len(data['assets'])} assets")
                    send(ws, {"type": "fleet_loaded", **fleet, **data})
                    continue

                if msg_type not in ("query", "answer", "dvr_confirm", "dvr_cancel"):
                    send(ws, {"type": "error", "message": f"Unknown message type: {msg_type}"})
                    continue
                if not thread_id:
                    send(ws, {"type": "error", "message": "Missing thread_id."})
                    continue
                if not (fleet["fleet_id"] and fleet["client_id"]):
                    send(ws, {"type": "response", "thread_id": thread_id, "status": "failed",
                              "message": "No fleet is loaded yet. Please load the fleet and try again."})
                    continue

                if msg_type == "query":
                    query = (payload.get("query") or "").strip()
                    if not query:
                        send(ws, {"type": "error", "message": "Empty query."})
                        continue
                    run_turn(ws, thread_id, fleet, {"query": query})

                elif msg_type == "answer":
                    run_turn(ws, thread_id, fleet, answer_update(thread_id, payload.get("value"), payload.get("label")))

                elif msg_type == "dvr_confirm":
                    update = dvr_confirm_update(thread_id, payload)
                    if update is None:
                        send(ws, {"type": "response", "thread_id": thread_id, "status": "failed",
                                  "message": "There is no DVR request waiting for confirmation."})
                        continue
                    run_turn(ws, thread_id, fleet, update)

                elif msg_type == "dvr_cancel":
                    # drops the pending request without a run: nothing left to do for it
                    message = "DVR request cancelled."
                    history = [*thread_state(thread_id).get("history", []), AIMessage(message)]
                    graph.update_state(thread_config(thread_id),
                                       {"plan": [], "cross_question": None, "dvr_params": None, "results": [],
                                        "history": history, "next_node": "END"},
                                       as_node="end_orchestrator")
                    send(ws, {"type": "response", "thread_id": thread_id, "status": "success", "message": message})

            except Exception as e:
                ws_logger.error(f"Error handling '{msg_type}' message: {e}", exc_info=True)
                send(ws, {"type": "error", "message": "Something went wrong on our end. Please try again."})

    except Exception as e:
        ws_logger.error(f"WebSocket connection error: {e}", exc_info=True)
    finally:
        ws_logger.info("Client disconnected.")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    debug = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    app.run(host="0.0.0.0", port=port, debug=debug)
