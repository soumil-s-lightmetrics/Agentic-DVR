"""
Create DVR request: confirms the DVR clip / timelapse params with the user, then places the request.

Called twice by the orchestrator:
  1. Without "confirm" -> returns a question showing the params for the user to confirm.
  2. With "confirm"    -> submits the params it is given (the user may have changed them).
"""
from datetime import datetime

from dotenv import load_dotenv

# utils.auth creates the AuthManager on import, which needs the API credentials from .env
load_dotenv()

from logger import debug_logger
from utils.auth import auth_manager

logger = debug_logger()

REQUEST_TYPES = {"clip": "DVR clip", "timelapse": "Timelapse"}
MAX_MINUTES = {"clip": 3, "timelapse": 60}
VIDEO_FORMATS = ["road", "driver", "sideBySide", "pictureInPicture", "separate"]
RESOLUTIONS = ["320x180", "640x360", "1280x720", "1920x1080"]
# capture / display intervals of a timelapse, as stored on the fleet's existing timelapse requests
TIMELAPSE_CAPTURE_INTERVAL = 5
TIMELAPSE_DISPLAY_INTERVAL = 0.1


def confirmation_question(params):
    request_type = REQUEST_TYPES[params["type"]]
    return (
        f"Please confirm the {request_type} request:\n"
        f"- Request type: {request_type}\n"
        f"- Driver: {params['driverId']}\n"
        f"- Asset: {params['assetId']}\n"
        f"- Start (UTC): {params['clipStart']}\n"
        f"- End (UTC): {params['clipEnd']}\n"
        f"- Video format: {params['videoFormat']} (options: {', '.join(VIDEO_FORMATS)})\n"
        f"- Resolution: {params['videoResolution']} (options: {', '.join(RESOLUTIONS)})\n"
        'Reply "confirm" to place the request, or tell me what to change.'
    )


def create_dvr_request(fleet_id, client_id, dvr_params, user_response=None):
    """
    Asks the user to confirm the DVR clip / timelapse params, or places the request once confirmed.

    Args:
        fleet_id (str): Fleet the trip belongs to.
        client_id (str): Account sent as the x-lm-desired-account header.
        dvr_params (dict): tripId, driverId, assetId, type ('clip' | 'timelapse'),
            clipStart, clipEnd (ISO 8601 UTC), and optionally videoFormat, videoResolution.
        user_response (str | None): The user's reply; "confirm" places the request.

    Returns:
        dict:
          - {"status": "needs_confirmation", "message": question, "params": params}
          - {"status": "invalid", "message": reason, "params": params}
          - {"status": "submitted", "uploadRequestId": id, "params": params}

    Raises:
        httpx.HTTPError: The DVR request API call failed.
    """
    params = {"type": "clip", "videoFormat": "road", "videoResolution": "640x360", **dvr_params}
    request_type = REQUEST_TYPES[params["type"]]

    minutes = (
        datetime.fromisoformat(params["clipEnd"].replace("Z", "+00:00"))
        - datetime.fromisoformat(params["clipStart"].replace("Z", "+00:00"))
    ).total_seconds() / 60
    max_minutes = MAX_MINUTES[params["type"]]
    if not 0 < minutes <= max_minutes:
        return {
            "status": "invalid",
            "message": f"A {request_type} can be at most {max_minutes} minutes long "
                       f"(requested {round(minutes, 1)}). Please change the start or end time.",
            "params": params,
        }

    if (user_response or "").strip().lower() != "confirm":
        return {"status": "needs_confirmation", "message": confirmation_question(params), "params": params}

    request_params = {
        "fleetId": fleet_id,
        "driverId": params["driverId"],
        "assetId": params["assetId"],
        "tripId": params["tripId"],
        "startTimeUTC": params["clipStart"],
        "endTimeUTC": params["clipEnd"],
        "dvrVideoType": params["videoFormat"],
        "videoResolution": params["videoResolution"],
        # without it the request is placed as a DVR clip, also for a timelapse
        "timelapseEnabled": params["type"] == "timelapse",
    }
    if params["type"] == "timelapse":
        request_params["timelapseCaptureInterval"] = TIMELAPSE_CAPTURE_INTERVAL
        request_params["timelapseDisplayInterval"] = TIMELAPSE_DISPLAY_INTERVAL

    logger.info(f"create_dvr_request: submitting {request_type} {request_params}")
    result = auth_manager.make_api_request(
        client_id=client_id,
        endpoint=f"/v2/fleets/{fleet_id}/dvr/create-upload-request",
        params=request_params,
        method="POST",
    )
    logger.info(f"create_dvr_request: response {result}")
    return {"status": "submitted", "uploadRequestId": result.get("uploadRequestId"), "params": params}
