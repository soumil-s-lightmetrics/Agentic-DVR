"""Driver and asset lookups for the fleet, filtered by name."""
from utils.auth import auth_manager


def get_drivers(fleet_id: str, client_id: str, driver_name: str | None = None, page_size: int = 500):
    """
    Pages through /fleets/{fleetId}/drivers/list and returns
    [{'driverName': ..., 'driverId': ...}]. If driver_name is given, only
    drivers whose name contains it (case-insensitive) are returned.
    """
    driver_url = f"/v2/fleets/{fleet_id}/drivers/list"

    drivers_list = []
    skip = 0
    while True:
        body = auth_manager.make_api_request(
            client_id=client_id,
            endpoint=driver_url,
            params={"limit": page_size, "skip": skip},
        )
        rows = body.get("rows", [])
        drivers_list.extend(
            {"driverName": row.get("driverName"), "driverId": row.get("driverId")}
            for row in rows
        )
        skip += len(rows)
        if not rows or len(rows) < page_size or skip >= body.get("totalCount", 0):
            break

    # by name or ID: trip labels and the users show driver IDs too
    if driver_name:
        drivers_list = [
            d for d in drivers_list
            if driver_name.lower() in (d["driverName"] or "").lower()
            or driver_name.lower() == (d["driverId"] or "").lower()
        ]
    return drivers_list


def get_asset(fleet_id: str, client_id: str, asset_name: str | None = None):
    
    asset_url = f"/v2/fleets/{fleet_id}/list-fleet-assets"

    body = auth_manager.make_api_request(
        client_id=client_id,
        endpoint=asset_url,
    )
    asset_list = [
        {"assetName": row.get("assetName"), "assetId": row.get("assetId")}
        for row in body.get("rows", [])
    ]

    # by name or ID: trips only carry the asset ID ("mithun-k245-fin" is the ID of "Mithun Mitac Fin")
    if asset_name:
        asset_list = [
            a for a in asset_list
            if asset_name.lower() in (a["assetName"] or "").lower()
            or asset_name.lower() == (a["assetId"] or "").lower()
        ]
    return asset_list
