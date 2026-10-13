def get_trip_upload_requests(trip):
    upload_requests = []
    for request in trip.get('uploadRequests', []):
        upload_requests.append({
            'requestId': request.get('uploadRequestId'),
            'status': request.get('status'),
            'timestamp': request.get('createdAt')})

    if len(upload_requests)!=0:
        message = '\n'.join(
            f"{i}. Request ID: {r['requestId']} | Status: {r['status']} | Created: {r['timestamp']}"
            for i, r in enumerate(upload_requests, start=1))
        
        return upload_requests, message

    return None, "No previous upload requests for this trip"

