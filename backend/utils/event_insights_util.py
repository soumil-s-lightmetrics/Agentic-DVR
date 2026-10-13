import json
import os
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone, timedelta

import boto3
from logger import debug_logger

logger = debug_logger()

_EVENT_TYPE_LABELS = {
    "Traffic-Speed-Violated":              "Speed Violation",
    "MaxSpeedExceeded":                    "Max Speed Exceeded",
    "Harsh-Braking":                       "Harsh Braking",
    "Harsh-Acceleration":                  "Harsh Acceleration",
    "Cornering":                           "Cornering",
    "Tail-Gating-Detected":               "Tailgating",
    "Forward-Collision-Warning":           "Forward Collision Warning",
    "Traffic-STOP-Sign-Violated":          "Stop Sign Violation",
    "Distracted-Driving":                  "Distracted Driving",
    "Cellphone-Distracted-Driving":        "Cellphone Distraction",
    "Smoking-Distracted-Driving":          "Smoking While Driving",
    "Drinking-Distracted-Driving":         "Drinking While Driving",
    "Texting-Distracted-Driving":          "Texting While Driving",
    "Lizard-Eye-Distracted-Driving":       "Inattentive Driving",
    "Drowsy-Driving-Detected":             "Drowsy Driving",
    "Driver-Fatigue-Detected":             "Driver Fatigue",
    "PotentialCrash":                      "Potential Crash",
    "Unbuckled-Seat-Belt":                 "Unbuckled Seat Belt",
    "Roll-Over-Detected":                  "Roll Over",
    "Lane-Drift-Found":                    "Lane Drift",
    "Traffic-Light-Violated":              "Traffic Light Violation",
    "Custom-Triggered":                    "Custom Event",
}


def _humanize_event_type(event_type: str) -> str:
    if not event_type:
        return "Unknown"
    label = _EVENT_TYPE_LABELS.get(event_type)
    if label:
        return label
    # Fallback: split camelCase, replace hyphens/underscores, title-case
    s = re.sub(r'([a-z])([A-Z])', r'\1 \2', event_type)
    s = s.replace("-", " ").replace("_", " ")
    return re.sub(r'\s+', ' ', s).strip().title()


# Deterministic severity rules keyed by originalEventType.
# Each entry: metric key to read from the violation, high threshold, low threshold,
# and whether the scale is inverted (lower value = worse, e.g. time-to-collision).
_SEVERITY_RULES = {
    "Traffic-Speed-Violated":           {"key": "speedingValue",                    "high": 32,   "low": 16,   "inverted": False},
    "MaxSpeedExceeded":                 {"key": "exceededSpeedingValue",             "high": 32,   "low": 16,   "inverted": False},
    "Harsh-Braking":                    {"key": "harshBrakingAccelerationValue",     "high": 3576, "low": 2682, "inverted": False},
    "Harsh-Acceleration":               {"key": "harshAccelerationValue",            "high": 3576, "low": 2682, "inverted": False},
    "Cornering":                        {"key": "corneringAccelerationValue",        "high": 3576, "low": 2682, "inverted": False},
    "Tail-Gating-Detected":             {"key": "tailgatingMetricValue",             "high": 1,    "low": 2,    "inverted": True},
    "Forward-Collision-Warning":        {"key": "forwardCollisionMetricValue",       "high": 1,    "low": 2,    "inverted": True},
    "Traffic-STOP-Sign-Violated":       {"key": "lowestSpeedKmph",                  "high": 24,   "low": 16,   "inverted": False},
    "Distracted-Driving":               {"key": "severity",                         "high": 80,   "low": 60,   "inverted": False},
    "Cellphone-Distracted-Driving":     {"key": "severity",                         "high": 80,   "low": 60,   "inverted": False},
    "Smoking-Distracted-Driving":       {"key": "severity",                         "high": 80,   "low": 60,   "inverted": False},
    "Drinking-Distracted-Driving":      {"key": "severity",                         "high": 80,   "low": 60,   "inverted": False},
    "Texting-Distracted-Driving":       {"key": "severity",                         "high": 80,   "low": 60,   "inverted": False},
    "Lizard-Eye-Distracted-Driving":    {"key": "severity",                         "high": 80,   "low": 60,   "inverted": False},
    "Drowsy-Driving-Detected":          {"key": "severity",                         "high": 80,   "low": 60,   "inverted": False},
    "Driver-Fatigue-Detected":          {"key": "severity",                         "high": 80,   "low": 60,   "inverted": False},
}

# Events where severity is not applicable — excluded from H/M/L bucketing.
_NA_EVENTS = {
    "PotentialCrash", "Unbuckled-Seat-Belt", "Roll-Over-Detected",
    "Lane-Drift-Found", "Traffic-Light-Violated", "Custom-Triggered",
}


def _classify_violation(violation):
    """Return 'High', 'Medium', 'Low', or None (not applicable) for a violation."""
    event_type = violation.get("originalEventType") or violation.get("eventType") or ""

    if event_type in _NA_EVENTS:
        return None

    rule = _SEVERITY_RULES.get(event_type)
    if not rule:
        return None

    # Metric value may sit at the top level or nested under metadata/data/details.
    metric_key = rule["key"]
    val = violation.get(metric_key)
    if val is None:
        for container in ("metadata", "data", "details", "attributes"):
            nested = violation.get(container)
            if isinstance(nested, dict):
                val = nested.get(metric_key)
                if val is not None:
                    break

    if val is None:
        return "Medium"  # value absent — default to Medium

    val = float(val)
    high, low, inverted = rule["high"], rule["low"], rule["inverted"]

    if inverted:
        if val < high:  return "High"
        if val > low:   return "Low"
    else:
        if val > high:  return "High"
        if val < low:   return "Low"
    return "Medium"


# Maps heeded event type names (from ai-coaching-heeded-stats) to the
# corresponding violation event type names used in the violations feed.
_HEEDED_TO_VIOLATION_TYPE = {
    "Traffic-Speed-Sign-Warning-Heeded":       "Traffic-Speed-Violated",
    "Tail-Gating-Warning-Heeded":              "Tail-Gating-Detected",
    "Forward-Collision-Audio-Warning-Heeded":  "Forward-Collision-Warning",
}


def _parse_heeded_stats(heeded_stats, type_counts):
    """Compute in-cabin alert adherence

    Returns a dict with percent, reactedAlerts, totalAlerts — or None when
    there is no usable data.
    """
    stats = (heeded_stats or {}).get("stats") if isinstance(heeded_stats, dict) else None
    if not stats:
        return None

    type_counts    = type_counts or {}
    total_reacted  = 0
    total_detected = 0

    for event in stats:
        heeded_type     = event.get("eventType", "")
        heeded_events   = event.get("totalEvents", 0) or 0
        violation_type  = _HEEDED_TO_VIOLATION_TYPE.get(heeded_type)
        violation_count = type_counts.get(violation_type, 0) if violation_type else 0
        total_reacted  += heeded_events
        total_detected += violation_count + heeded_events

    if total_detected == 0:
        return None

    return {
        "percent": round((total_reacted / total_detected) * 100),
        "reactedAlerts": total_reacted,
        "totalAlerts":   total_detected,
    }


def _session_date(session):
    return session.get("sessionDate") or session.get("createdAt") or session.get("date") or ""


def _epm(count, distance):
    if not distance or distance <= 0:
        return None
    return round((count / distance) * 100, 2)


def _epm_by_type(violations, distance):
    """Return {eventType: eventsPer100Units} for each event type in violations."""
    if not distance or distance <= 0:
        return {}
    counts = {}
    for v in violations:
        et = v.get("originalEventType") or v.get("eventType") or "Unknown"
        counts[et] = counts.get(et, 0) + 1
    return {et: round((cnt / distance) * 100, 2) for et, cnt in counts.items()}



def _aggregate_insight(driver, coaching_sessions=None, streaks=None, aggregate=None, units="km", fleet_epm=None, heeded_stats=None, prev_epm=None, before_epm_by_type=None, after_epm_by_type=None):
    """Compute the entire deterministic insight payload from raw data."""
    violations = driver.get("violations", []) or []
    total = len(violations)

    aggregate = aggregate or {}
    distance = (
        aggregate.get("tripDistance")
        or aggregate.get("totalDistance")
        or aggregate.get("distance")
    )
    trip_count = (
        aggregate.get("tripCount")
        or aggregate.get("totalTrips")
        or aggregate.get("trips")
    )
    # eventsPer100Units, tripDistance, and streak distances are all returned in the requested
    # unit by their respective APIs (metricUnit param), so no client-side conversion is needed.
    driver_epm = aggregate.get("eventsPer100Units")

    now = datetime.now(timezone.utc)
    current_start = now - timedelta(days=30)
    previous_start = now - timedelta(days=60)
    current_window = f"{current_start.strftime('%b %d, %Y')} – {now.strftime('%b %d, %Y')}"
    previous_window = f"{previous_start.strftime('%b %d, %Y')} – {current_start.strftime('%b %d, %Y')}"

    severity_counts = {"High": 0, "Medium": 0, "Low": 0}
    type_counts = {}
    type_severity = {}
    for v in violations:
        sev = _classify_violation(v)
        event_type = v.get("originalEventType") or v.get("eventType") or "Unknown"
        type_counts[event_type] = type_counts.get(event_type, 0) + 1
        bucket = type_severity.setdefault(event_type, {"High": 0, "Medium": 0, "Low": 0})
        if sev in severity_counts:
            severity_counts[sev] += 1
            bucket[sev] += 1

    top_types = sorted(
        ((et, cnt) for et, cnt in type_counts.items() if et in _SEVERITY_RULES),
        key=lambda kv: (-kv[1], -type_severity.get(kv[0], {}).get("High", 0)),
    )[:3]

    what_needs_improvement = [
        {
            "rank": rank,
            "violationType": event_type,
            "count": count,
            "high":   type_severity[event_type]["High"],
            "medium": type_severity[event_type]["Medium"],
            "low":    type_severity[event_type]["Low"],
        }
        for rank, (event_type, count) in enumerate(top_types, 1)
    ]

    streak_list = []
    if streaks:
        entries = [
            (event_type, data)
            for event_type, data in streaks.items()
            if isinstance(data, dict)
        ]
        entries.sort(key=lambda x: x[1].get("runningStreak", 0), reverse=True)
        for rank, (event_type, data) in enumerate(entries[:3], 1):
            streak_list.append({
                "rank": rank,
                "eventType": event_type,
                "cleanDistance": round(data.get("runningStreak", 0), 2),
            })

    coaching_eff = {
        "lastSessionDate": None,
        "topics": [],
        "comparisons": [],
        "message": "There were no coaching sessions.",
    }
    if coaching_sessions:
        latest = max(coaching_sessions, key=_session_date, default=None)
        if latest:
            date_str = _session_date(latest)[:10] or None
            seen = set()
            topics = []
            for s in coaching_sessions:
                raw = s.get("topics") or s.get("events") or []
                if not isinstance(raw, list):
                    raw = [raw] if raw else []
                for t in raw:
                    if t and t['eventType'] not in seen:
                        seen.add(t['eventType'])
                        topics.append(t)
            comparisons = []
            for t in topics:
                et = t.get("eventType") if isinstance(t, dict) else str(t)
                before_val = (before_epm_by_type or {}).get(et)
                after_val  = (after_epm_by_type or {}).get(et)
                if before_val is not None and after_val is not None and before_val > 0:
                    pct_change = round(((after_val - before_val) / before_val) * 100)
                    direction  = "improving" if after_val < before_val else "worsened" if after_val > before_val else "flat"
                else:
                    pct_change = None
                    direction  = None
                comparisons.append({
                    "eventType":     et,
                    "before":        before_val,
                    "after":         after_val,
                    "percentChange": pct_change,
                    "direction":     direction,
                })
            coaching_eff = {
                "lastSessionDate": date_str,
                "topics": [t.get("eventType") if isinstance(t, dict) else str(t) for t in topics],
                "comparisons": comparisons,
                "message": None,
            }

    return {
        "performanceSnapshot": {
            "totalViolations": total,
            "tripCount": trip_count,
            "tripDistance": distance,
            "units": units,
            "severityBreakdown": [
                {"severity": "High",   "count": severity_counts["High"],   "eventsPer100Units": _epm(severity_counts["High"],   distance)},
                {"severity": "Medium", "count": severity_counts["Medium"]},
                {"severity": "Low",    "count": severity_counts["Low"]},
            ],
            "eventsPer100Units": {"driver": driver_epm, "fleetAverage": fleet_epm},
            "trend": {
                "currentPeriod":  {"window": current_window,  "eventsPer100Units": driver_epm, "violationCount": total},
                "previousPeriod": {"window": previous_window, "eventsPer100Units": prev_epm,   "violationCount": None},
                "delta": None,
                "percentChange": None,
                "direction": (
                    "improving" if driver_epm is not None and prev_epm is not None and driver_epm < prev_epm else
                    "worsened" if driver_epm is not None and prev_epm is not None and driver_epm > prev_epm else
                    "flat"      if driver_epm is not None and prev_epm is not None else None
                ),
            },
        },
        "whatNeedsImprovement": what_needs_improvement,
        "whatWentWell": {"alertAdherence": _parse_heeded_stats(heeded_stats, type_counts)},
        "streaks": streak_list,
        "coachingEffectiveness": coaching_eff,
    }


def _build_summary_prompt(driver, aggregated):
    snap = aggregated["performanceSnapshot"]
    units = snap.get("units", "km")
    severity = {s["severity"]: s["count"] for s in snap["severityBreakdown"]}
    needs = aggregated["whatNeedsImprovement"]
    streaks = aggregated["streaks"]
    coaching = aggregated["coachingEffectiveness"]

    top_lines = "\n".join(
        f"  {n['rank']}. {_humanize_event_type(n['violationType'])} — {n['count']} events"
        for n in needs
    ) or "  (none)"

    streak_lines = "\n".join(
        f"  - {_humanize_event_type(s['eventType'])}: {s['cleanDistance']} clean {units}"
        for s in streaks
    ) or "  (no streaks)"

    if coaching.get("message"):
        coaching_text = coaching["message"]
    else:
        topics = ", ".join(_humanize_event_type(t) for t in (coaching.get("topics") or [])) or "none"
        coaching_text = f"Last session {coaching.get('lastSessionDate')}, topics: {topics}"

    e100 = snap.get("eventsPer100Units", {})
    driver_epm = e100.get("driver")
    fleet_epm  = e100.get("fleetAverage")
    epm_text   = f"{driver_epm} per 100 {units}" if driver_epm is not None else "n/a"
    fleet_text = f"{fleet_epm} per 100 {units}" if fleet_epm is not None else "n/a"

    trend      = snap.get("trend", {})
    prev_epm   = (trend.get("previousPeriod") or {}).get("eventsPer100Units")
    direction  = trend.get("direction")
    trend_text = (
        f"{direction} — current {driver_epm} vs previous {prev_epm} per 100 {units}"
        if driver_epm is not None and prev_epm is not None
        else "insufficient data"
    )

    # Build bullet list only for items where data is actually available
    went = aggregated.get("whatWentWell", {})
    bullets = []
    if driver_epm is not None and fleet_epm is not None:
        bullets.append(f'Fleet benchmark — driver rate vs fleet average as a percentage difference (e.g. "Driver is <span class="txt-good">**32% below** the fleet average</span> — **4.2**/100 {units} vs fleet benchmark of **6.1**")')
    bullets.append("Volume — distance driven, trip count, total violations")
    if needs:
        bullets.append("Dominant concern — top violation type and why it matters")
    if driver_epm is not None and prev_epm is not None:
        bullets.append('Trend — use the trend data below; if improving wrap "improving ↓" in txt-good, if worsened wrap "worsened ↑" in txt-bad')
    if went.get("alertAdherence"):
        bullets.append("What went well — alert adherence percentage")
    if not coaching.get("message") and coaching.get("lastSessionDate"):
        bullets.append("Coaching impact — most recent session topics and improvement noted")

    count = len(bullets)
    numbered = "\n".join(f"{i + 1}. {b}" for i, b in enumerate(bullets))

    return f"""You are a fleet safety AI analyst. Generate exactly {count} bullet points summarising this driver's past 30 days. Each bullet is one concise sentence. Use the data below strictly — do not invent numbers.

Formatting rules:
- Wrap key numbers and event names in **double asterisks** for bold.
- For positive outcomes (below average, improving, good adherence), wrap the positive phrase in <span class="txt-good">…</span>.
- For negative outcomes (above average, worsened), wrap the negative phrase in <span class="txt-bad">…</span>.
- No other HTML or markdown.

Return a JSON array of {count} strings, one per bullet, in this exact order:
{numbered}

Data:
Driver: {driver.get('driverId')} (fleet: {driver.get('fleetId')})
Events per 100 {units} — driver: {epm_text}, fleet average: {fleet_text}
Trend: {trend_text}
Total violations: {snap['totalViolations']}, trips: {snap.get('tripCount') or 'n/a'}, distance: {snap.get('tripDistance') or 'n/a'} {units}
Severity — High: {severity.get('High', 0)}, Medium: {severity.get('Medium', 0)}, Low: {severity.get('Low', 0)}
Top violation types:
{top_lines}
Positive streaks:
{streak_lines}
Coaching: {coaching_text}

Output ONLY the raw JSON array. No markdown fences, no keys, no explanation."""


def _generate_ai_summary(driver, aggregated):
    profile_arn = os.environ.get("BEDROCK_APP_INFERENCE_PROFILE_ARN")
    if not profile_arn:
        return None

    region = os.environ.get("BEDROCK_REGION", "us-east-1")
    client = boto3.client("bedrock-runtime", region_name=region)
    prompt = _build_summary_prompt(driver, aggregated)

    try:
        response = client.converse(
            modelId=profile_arn,
            messages=[{"role": "user", "content": [{"text": prompt}]}],
        )
        text = response["output"]["message"]["content"][0]["text"].strip()
        # Strip markdown fences if present
        if text.startswith("```"):
            text = text.split("\n", 1)[1] if "\n" in text else ""
        if text.endswith("```"):
            text = text.rsplit("```", 1)[0]
        text = text.strip()
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return [str(item).strip() for item in parsed if item]
        logger.warning("Bedrock AI summary: unexpected response format (not a list)")
        return None
    except json.JSONDecodeError as e:
        logger.warning(f"Bedrock AI summary: failed to parse JSON response — {e}")
        return None
    except Exception as e:
        logger.error(f"Bedrock AI summary: unexpected error — {e}")
        return None


def _build_markdown(aggregated: dict, ai_summary: list | None, units: str) -> str:
    snap     = aggregated.get("performanceSnapshot", {})
    needs    = aggregated.get("whatNeedsImprovement", [])
    went     = aggregated.get("whatWentWell", {})
    streaks  = aggregated.get("streaks", [])
    coaching = aggregated.get("coachingEffectiveness", {})

    lines = []

    # AI Summary
    if ai_summary:
        lines.append("## AI Summary\n")
        for b in ai_summary:
            lines.append(f"- {b.replace('**', '**')}")
        lines.append("")

    # Performance Snapshot
    lines.append("## Performance Snapshot\n")
    total      = snap.get("totalViolations")
    trips      = snap.get("tripCount")
    sev        = {s["severity"]: s for s in snap.get("severityBreakdown", [])}
    h_count    = sev.get("High",   {}).get("count", 0)
    m_count    = sev.get("Medium", {}).get("count", 0)
    l_count    = sev.get("Low",    {}).get("count", 0)
    e100       = snap.get("eventsPer100Units", {})
    driver_epm = e100.get("driver")
    fleet_epm  = e100.get("fleetAverage")
    trend      = snap.get("trend", {})
    current    = trend.get("currentPeriod", {})
    previous   = trend.get("previousPeriod", {})
    direction  = trend.get("direction")
    prev_epm   = previous.get("eventsPer100Units")

    vol_line = f"- **{total} events"
    if trips is not None:
        vol_line += f" across {trips} trips"
    vol_line += f"** — {h_count} High · {m_count} Medium · {l_count} Low"
    lines.append(vol_line)

    epm_parts = []
    if driver_epm is not None:
        epm_parts.append(f"**Events per 100 {units}:** {driver_epm}")
    if fleet_epm is not None:
        epm_parts.append(f"**Fleet avg:** {fleet_epm}")
    if epm_parts:
        lines.append(f"- {' &bull; '.join(epm_parts)}")

    if direction and driver_epm is not None and prev_epm is not None:
        if direction == "improving":
            trend_span = f'<span class="txt-good">↓ **{driver_epm}** vs **{prev_epm}** events/100 {units}</span>'
        elif direction in ("worsening", "worsened"):
            trend_span = f'<span class="txt-bad">↑ **{driver_epm}** vs **{prev_epm}** events/100 {units}</span>'
        else:
            trend_span = f"**{driver_epm}** vs **{prev_epm}** events/100 {units}"
        lines.append(f"- {trend_span}")

    if current.get("window"):
        lines.append(f"  *{current['window']}*")
    lines.append("")

    # What Needs Improvement
    lines.append("## What Needs Improvement\n")
    if needs:
        for n in needs:
            lines.append(f"{n['rank']}. **{_humanize_event_type(n['violationType'])}:** {n['count']} events")
            lines.append(f"   - High: **{n['high']}** · Medium: **{n['medium']}** · Low: **{n['low']}**")
    else:
        lines.append("*No specific concerns identified.*")
    lines.append("")

    # What Went Well
    lines.append("## What Went Well\n")
    adherence = went.get("alertAdherence")
    if adherence:
        pct = adherence["percent"]
        reacted = adherence["reactedAlerts"]
        total_a = adherence["totalAlerts"]
        lines.append(f'- <span class="txt-good">**{pct}% alert adherence**</span> ({reacted}/{total_a} alerts responded to)')
    else:
        lines.append("*No in-cabin alert data available.*")
    lines.append("")

    # Clean Driving Streaks
    lines.append("## Clean Driving Streaks\n")
    if streaks:
        for i, s in enumerate(streaks):
            et   = _humanize_event_type(s.get("eventType") or "")
            dist = s.get("cleanDistance", "—")
            if i == 0:
                lines.append(f"- **Longest streak: {et} ({dist} {units})** — sustained improvement observed")
            else:
                lines.append(f"- **{et}: {dist} {units}** without recurrence")
    else:
        lines.append("*No streak data available.*")
    lines.append("")

    # Coaching Effectiveness
    lines.append("## Coaching Effectiveness\n")
    if coaching.get("message"):
        lines.append(f"*{coaching['message']}*")
    elif coaching.get("lastSessionDate"):
        lines.append(f"**Last session:** {coaching['lastSessionDate']}\n")
        cmps = coaching.get("comparisons", [])
        has_data = any(c.get("before") is not None and c.get("after") is not None for c in cmps)
        if has_data:
            lines.append("After coaching:\n")
        for c in cmps:
            et    = _humanize_event_type(c.get("eventType") or "")
            before = c.get("before")
            after  = c.get("after")
            d      = c.get("direction")
            pct    = c.get("percentChange")
            if before is not None and after is not None:
                if d == "improving":
                    after_str  = f'<span class="txt-good">**{after}**</span>'
                    change_str = f'<span class="txt-good">({abs(pct)}% improvement ↓)</span>' if pct is not None else ""
                elif d in ("worsening", "worsened"):
                    after_str  = f'<span class="txt-bad">**{after}**</span>'
                    change_str = f'<span class="txt-bad">({abs(pct)}% increase ↑)</span>' if pct is not None else ""
                else:
                    after_str  = f"**{after}**"
                    change_str = ""
                lines.append(f"- **{et}:** {before} → {after_str} /100 {units} {change_str}".strip())
    else:
        lines.append("*No coaching sessions found in the period.*")

    return "\n".join(lines)


def generate_driver_insight(
    driver: dict,
    coaching_sessions: list | None = None,
    streaks: dict | None = None,
    aggregate: dict | None = None,
    units: str = "km",
    fleet_epm: float | None = None,
    heeded_stats: dict | None = None,
    prev_epm: float | None = None,
    before_epm_by_type: dict | None = None,
    after_epm_by_type: dict | None = None,
) -> dict:
    aggregated = _aggregate_insight(driver, coaching_sessions, streaks, aggregate, units=units, fleet_epm=fleet_epm, heeded_stats=heeded_stats, prev_epm=prev_epm, before_epm_by_type=before_epm_by_type, after_epm_by_type=after_epm_by_type)
    summary  = _generate_ai_summary(driver, aggregated)
    markdown = _build_markdown(aggregated, None, units)
    return {"aiSummary": summary, "markdown": markdown, **aggregated}


# ---------------------------------------------------------------------------
# Data fetching
# ---------------------------------------------------------------------------

def _fetch_violations(fleet_id, driver_id, account, after, before, auth_manager, logger):
    page_size = 500
    skip = 0
    all_violations = []
    while True:
        try:
            body = auth_manager.make_api_request(
                client_id=account,
                endpoint=f"/v2/fleets/{fleet_id}/violations",
                params={
                    "after": after,
                    "before": before,
                    "sortBy": "severity",
                    "sort": "desc",
                    "status": "UPLOADED",
                    "driverId": driver_id,
                    "skip": skip,
                    "limit": page_size,
                },
            )
        except Exception as e:
            logger.warning(f"Violations fetch failed at skip={skip}: {e}")
            break

        rows = body.get("rows") or body.get("data") or body.get("violations") or []
        if not rows:
            break
        all_violations.extend(rows)
        if len(rows) < page_size:
            break
        skip += page_size

    return all_violations


def _fetch_driver_aggregate(fleet_id, driver_id, account, after, before, units, auth_manager, logger):
    try:
        body = auth_manager.make_api_request(
            client_id=account,
            endpoint=f"/v1/fleets/{fleet_id}/drivers/{driver_id}/aggregate",
            params={"after": after, "before": before, "metricUnit": units},
        )
        if isinstance(body, dict):
            rows = body.get("rows") or []
            if rows and isinstance(rows[0], dict):
                value = rows[0].get("value")
                if isinstance(value, dict):
                    return value
    except Exception as e:
        logger.warning(f"Could not fetch driver aggregate ({after} – {before}): {e}")
    return {}


def fetch_driver_insight(fleet_id, driver_id, tsp_id, units, auth_manager, logger):

    if not auth_manager:
        raise ValueError("AuthManager not configured")

    now_dt      = datetime.now(timezone.utc)
    after_date  = (now_dt - timedelta(days=30)).strftime("%Y-%m-%d")
    before_date = now_dt.strftime("%Y-%m-%d")
    before_ts   = now_dt.strftime("%Y-%m-%dT%H:%M:%S.999Z")
    prev_start  = (now_dt - timedelta(days=60)).strftime("%Y-%m-%d")
    prev_end    = (now_dt - timedelta(days=30)).strftime("%Y-%m-%d")

    def _fetch_coaching():
        try:
            body = auth_manager.make_api_request(
                client_id=tsp_id,
                endpoint=f"/v1/fleets/{fleet_id}/coaching-sessions",
                params={
                    "after": after_date,
                    "before": before_ts,
                    "incidentThreshold": 10,
                    "limit": 5,
                    "skip": 0,
                    "fleetId": fleet_id,
                    "driverIds[]": driver_id,
                },
            )
            return body.get("rows") or body.get("data") or body.get("sessions") or []
        except Exception as e:
            logger.warning(f"Could not fetch coaching sessions: {e}")
            return []

    def _fetch_streaks():
        try:
            body = auth_manager.make_api_request(
                client_id=tsp_id,
                endpoint=f"/v1/fleets/{fleet_id}/drivers/{driver_id}/streaks",
                params={"after": after_date, "before": before_date, "metricUnit": units},
            )
            if isinstance(body, dict):
                data = body.get("data") or body.get("streaks") or body
                return data if isinstance(data, dict) else {}
        except Exception as e:
            logger.warning(f"Could not fetch streaks: {e}")
        return {}

    def _fetch_heeded():
        try:
            body = auth_manager.make_api_request(
                client_id=tsp_id,
                endpoint=f"/v1/fleets/{fleet_id}/drivers/{driver_id}/ai-coaching-heeded-stats",
                params={"after": after_date, "before": before_date},
            )
            if isinstance(body, dict):
                return body
        except Exception as e:
            logger.warning(f"Could not fetch heeded stats: {e}")
        return None

    def _fetch_fleet_epm():
        try:
            body = auth_manager.make_api_request(
                client_id=tsp_id,
                endpoint=f"/v1/fleets/{fleet_id}/aggregate",
                params={"groupBy": "month", "after": after_date, "before": before_date, "metricUnit": units},
            )
            rows   = (body or {}).get("rows") or []
            values = [r["value"] for r in rows if isinstance(r.get("value"), dict)]
            total_distance = sum(v.get("tripDistance") or 0 for v in values)
            if total_distance > 0:
                return round(
                    sum((v.get("eventsPer100Units") or 0) * (v.get("tripDistance") or 0) for v in values)
                    / total_distance,
                    2,
                )
            elif values:
                epms = [v["eventsPer100Units"] for v in values if v.get("eventsPer100Units") is not None]
                return round(sum(epms) / len(epms), 2) if epms else None
        except Exception as e:
            logger.warning(f"Could not fetch fleet aggregate: {e}")
        return None

    # Wave 1: all independent fetches in parallel
    with ThreadPoolExecutor(max_workers=7) as executor:
        fut_violations = executor.submit(_fetch_violations, fleet_id, driver_id, tsp_id, after_date, before_date, auth_manager, logger)
        fut_coaching   = executor.submit(_fetch_coaching)
        fut_streaks    = executor.submit(_fetch_streaks)
        fut_aggregate  = executor.submit(_fetch_driver_aggregate, fleet_id, driver_id, tsp_id, after_date, before_date, units, auth_manager, logger)
        fut_heeded     = executor.submit(_fetch_heeded)
        fut_fleet_epm  = executor.submit(_fetch_fleet_epm)
        fut_prev_agg   = executor.submit(_fetch_driver_aggregate, fleet_id, driver_id, tsp_id, prev_start, prev_end, units, auth_manager, logger)

    violations        = fut_violations.result()
    coaching_sessions = fut_coaching.result()
    streaks_data      = fut_streaks.result()
    aggregate_data    = fut_aggregate.result()
    heeded_stats      = fut_heeded.result()
    fleet_epm         = fut_fleet_epm.result()
    prev_epm          = fut_prev_agg.result().get("eventsPer100Units")

    driver = {
        "driverId":    driver_id,
        "fleetId":     fleet_id,
        "violations":  violations,
        "totalEvents": len(violations),
    }

    # Wave 2: coaching effectiveness (depends on coaching_sessions from wave 1)
    before_epm_by_type = {}
    after_epm_by_type  = {}
    if coaching_sessions:
        latest_session = max(coaching_sessions, key=_session_date, default=None)
        if latest_session:
            session_date_str = _session_date(latest_session)[:10]
            try:
                session_dt   = datetime.strptime(session_date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
                before_start = (session_dt - timedelta(days=30)).strftime("%Y-%m-%d")
                session_date = session_dt.strftime("%Y-%m-%d")

                with ThreadPoolExecutor(max_workers=4) as executor:
                    fut_bv = executor.submit(_fetch_violations,       fleet_id, driver_id, tsp_id, before_start, session_date, auth_manager, logger)
                    fut_ba = executor.submit(_fetch_driver_aggregate, fleet_id, driver_id, tsp_id, before_start, session_date, units, auth_manager, logger)
                    fut_av = executor.submit(_fetch_violations,       fleet_id, driver_id, tsp_id, session_date, before_date,  auth_manager, logger)
                    fut_aa = executor.submit(_fetch_driver_aggregate, fleet_id, driver_id, tsp_id, session_date, before_date,  units, auth_manager, logger)

                before_agg         = fut_ba.result()
                after_agg          = fut_aa.result()
                before_dist        = before_agg.get("tripDistance") or before_agg.get("totalDistance") or before_agg.get("distance")
                after_dist         = after_agg.get("tripDistance") or after_agg.get("totalDistance") or after_agg.get("distance")
                before_epm_by_type = _epm_by_type(fut_bv.result(), before_dist)
                after_epm_by_type  = _epm_by_type(fut_av.result(), after_dist)
            except Exception as e:
                logger.warning(f"Could not compute coaching effectiveness EPM: {e}")

    return generate_driver_insight(
        driver,
        coaching_sessions,
        streaks_data,
        aggregate_data,
        units=units,
        fleet_epm=fleet_epm,
        heeded_stats=heeded_stats,
        prev_epm=prev_epm,
        before_epm_by_type=before_epm_by_type,
        after_epm_by_type=after_epm_by_type,
    )
