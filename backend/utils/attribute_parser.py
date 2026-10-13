"""
Attribute Parser Utility for Pinecone Vector Store Sync

This module provides utilities to convert LLM-extracted attributes (in dict format)
into structured Pinecone metadata attributes.

The conversion handles:
- Version parsing (e.g., "10.9.0" → major/minor/patch integers)
- Array attributes (e.g., ["model1", "model2"] → direct assignment to metadata)
- Default values for missing attributes
"""


def parse_version(value: str, parts: int) -> list[int]:
    """
    Parse version string into integer components.

    Args:
        value: Version string (e.g., "v10.9.0", "10.9", "1.20")
        parts: Number of version parts to extract (2 or 3)

    Returns:
        List of integers representing version components.
        Missing parts are filled with 0.

    Examples:
        >>> parse_version("v10.9.0", 3)
        [10, 9, 0]
        >>> parse_version("1.20", 2)
        [1, 20]
        >>> parse_version("9", 3)
        [9, 0, 0]
    """
    nums = value.lstrip("v").split(".")
    out = []
    for i in range(parts):
        out.append(int(nums[i]) if i < len(nums) and nums[i].isdigit() else 0)
    return out


def parse_pipe_list(value: str) -> list[str]:
    """
    Parse pipe-separated string into list of trimmed values.

    Args:
        value: Pipe-separated string (e.g., "model1|model2|model3")

    Returns:
        List of trimmed non-empty strings.
        Returns empty list if input is empty or contains only whitespace.

    Examples:
        >>> parse_pipe_list("jimi-jc261|mitac-gemini")
        ['jimi-jc261', 'mitac-gemini']
        >>> parse_pipe_list("")
        []
        >>> parse_pipe_list("  |  ")
        []
    """
    items = [v.strip() for v in value.split("|") if v.strip()]
    return items or []


def get_attributes_from_tags(tags: dict) -> dict:
    """
    Convert LLM-extracted attributes dict to Pinecone metadata attributes.

    Takes a dictionary of attribute key-value pairs and converts them
    into a structured dictionary suitable for Pinecone vector metadata.

    Args:
        tags: Dictionary of attributes extracted from LLM.
              Example:
              {
                  "fleet_portal_version": "10.9.0",
                  "device_models": ["jimi-jc261", "mitac-gemini"],
                  "plans_nin": ["SHIELD"],
                  "event_types": ["Traffic-Light-Violated"],
                  "required_features": ["ADAS", "DMS"]
              }

    Returns:
        Dictionary with structured attributes:
        {
            "fleet_portal_version_major": int,
            "fleet_portal_version_minor": int,
            "fleet_portal_version_patch": int,
            "master_portal_version_major": int,
            "master_portal_version_minor": int,
            "master_portal_version_patch": int,
            "device_apk_version_major": int,
            "device_apk_version_minor": int,
            "device_models_in": list[str],
            "plans_in": list[str],
            "plans_nin": list[str],
            "event_type_in": list[str],
            "required_features": list[str],
            "category_in": list[str]
        }

    Example:
        >>> tags = {
        ...     "fleet_portal_version": "10.9.0",
        ...     "device_apk_version": "1.20",
        ...     "device_models": ["jimi-jc261"],
        ...     "plans_nin": ["SHIELD"],
        ...     "event_types": ["Traffic-Light-Violated"],
        ...     "required_features": ["ADAS", "DMS"]
        ... }
        >>> attrs = get_attributes_from_tags(tags)
        >>> attrs['fleet_portal_version_major']
        10
        >>> attrs['device_models_in']
        ['jimi-jc261']
    """
    attributes = {
        "fleet_portal_version_major": 0,
        "fleet_portal_version_minor": 0,
        "fleet_portal_version_patch": 0,
        "master_portal_version_major": 0,
        "master_portal_version_minor": 0,
        "master_portal_version_patch": 0,
        "device_apk_version_major": 0,
        "device_apk_version_minor": 0,
    }

    # Process fleet_portal_version
    if "fleet_portal_version" in tags and tags["fleet_portal_version"]:
        major, minor, patch = parse_version(tags["fleet_portal_version"], 3)
        attributes.update({
            "fleet_portal_version_major": major,
            "fleet_portal_version_minor": minor,
            "fleet_portal_version_patch": patch,
        })

    # Process master_portal_version
    if "master_portal_version" in tags and tags["master_portal_version"]:
        major, minor, patch = parse_version(tags["master_portal_version"], 3)
        attributes.update({
            "master_portal_version_major": major,
            "master_portal_version_minor": minor,
            "master_portal_version_patch": patch,
        })

    # Process device_apk_version
    if "device_apk_version" in tags and tags["device_apk_version"]:
        major, minor = parse_version(tags["device_apk_version"], 2)
        attributes.update({
            "device_apk_version_major": major,
            "device_apk_version_minor": minor,
        })

    # Process device_models (now received as array from LLM)
    if "device_models" in tags and tags["device_models"]:
        attributes["device_models_in"] = tags["device_models"]

    # Process plans_in (if present, received as array)
    if "plans_in" in tags and tags["plans_in"]:
        attributes["plans_in"] = tags["plans_in"]

    # Process plans_nin (received as array from LLM)
    if "plans_nin" in tags and tags["plans_nin"]:
        attributes["plans_nin"] = tags["plans_nin"]

    # Process event_types (received as array from LLM)
    if "event_types" in tags and tags["event_types"]:
        attributes["event_type_in"] = tags["event_types"]

    # Process required_features (received as array from LLM)
    if "required_features" in tags and tags["required_features"]:
        attributes["required_features"] = tags["required_features"]

    # Process category_in (if present, received as array)
    if "category_in" in tags and tags["category_in"]:
        attributes["category_in"] = tags["category_in"]

    return attributes


def validate_attributes(attributes: dict) -> tuple[bool, str]:
    """
    Validate extracted attributes before Pinecone upsert.

    Checks:
    - Required fields present
    - Correct data types
    - Value ranges

    Args:
        attributes: Dictionary of extracted attributes

    Returns:
        Tuple of (is_valid, error_message)
        If valid, error_message is empty string.

    Example:
        >>> attrs = {"fleet_portal_version_major": 10}
        >>> valid, error = validate_attributes(attrs)
        >>> valid
        True
    """
    # Check version numbers are non-negative
    version_fields = [
        "fleet_portal_version_major",
        "fleet_portal_version_minor",
        "fleet_portal_version_patch",
        "master_portal_version_major",
        "master_portal_version_minor",
        "master_portal_version_patch",
        "device_apk_version_major",
        "device_apk_version_minor",
    ]

    for field in version_fields:
        value = attributes.get(field, 0)
        if not isinstance(value, int) or value < 0:
            return False, f"Invalid version field {field}: must be non-negative integer"

    # Check list fields are actually lists
    list_fields = [
        "device_models_in",
        "plans_in",
        "plans_nin",
        "event_type_in",
        "required_features",
        "category_in",
    ]

    for field in list_fields:
        if field in attributes and not isinstance(attributes[field], list):
            return False, f"Invalid field {field}: must be a list"

    return True, ""
