import os
from utils.tts_config import (
    MAX_TEXT_LENGTH, MIN_SPEED, MAX_SPEED, AUDIO_FORMATS,
    OPENAI_VOICES, GENDERS, TONES, GENDER_VOICES, TONE_INSTRUCTIONS,
    DEFAULT_VOICE, DEFAULT_FORMAT, DEFAULT_SPEED
)


def validate_text(text):
    """
    Validate text input for TTS.

    Args:
        text: The text to validate

    Returns:
        tuple: (is_valid: bool, error_message: str or None)
    """
    if not text or not isinstance(text, str):
        return False, "Invalid input: 'text' must be a non-empty string"

    text = text.strip()
    if len(text) == 0:
        return False, "Invalid input: 'text' must be a non-empty string"

    if len(text) > MAX_TEXT_LENGTH:
        return False, f"Invalid input: 'text' exceeds maximum length of {MAX_TEXT_LENGTH} characters"

    return True, None


def validate_voice_params(gender, tone, voice):
    """
    Validate voice-related parameters and resolve to OpenAI voice name.

    Args:
        gender: Gender selection (male/female) or None
        tone: Tone selection or None
        voice: Direct voice name or None

    Returns:
        tuple: (is_valid: bool, error_message: str or None, resolved_voice: str or None, tone: str or None)
    """
    # Check for conflicting parameters
    if (gender or tone) and voice:
        return False, "Cannot specify both 'voice' and 'gender/tone' parameters", None, None

    # Modern approach: gender + tone
    if gender or tone:
        if not gender:
            return False, "Gender must be specified when using tone", None, None

        if gender not in GENDERS:
            return False, f"Invalid gender: must be one of {GENDERS}", None, None

        # Validate tone if provided
        if tone and tone not in TONES:
            return False, f"Invalid tone: must be one of {TONES}", None, None

        # Select voice based on gender
        resolved_voice = GENDER_VOICES.get(gender, DEFAULT_VOICE)

        return True, None, resolved_voice, tone

    # Legacy approach: direct voice name
    if voice:
        if voice not in OPENAI_VOICES:
            return False, f"Invalid voice: must be one of {OPENAI_VOICES}", None, None
        return True, None, voice, None

    # No parameters provided, use default
    return True, None, DEFAULT_VOICE, None


def validate_format(format_param):
    """
    Validate audio format parameter.

    Args:
        format_param: The audio format to validate

    Returns:
        tuple: (is_valid: bool, error_message: str or None)
    """
    if format_param not in AUDIO_FORMATS:
        return False, f"Invalid format: must be one of {list(AUDIO_FORMATS.keys())}"
    return True, None


def validate_speed(speed):
    """
    Validate speed parameter.

    Args:
        speed: The speed value to validate

    Returns:
        tuple: (is_valid: bool, error_message: str or None)
    """
    try:
        speed_float = float(speed)
    except (TypeError, ValueError):
        return False, "Invalid speed: must be a number"

    if speed_float < MIN_SPEED or speed_float > MAX_SPEED:
        return False, f"Invalid speed: must be between {MIN_SPEED} and {MAX_SPEED}"

    return True, None


def validate_tts_request(data):
    """
    Validate complete TTS request and return validated parameters.

    Args:
        data: The request data dictionary

    Returns:
        tuple: (success: bool, error_message: str or None, validated_params: dict or None)
    """
    # Extract parameters
    text = data.get("text", "")
    gender = data.get("gender")
    tone = data.get("tone")
    voice = data.get("voice")
    format_param = data.get("format", DEFAULT_FORMAT)
    speed = data.get("speed", DEFAULT_SPEED)
    filename = data.get("filename")

    # Validate text
    is_valid, error = validate_text(text)
    if not is_valid:
        return False, error, None

    # Validate and resolve voice
    is_valid, error, resolved_voice, resolved_tone = validate_voice_params(gender, tone, voice)
    if not is_valid:
        return False, error, None

    # Validate format
    is_valid, error = validate_format(format_param)
    if not is_valid:
        return False, error, None

    # Validate speed
    is_valid, error = validate_speed(speed)
    if not is_valid:
        return False, error, None

    # Get tone instruction if tone is specified
    instructions = None
    if resolved_tone and resolved_tone in TONE_INSTRUCTIONS:
        instructions = TONE_INSTRUCTIONS[resolved_tone]

    # Generate download filename
    if filename:
        # Strip any user-provided extension and use format param extension
        filename_without_ext = os.path.splitext(filename)[0]
        download_filename = f"{filename_without_ext}.{format_param}"
    else:
        # Default: use "speech" as filename
        download_filename = f"speech.{format_param}"

    # Return validated parameters
    validated_params = {
        "text": text.strip(),
        "voice": resolved_voice,
        "format": format_param,
        "speed": float(speed),
        "instructions": instructions,
        "filename": download_filename
    }

    return True, None, validated_params
