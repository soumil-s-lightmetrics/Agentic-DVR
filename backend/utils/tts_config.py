# TTS Configuration Constants

# Text constraints
MAX_TEXT_LENGTH = 4096  # OpenAI TTS API limit

# Speed constraints
MIN_SPEED = 0.25
MAX_SPEED = 4.0
DEFAULT_SPEED = 1.0

# Supported audio formats and their MIME types
AUDIO_FORMATS = {
    "mp3": "audio/mpeg",
    "opus": "audio/ogg",
    "aac": "audio/aac",
    "flac": "audio/flac",
    "wav": "audio/wav",
    "pcm": "audio/wave"
}

# Supported OpenAI voices
OPENAI_VOICES = [
    "alloy", "ash", "ballad", "coral", "echo", "fable",
    "onyx", "nova", "sage", "shimmer", "verse", "marin"
]

# Gender and tone configurations
GENDERS = ["male", "female"]

# Common tones available for all genders
TONES = ["neutral", "deep", "warm", "bright", "soft", "friendly", "authoritative", "calm"]

# Voice selection by gender - pick any voice, instructions will control tone
GENDER_VOICES = {
    "male": "onyx",    # Use onyx for male
    "female": "marin",   # Use marin for female
}

# Tone instructions - prepended to text to achieve desired tone
TONE_INSTRUCTIONS = {
    "neutral": "Speak in a neutral, clear tone.",
    "deep": "Speak in a deep, authoritative tone.",
    "warm": "Speak in a warm, friendly tone.",
    "bright": "Speak in a bright, enthusiastic tone.",
    "soft": "Speak in a soft, gentle tone.",
    "friendly": "Speak in a friendly, conversational tone.",
    "calm": "Speak in a calm, soothing tone.",
    "authoritative": "Speak in an authoritative, confident tone.",
}

# Default values
DEFAULT_VOICE = "onyx"  # Maintain backward compatibility
DEFAULT_FORMAT = "mp3"
