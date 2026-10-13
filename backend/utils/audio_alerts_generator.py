from openai import OpenAI
from werkzeug.datastructures import FileStorage

class AudioAlertsGenerator:
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)
        self.api_key = api_key
        self.api_key = api_key

    def convert_text_to_speech(self, text: str, voice: str = "marin", format: str = "mp3", speed: float = 1.0, instructions: str = None):
        """
        Converts the given text to speech using OpenAI's audio generation capabilities.

        Args:
            text (str): The text to be converted to speech.
            voice (str): The voice to be used for speech synthesis. Default is "marin".
            format (str): The audio format for the output. Default is "mp3".
            speed (float): The speed of the generated speech. Default is 1.0.
            instructions (str): Optional instructions to guide the tone/style of speech.

        Returns:
            HttpxBinaryResponseContent: The response containing the generated audio data.
        """
        params = {
            "model": "gpt-4o-mini-tts",
            "voice": voice,
            "input": text,
            "response_format": format,
            "speed": speed
        }

        # Add instructions only if provided
        if instructions:
            params["instructions"] = instructions

        response = self.client.audio.speech.create(**params)
        return response

    def convert_speech_to_text(self, audio_file: FileStorage, translate_to_english: bool = False):

        """
        Converts speech from an audio file to text using OpenAI's transcription capabilities.

        Args:
            audio_file (str): Path to the audio file to be transcribed.
            translate_to_english (bool): Whether to translate the audio to English. Default is False.

        Returns:
            Translation or Transcription object: The response containing the transcribed or translated text.
        """

        audio_file_data = (audio_file.filename, audio_file.read(), audio_file.mimetype)

        if translate_to_english:
            response = self.client.audio.translations.create(
                model="whisper-1",
                file=audio_file_data
            )
        else:
            response = self.client.audio.transcriptions.create(
                model="gpt-4o-transcribe",
                file=audio_file_data
            )
        return response