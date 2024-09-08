from _typeshed import Incomplete
from mbodied.agents import Agent as Agent

class AudioAgent(Agent):
    '''Handles audio recording, playback, and speech-to-text transcription.

    This module uses OpenAI\'s API to transcribe audio input and synthesize speech.
    Set Environment Variable NO_AUDIO=1 to disable audio recording and playback.
    It will then take input from the terminal.

    Usage:
        audio_agent = AudioAgent(api_key="your-openai-api-key", use_pyaudio=False)
        audio_agent.speak("How can I help you?")
        message = audio_agent.listen()
    '''
    mode: Incomplete
    recording: bool
    record_lock: Incomplete
    listen_filename: Incomplete
    speak_filename: Incomplete
    use_pyaudio: Incomplete
    run_local: bool
    model: Incomplete
    client: Incomplete
    def __init__(self, listen_filename: str = 'tmp_listen.wav', tmp_speak_filename: str = 'tmp_speak.mp3', use_pyaudio: bool = True, api_key: str = None, run_local: bool = False) -> None:
        """Initializes the AudioAgent with specified parameters.

        Args:
            listen_filename: The filename for storing recorded audio.
            tmp_speak_filename: The filename for storing synthesized speech.
            use_pyaudio: Whether to use PyAudio for playback. Prefer setting to False for Mac.
            client: An optional OpenAI client instance.
            api_key: The API key for OpenAI.
            run_local: Whether to run the whisper model locally instead of using OpenAI.
        """
    def act(self, *args, **kwargs): ...
    def listen(self, keep_audio: bool = False, mode: str = 'speak') -> str:
        """Listens for audio input and transcribes it using OpenAI's API.

        Args:
            keep_audio: Whether to keep the recorded audio file.
            mode: The mode of input (speak, type, speak_or_type).

        Returns:
            The transcribed text from the audio input.
        """
    def record_audio(self) -> None:
        """Records audio from the microphone and saves it to a file."""
    playback_thread: Incomplete
    def speak(self, message: str, voice: str = 'onyx', api_key: str = None) -> None:
        """Synthesizes speech from text using OpenAI's API and plays it back.

        Args:
            message: The text message to synthesize.
            voice: The voice model to use for synthesis.
            api_key: The API key for OpenAI.
        """
    def play_audio(self, filename: str) -> None:
        """Plays an audio file.

        Args:
            filename: The filename of the audio file to play.
        """
