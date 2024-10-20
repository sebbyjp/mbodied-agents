import asyncio
import json
import logging as log
import signal

import numpy as np
import soundfile as sf
from websockets.client import connect
from mbodied.agents import Agent





WEBSOCKET_URI = "ws://149.130.215.195:5018/audio/speech"
DEFAULT_SAMPLE_RATE = 32767

class WebSocketClient:
    def __init__(self, sr: int = DEFAULT_SAMPLE_RATE, endpoint: str = WEBSOCKET_URI, model: str | None = None):
        self.sr = sr
        self.endpoint = endpoint
        self.model = model

    async def stream_speech_async(self, text: str) -> None:
        """Async method to stream audio data to the server and handle audio responses."""
        async with connect(self.endpoint) as websocket:
            await websocket.send(text)

            # Close the connection when receiving SIGTERM.
            loop = asyncio.get_running_loop()
            loop.add_signal_handler(signal.SIGTERM, loop.create_task, websocket.close())

            # Process messages received on the connection.
            audio = np.array([0], dtype=np.int16)
            async for message in websocket:
                if isinstance(message, bytes):
                    audio = np.append(audio, np.frombuffer(message, dtype=np.int16))
                    sf.write("audio_async.wav", audio, self.sr)
                    print("Saved audio (async)")
                elif isinstance(message, str):
                    try:
                        json.loads(message)
                    except json.JSONDecodeError:
                        log.error(f"Unexpected message: {message}")
                        print(f"message: {message}")
                    break
                else:
                    log.error(f"Unexpected message: {message}")
                    break
            print("Done")

    def stream_speech_sync(self, text: str) -> None:
        """Sync method to stream audio data to the server by wrapping the async method."""
        loop = asyncio.get_event_loop()
        loop.run_until_complete(self.stream_speech_async(text))


# Sync and Async clients
class Speaker(Agent):
    def __init__(self, sr: int = DEFAULT_SAMPLE_RATE, endpoint: str = WEBSOCKET_URI, model: str | None = None):
        self.async_client = WebSocketClient(sr=sr, endpoint=endpoint, model=model)

    def stream_speech_async(self, text: str) -> None:
        """Run the async speech client."""
        asyncio.run(self.async_client.stream_speech_async(text))

    def stream_speech_sync(self, text: str) -> None:
        """Run the sync speech client."""
        self.async_client.stream_speech_sync(text)
    
    def act(self, message: str) -> None:
        """Act on the received message."""
        self.stream_speech_sync(message)


# Reusable client for sync and async calls
speech_client = SpeechClient(sr=22050, endpoint=WEBSOCKET_URI, model="your_model_name")


# Example for sync client
def sync_client_example():
    speech_client.stream_speech_sync("Hello, this is a synchronous test.")


# Example for async client
async def async_client_example():
    await speech_client.stream_speech_async("Hello, this is an asynchronous test.")



class Speaker(Agent):
  def __init__(self, sr: int = DEFAULT_SAMPLE_RATE, endpoint: str = WEBSOCKET_URI, model: str | None = None):
    self.sr = sr
    self.endpoint = endpoint
    self.model = model
    self.client

# Running the sync client
if __name__ == "__main__":
    # Run sync example
    sync_client_example()

    # To run async example, uncomment the below:
    # asyncio.run(async_client_example())