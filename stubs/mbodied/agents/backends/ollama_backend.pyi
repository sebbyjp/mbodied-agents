from typing import Any

from _typeshed import Incomplete

from mbodied.agents.backends.httpx_backend import HttpxBackend as HttpxBackend
from mbodied.agents.backends.serializer import Serializer as Serializer
from mbodied.types.message import Message as Message
from mbodied.types.sense.vision import Image as Image

class OllamaSerializer(Serializer):
    """Serializer for Ollama-specific data formats."""
    @classmethod
    def serialize_image(cls, image: Image) -> str:
        """Serializes an image to the Ollama format."""
    @classmethod
    def serialize_text(cls, text: str) -> str:
        """Serializes a text string to the Ollama format."""
    @classmethod
    def serialize_msg(cls, message: Message) -> dict[str, Any]:
        """Serializes a message to the Ollama format."""
    @classmethod
    def extract_response(cls, response: dict[str, Any]) -> str:
        """Extracts the response from the Ollama format."""
    @classmethod
    def extract_stream(cls, response): ...

class OllamaBackend(HttpxBackend):
    """Backend for interacting with Ollama's API."""
    INITIAL_CONTEXT: Incomplete
    DEFAULT_MODEL: str
    SERIALIZER = OllamaSerializer
    DEFAULT_SRC: str
    def __init__(self, api_key: str | None = None, endpoint: str = None) -> None:
        """Initializes an OllamaBackend instance."""
