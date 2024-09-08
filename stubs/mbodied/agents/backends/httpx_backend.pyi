from typing import AsyncGenerator, Generator

from _typeshed import Incomplete

from mbodied.agents.backends.openai_backend import OpenAIBackendMixin as OpenAIBackendMixin
from mbodied.agents.backends.serializer import Serializer as Serializer
from mbodied.types.message import Message as Message
from mbodied.types.sense import Image as Image

class HttpxSerializer(Serializer):
    def __call__(self, messages: list[Message]) -> list[dict]: ...
    @classmethod
    def serialize_message(cls, message: Message) -> dict: ...
    @classmethod
    def serialize_content(cls, content) -> dict: ...
    @classmethod
    def serialize_image(cls, image: Image) -> dict: ...
    @classmethod
    def serialize_text(cls, text) -> dict: ...
    @classmethod
    def extract_response(cls, response) -> str: ...
    @classmethod
    def extract_stream(cls, response) -> str: ...

class HttpxBackend(OpenAIBackendMixin):
    SERIALIZER = HttpxSerializer
    DEFAULT_SRC: str
    DEFAULT_MODEL: str
    base_url: Incomplete
    api_key: Incomplete
    headers: Incomplete
    serialized: Incomplete
    kwargs: Incomplete
    def __init__(self, api_key: Incomplete | None = None, endpoint: str | None = None, serializer: Serializer | None = None, **kwargs) -> None:
        """Initializes the CompleteBackend. Defaults to using the API key from the environment and.

        Args:
            api_key (Optional[str]): The API key for the Complete service.
            endpoint (str): The base URL for the Complete API.
            serializer (Optional[Serializer]): The serializer to use for serializing messages.
        """
    def predict(self, messages: list[Message], model: str | None = None, **kwargs) -> str: ...
    def stream(self, messages: list[Message], model: str | None = None, **kwargs) -> Generator[str, None, None]: ...
    async def astream(self, messages: list[Message], model: str | None = None, **kwargs) -> AsyncGenerator[str, None]: ...
