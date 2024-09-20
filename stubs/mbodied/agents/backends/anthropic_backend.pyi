from typing import Any

import anthropic
from _typeshed import Incomplete

from mbodied.agents.backends.openai_backend import OpenAIBackendMixin as OpenAIBackendMixin
from mbodied.agents.backends.serializer import Serializer as Serializer
from mbodied.types.message import Message as Message
from mbodied.types.sense.vision import Image as Image

class AnthropicSerializer(Serializer):
    """Serializer for Anthropic-specific data formats."""
    @classmethod
    def serialize_image(cls, image: Image) -> dict[str, Any]:
        """Serializes an image to the Anthropic format.

        Args:
            image: The image to be serialized.

        Returns:
            A dictionary representing the serialized image.
        """
    @classmethod
    def serialize_text(cls, text: str) -> dict[str, Any]:
        """Serializes a text string to the Anthropic format.

        Args:
            text: The text to be serialized.

        Returns:
            A dictionary representing the serialized text.
        """

class AnthropicBackend(OpenAIBackendMixin):
    """Backend for interacting with Anthropic's API.

    Attributes:
        api_key: The API key for the Anthropic service.
        client: The client for the Anthropic service.
        serialized: The serializer for Anthropic-specific data formats.
    """
    DEFAULT_MODEL: str
    INITIAL_CONTEXT: Incomplete
    api_key: Incomplete
    client: Incomplete
    model: Incomplete
    serialized: Incomplete
    def __init__(self, api_key: str | None, client: anthropic.Anthropic | None = None, **kwargs) -> None:
        """Initializes the AnthropicBackend with the given API key and client.

        Args:
            api_key: The API key for the Anthropic service.
            client: An optional client for the Anthropic service.
            kwargs: Additional keyword arguments.
        """
    def predict(self, message: Message, context: list[Message] | None = None, model: str | None = 'claude-3-5-sonnet-20240620', **kwargs) -> str:
        """Creates a completion for the given messages using the Anthropic API.

        Args:
            message: the message to be sent to the completion API.
            context: The context for the completion.
            model: The model to be used for the completion.
            **kwargs: Additional keyword arguments.

        Returns:
            str: The content of the completion response.
        """
    async def async_predict(self, message: Message, context: list[Message] | None = None, model: Any | None = None) -> str:
        """Asynchronously predict the next message in the conversation."""
