from typing import Any

from _typeshed import Incomplete

from mbodied.agents.backends.backend import Backend as Backend
from mbodied.agents.backends.serializer import Serializer as Serializer
from mbodied.types.message import Message as Message
from mbodied.types.sense.vision import Image as Image

ERRORS: Incomplete

class OpenAISerializer(Serializer):
    """Serializer for OpenAI-specific data formats."""
    @classmethod
    def serialize_image(cls, image: Image) -> dict[str, Any]:
        """Serializes an image to the OpenAI format.

        Args:
            image: The image to be serialized.

        Returns:
            A dictionary representing the serialized image.
        """
    @classmethod
    def serialize_text(cls, text: str) -> dict[str, Any]:
        """Serializes a text string to the OpenAI format.

        Args:
            text: The text to be serialized.

        Returns:
            A dictionary representing the serialized text.
        """

class OpenAIBackendMixin(Backend):
    """Backend for interacting with OpenAI's API.

    Attributes:
        api_key: The API key for the OpenAI service.
        client: The client for the OpenAI service.
        serialized: The serializer for the OpenAI backend.
        response_format: The format for the response.
    """
    INITIAL_CONTEXT: Incomplete
    DEFAULT_MODEL: str
    api_key: Incomplete
    client: Incomplete
    aclient: Incomplete
    serialized: Incomplete
    response_format: Incomplete
    def __init__(self, api_key: str | None = None, client: Any | None = None, response_format: str = None, aclient: bool = False, **kwargs) -> None:
        """Initializes the OpenAIBackend with the given API key and client.

        Args:
            api_key: The API key for the OpenAI service.
            client: An optional client for the OpenAI service.
            response_format: The format for the response.
            aclient: Whether to use the asynchronous client.
            **kwargs: Additional keyword arguments.
        """
    def predict(self, message: Message, context: list[Message] | None = None, model: Any | None = None, **kwargs) -> str:
        """Create a completion based on the given message and context.

        Args:
            message (Message): The message to process.
            context (Optional[List[Message]]): The context of messages.
            model (Optional[Any]): The model used for processing the messages.
            **kwargs: Additional keyword arguments.

        Returns:
            str: The result of the completion.
        """
    def stream(self, message: Message, context: list[Message] = None, model: str = 'gpt-4o', **kwargs):
        """Streams a completion for the given messages using the OpenAI API standard.

        Args:
            messages: A list of messages to be sent to the completion API.
            context: The context of the messages.
            model: The model to be used for the completion.
            **kwargs: Additional keyword arguments.
        """
    async def astream(self, message: Message, context: list[Message] = None, model: str = 'gpt-4o', **kwargs):
        """Streams a completion asynchronously for the given messages using the OpenAI API standard.

        Args:
            messages: A list of messages to be sent to the completion API.
            context: The context of the messages.
            model: The model to be used for the completion.
            **kwargs: Additional keyword arguments.
        """
