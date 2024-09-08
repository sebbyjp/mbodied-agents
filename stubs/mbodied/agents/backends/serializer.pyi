from typing import Any

from pydantic import ConfigDict

from mbodied.types.message import Message as Message
from mbodied.types.sample import Sample as Sample
from mbodied.types.sense.vision import Image as Image

class Serializer(Sample):
    """A class to serialize messages and samples.

    This class provides a mechanism to serialize messages and samples into a dictionary format
    used by i.e. OpenAI, Anthropic, or other APIs.

    Attributes:
        wrapped: The message or sample to be serialized.
        model_config: The Pydantic configuration for the Serializer model.
    """
    wrapped: Any | None
    model_config: ConfigDict
    def __init__(self, wrapped: Message | Sample | list[Message] | None = None, *, message: Message | None = None, sample: Sample | None = None, **data) -> None:
        """Initializes the Serializer with various possible wrapped types.

        Args:
            wrapped: An instance of Message, Sample, a list of Messages, or None.
            message: An optional Message to be wrapped.
            sample: An optional Sample to be wrapped.
            **data: Additional data to initialize the Sample base class.

        """
    @classmethod
    def validate_model(cls, values: dict[str, Any]) -> dict[str, Any] | list[Any]:
        """Validates the 'wrapped' field of the model.

        Args:
            values: A dictionary of field values to validate.

        Returns:
            The validated values dictionary.

        Raises:
            ValueError: If the 'wrapped' field contains an invalid type.

        """
    def serialize_sample(self, sample: Any) -> dict[str, Any]:
        """Serializes a given sample.

        Args:
            sample: The sample to be serialized.

        Returns:
            A dictionary representing the serialized sample.

        """
    def serialize(self) -> dict[str, Any] | list[Any]:
        """Serializes the wrapped content of the Serializer instance.

        Returns:
            A dictionary representing the serialized wrapped content.

        """
    def serialize_msg(self, message: Message) -> dict[str, Any]:
        """Serializes a Message instance.

        Args:
            message: The Message to be serialized.

        Returns:
            A dictionary representing the serialized Message.

        """
    @classmethod
    def serialize_image(cls, image: Image) -> dict[str, Any]:
        """Serializes an Image instance.

        Args:
            image: The Image to be serialized.

        Returns:
            A dictionary representing the serialized Image.

        """
    @classmethod
    def serialize_text(cls, text: str) -> dict[str, Any]:
        """Serializes a text string.

        Args:
            text: The text to be serialized.

        Returns:
            A dictionary representing the serialized text.

        """
    def __call__(self) -> dict[str, Any] | list[Any]:
        """Calls the serialize method.

        Returns:
            A dictionary representing the serialized wrapped content.

        """
