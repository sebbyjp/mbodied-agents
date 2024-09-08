from _typeshed import Incomplete
from mbodied.types.sample import Sample as Sample
from mbodied.types.sense.vision import Image as Image
from typing import Any

Role: Incomplete

class Message(Sample):
    """Single completion sample space.

    Message can be text, image, list of text/images, Sample, or other modality.

    Attributes:
        role: The role of the message sender (user, assistant, or system).
        content: The content of the message, which can be of various types.
    """
    role: Role
    content: Any | None
    @classmethod
    def supports(cls, arg: Any) -> bool:
        """Checks if the argument type is supported by the Message class.

        Args:
            arg: The argument to be checked.

        Returns:
            True if the argument type is supported, False otherwise.
        """
    def __init__(self, content: Any | None = None, role: Role = 'user') -> None:
        '''Initializes a Message instance.

        Args:
            content: The content of the message, which can be of various types.
            role: The role of the message sender (default is "user").
        '''
