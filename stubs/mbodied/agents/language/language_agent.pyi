from dataclasses import dataclass
from typing import AsyncGenerator, Generator, Literal, TypeAlias

from _typeshed import Incomplete
from pydantic import AnyUrl as AnyUrl
from pydantic import DirectoryPath as DirectoryPath
from pydantic import FilePath as FilePath
from pydantic import NewPath as NewPath

from mbodied.agents import Agent as Agent
from mbodied.agents.backends import OpenAIBackend as OpenAIBackend
from mbodied.types.message import Message as Message
from mbodied.types.sample import Sample as Sample
from mbodied.types.sense.vision import Image as Image

SupportsOpenAI: TypeAlias

@dataclass
class Reminder:
    """A reminder to show the agent a prompt every n messages."""
    prompt: str | Image | Message
    n: int
    def __iter__(self): ...
    def __getitem__(self, key): ...
    def __init__(self, prompt, n) -> None: ...

def make_context_list(context: list[str | Image | Message] | Image | str | Message | None) -> list[Message]:
    """Convert the context to a list of messages."""

class LanguageAgent(Agent):
    """An agent that can interact with users using natural language.

    This class extends the functionality of a base Agent to handle natural language interactions. It manages memory, dataset-recording, and asynchronous remote inference, supporting multiple platforms including OpenAI, Anthropic, and Gradio.

    Attributes:
        reminders (List[Reminder]): A list of reminders that prompt the agent every n messages.
        context (List[Message]): The current context of the conversation.
        Inherits all attributes from the parent class `Agent`.

    Examples:
        Basic usage with OpenAI:
            >>> cognitive_agent = LanguageAgent(api_key="...", model_src="openai", recorder="default")
            >>> cognitive_agent.act("your instruction", image)

        Automatically act and record to dataset:
            >>> cognitive_agent.act_and_record("your instruction", image)
    """
    reminders: Incomplete
    context: Incomplete
    def __init__(self, model_src: Literal['openai', 'anthropic', 'gradio', 'ollama', 'http'] | AnyUrl | FilePath | DirectoryPath | NewPath = 'openai', context: list | Image | str | Message = None, api_key: str | None = ..., model_kwargs: dict = None, recorder: Literal['default', 'omit'] | str = 'omit', recorder_kwargs: dict = None) -> None:
        r"""Agent with memory,  asynchronous remote acting, and automatic dataset recording.

         Additionally supports asynchronous remote inference,
            supporting multiple platforms including OpenAI, Anthropic, vLLM, Gradio, and Ollama.

        Args:
            model_src: The source of the model to use for inference. It can be one of the following:
                - "openai": Use the OpenAI backend (or vLLM).
                - "anthropic": Use the Anthropic backend.
                - "gradio": Use the Gradio backend.
                - "ollama": Use the Ollama backend.
                - "http": Use a custom HTTP API backend.
                - AnyUrl: A URL pointing to the model source.
                - FilePath: A local path to the model\'s weights.
                - DirectoryPath: A local directory containing the model\'s weights.
                - NewPath: A new path object representing the model source.
            context (Union[list, Image, str, Message], optional): The starting context to use for the conversation.
                    It can be a list of messages, an image, a string, or a message.
                    If a string is provided, it will be interpreted as a user message. Defaults to None.
            api_key (str, optional): The API key to use for the remote actor (if applicable).
                 Defaults to the value of the OPENAI_API_KEY environment variable.
            model_kwargs (dict, optional): Additional keyword arguments to pass to the model source.
                See the documentation of the specific backend for more details. Defaults to None.
            recorder (Union[str, Literal["default", "omit"]], optional):
                The recorder configuration or name or action. Defaults to "omit".
            recorder_kwargs (dict, optional): Additional keyword arguments to pass to the recorder. Defaults to None.
        """
    def forget_last(self) -> Message:
        """Forget the last message in the context."""
    def forget_after(self, first_n: int) -> None:
        """Forget after the first n messages in the context.

        Args:
            first_n: The number of messages to keep.
        """
    def forget(self, everything: bool = False, last_n: int = -1) -> list[Message]:
        """Forget the last n messages in the context.

        Args:
            everything: Whether to forget everything.
            last_n: The number of messages to forget.
        """
    def history(self) -> list[Message]:
        """Return the conversation history."""
    def remind_every(self, prompt: str | Image | Message, n: int) -> None:
        """Remind the agent of the prompt every n messages.

        Args:
            prompt: The prompt to remind the agent of.
            n: The frequency of the reminder.
        """
    def act_and_parse(self, instruction: str, image: Image = None, parse_target: type[Sample] = ..., context: list | str | Image | Message = None, model: Incomplete | None = None, max_retries: int = 1, record: bool = False, **kwargs) -> Sample:
        """Responds to the given instruction, image, and context and parses the response into a Sample object.

        Args:
            instruction: The instruction to be processed.
            image: The image to be processed.
            parse_target: The target type to parse the response into.
            context: Additonal context to include in the response. If context is a list of messages, it will be interpreted
                as new memory.
            model: The model to use for the response.
            max_retries: The maximum number of retries to parse the response.
            record: Whether to record the interaction for training.
            **kwargs: Additional keyword arguments.
        """
    async def async_act_and_parse(self, instruction: str, image: Image = None, parse_target: Sample = ..., context: list | str | Image | Message = None, model: Incomplete | None = None, max_retries: int = 1, **kwargs) -> Sample:
        """Responds to the given instruction, image, and context asynchronously and parses the response into a Sample object."""
    def prepare_inputs(self, instruction: str, image: Image = None, context: list | str | Image | Message = None) -> tuple[Message, list[Message]]:
        """Helper method to prepare the inputs for the agent.

        Args:
            instruction: The instruction to be processed.
            image: The image to be processed.
            context: Additonal context to include in the response. If context is a list of messages, it will be interpreted
                as new memory.
        """
    def postprocess_response(self, response: str, message: Message, memory: list[Message], **kwargs) -> str:
        """Postprocess the response."""
    def act(self, instruction: str, image: Image = None, context: list | str | Image | Message = None, model: Incomplete | None = None, **kwargs) -> str:
        '''Responds to the given instruction, image, and context.

        Uses the given instruction and image to perform an action.

        Args:
            instruction: The instruction to be processed.
            image: The image to be processed.
            context: Additonal context to include in the response. If context is a list of messages, it will be interpreted
                as new memory.
            model: The model to use for the response.
            **kwargs: Additional keyword arguments.

        Returns:
            str: The response to the instruction.

        Example:
            >>> agent.act("Hello, world!", Image("scene.jpeg"))
            "Hello! What can I do for you today?"
            >>> agent.act("Return a plan to pickup the object as a python list.", Image("scene.jpeg"))
            "[\'Move left arm to the object\', \'Move right arm to the object\']"
        '''
    def act_and_stream(self, instruction: str, image: Image = None, context: list | str | Image | Message = None, model: Incomplete | None = None, **kwargs) -> Generator[str, None, str]:
        """Responds to the given instruction, image, and context and streams the response."""
    async def async_act_and_stream(self, instruction: str, image: Image = None, context: list | str | Image | Message = None, model: Incomplete | None = None, **kwargs) -> AsyncGenerator[str, None]: ...

def main() -> None: ...
async def async_main() -> None: ...
