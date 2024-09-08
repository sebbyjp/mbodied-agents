from _typeshed import Incomplete
from mbodied.agents.backends import AnthropicBackend as AnthropicBackend, GradioBackend as GradioBackend, HttpxBackend as HttpxBackend, OllamaBackend as OllamaBackend, OpenAIBackend as OpenAIBackend
from mbodied.data import Sample as Sample
from mbodied.olddata.recording import Recorder as Recorder
from typing import Literal

Backend = AnthropicBackend | GradioBackend | OpenAIBackend | HttpxBackend | OllamaBackend

class Agent:
    """Abstract base class for agents.

    This class provides a template for creating agents that can
    optionally record their actions and observations.

    Attributes:
        recorder (Recorder): The recorder to record observations and actions.
        actor (Backend): The backend actor to perform actions.
        kwargs (dict): Additional arguments to pass to the recorder.
    """
    ACTOR_MAP: Incomplete
    @staticmethod
    def init_backend(model_src: str, model_kwargs: dict, api_key: str) -> type:
        """Initialize the backend based on the model source.

        Args:
            model_src: The model source to use.
            model_kwargs: The additional arguments to pass to the model.
            api_key: The API key to use for the remote actor.

        Returns:
            type: The backend class to use.
        """
    @staticmethod
    def handle_default(model_src: str, model_kwargs: dict) -> None:
        """Default to gradio then httpx backend if the model source is not recognized.

        Args:
            model_src: The model source to use.
            model_kwargs: The additional arguments to pass to the model.
        """
    recorder: Incomplete
    actor: Incomplete
    def __init__(self, recorder: Literal['omit', 'auto'] | str = 'omit', recorder_kwargs: Incomplete | None = None, api_key: str = None, model_src: Incomplete | None = None, model_kwargs: Incomplete | None = None) -> None:
        '''Initialize the agent, optionally setting up a recorder, remote actor, or loading a local model.

        Args:
            recorder: The recorder config or name to use for recording observations and actions.
            recorder_kwargs: Additional arguments to pass to the recorder.
            api_key: The API key to use for the remote actor (if applicable).
            model_src: The model or inference client or weights path to setup and preload if applicable.
                       You can pass in for example, "openai", "anthropic", "gradio", or a gradio endpoint,
                       or a path to a weights file.
            model_kwargs: Additional arguments to pass to the remote actor.
        '''
    def load_model(self, model: str) -> None:
        """Load a model from a file or path. Required if the model is a weights path.

        Args:
            model: The path to the model file.
        """
    def act(self, *args, **kwargs) -> Sample:
        """Act based on the observation.

        Subclass should implement this method.

        For remote actors, this method should call actor.act() correctly to perform the actions.
        """
    async def async_act(self, *args, **kwargs) -> Sample:
        """Act asynchronously based on the observation.

        Subclass should implement this method.

        For remote actors, this method should call actor.async_act() correctly to perform the actions.
        """
    def act_and_record(self, *args, **kwargs) -> Sample:
        """Peform action based on the observation and record the action, if applicable.

        Args:
            *args: Additional arguments to customize the action.
            **kwargs: Additional arguments to customize the action.

        Returns:
            Sample: The action sample created by the agent.
        """
    async def async_act_and_record(self, *args, **kwargs) -> Sample:
        """Act asynchronously based on the observation.

        Subclass should implement this method.

        For remote actors, this method should call actor.async_act() correctly to perform the actions.
        """
    @staticmethod
    def create_observation_from_args(observation_space, function, args, kwargs) -> dict:
        """Helper method to create an observation from the arguments of a function."""
