import abc
from _typeshed import Incomplete
from gradio_client.client import Job as Job
from mbodied.agents.backends.backend import Backend as Backend

class GradioBackend(Backend, metaclass=abc.ABCMeta):
    """Gradio backend that handles connections to gradio servers."""
    endpoint: Incomplete
    client: Incomplete
    def __init__(self, endpoint: str = None, **kwargs) -> None:
        """Initializes the GradioBackend.

        Args:
            endpoint: The url of the gradio server.
            **kwargs: The keywrod arguments to pass to the gradio client.
        """
    def predict(self, *args, **kwargs) -> str:
        """Forward queries to the gradio api endpoint `predict`.

        Args:
            *args: The arguments to pass to the gradio server.
            **kwargs: The keywrod arguments to pass to the gradio server.
        """
    def submit(self, *args, api_name: str = '/predict', result_callbacks: Incomplete | None = None, **kwargs) -> Job:
        """Submit queries asynchronously without need of asyncio.

        Args:
            *args: The arguments to pass to the gradio server.
            api_name: The name of the api endpoint to submit the job.
            result_callbacks: The callbacks to apply to the result.
            **kwargs: The keywrod arguments to pass to the gradio server.

        Returns:
            Job: Gradio job object.
        """
