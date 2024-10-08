# Copyright 2024 mbodi ai
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from gradio_client import Client
from gradio_client.client import DEFAULT_TEMP_DIR
from gradio_client.client import Job
from typing_extensions import Literal, Any, Protocol, ParamSpec, TypedDict
from pathlib import Path
from mbodied.agents.backends.backend import Backend


P = ParamSpec("P", bound=TypedDict)
class GradioParams(Protocol[P]):    
    src: str
    hf_token: str | None = None
    max_workers: int = 40
    verbose: bool = True
    auth: tuple[str, str] | None = None
    httpx_kwargs: dict[str, Any] | None = None
    headers: dict[str, str] | None = None
    download_files: str | Path | Literal[False] = DEFAULT_TEMP_DIR,
    ssl_verify: bool = True
    _skip_components: bool = True


class GradioBackend(Backend):
    """Gradio backend that handles connections to gradio servers."""

    def __init__(
        self,
        endpoint: str = None,
        **kwargs,
    ) -> None:
        """Initializes the GradioBackend.

        Args:
            endpoint: The url of the gradio server.
            **kwargs: The keywrod arguments to pass to the gradio client.
        """
        self.endpoint = endpoint
        self.client = Client(src=endpoint, **kwargs)

    def predict(self, *args, **kwargs) -> str:
        """Forward queries to the gradio api endpoint `predict`.

        Args:
            *args: The arguments to pass to the gradio server.
            **kwargs: The keywrod arguments to pass to the gradio server.
        """
        return self.client.predict(*args, **kwargs)

    def submit(self, *args, api_name="/predict", result_callbacks=None, **kwargs) -> Job:
        """Submit queries asynchronously without need of asyncio.

        Args:
            *args: The arguments to pass to the gradio server.
            api_name: The name of the api endpoint to submit the job.
            result_callbacks: The callbacks to apply to the result.
            **kwargs: The keywrod arguments to pass to the gradio server.

        Returns:
            Job: Gradio job object.
        """
        return self.client.submit(api_name=api_name, result_callbacks=result_callbacks, *args, **kwargs)
