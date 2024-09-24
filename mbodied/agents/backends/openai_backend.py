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


import os
import inspect

from typing_extensions import Any, List, Iterator, ParamSpec, dataclass_transform, Generic, TypedDict

import backoff
import httpx
from anthropic import RateLimitError as AnthropicRateLimitError
from openai._exceptions import RateLimitError as OpenAIRateLimitError

from mbodied.agents.backends.backend import Backend
from mbodied.agents.backends.serializer import Serializer
from mbodied.types.message import Message
from mbodied.types.sense.vision import Image


from openai import OpenAI, AsyncOpenAI
from openai.types.chat.chat_completion import ChatCompletion
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk
from openai.types.chat.completion_create_params import CompletionCreateParams


P = ParamSpec("P", bound=CompletionCreateParams)
@dataclass_transform()
class ChatCompletionParams(TypedDict, Generic[P]):
    pass


ERRORS = (
    OpenAIRateLimitError,
    AnthropicRateLimitError,
    httpx.HTTPError,
    ConnectionError,
)


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
        return {
            "type": "image_url",
            "image_url": {
                "url": image.url,
            },
        }

    @classmethod
    def serialize_text(cls, text: str) -> dict[str, Any]:
        """Serializes a text string to the OpenAI format.

        Args:
            text: The text to be serialized.

        Returns:
            A dictionary representing the serialized text.
        """
        return {"type": "text", "text": text}


class OpenAIBackend(Backend):
    """Backend for interacting with OpenAI's API.

    Attributes:
        api_key: The API key for the OpenAI service.
        client: The client for the OpenAI service.
        serialized: The serializer for the OpenAI backend.
        response_format: The format for the response.
    """

    INITIAL_CONTEXT = [
        Message(role="system", content="You are a robot with advanced spatial reasoning."),
    ]
    DEFAULT_MODEL = "gpt-4o"

    def __init__(
        self,
        api_key: str | None = None,
        response_format: str = None,
        aclient=False,
        **default_model_kwargs: ChatCompletionParams,
    ):
        """Initializes the OpenAIBackend with the given API key and client.

        Args:
            api_key: The API key for the OpenAI service.
            client: An optional client for the OpenAI service.
            response_format: The format for the response.
            aclient: Whether to use the asynchronous client.
            **kwargs: Additional keyword arguments.
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY") or os.getenv("MBODI_API_KEY")
        init_kwargs = {k: v for k, v in default_model_kwargs if k in inspect.signature(OpenAI.__init__).parameters}
        ainit_kwargs = {k: v for k, v in default_model_kwargs if k in inspect.signature(AsyncOpenAI.__init__).parameters}
        self.client = OpenAI(api_key=self.api_key, **init_kwargs)
        self.aclient = AsyncOpenAI(api_key=self.api_key, **ainit_kwargs)

        self.serialized = OpenAISerializer
        self.response_format = response_format

    @backoff.on_exception(
        backoff.expo,
        ERRORS,
        max_tries=3,
        on_backoff=lambda details: print(f"Backing off {details['wait']:.1f} seconds after {details['tries']} tries."),  # noqa
    )
    def predict(
        self, message: Message, context: List[Message] | None = None, model: Any | None = None, **kwargs: ChatCompletionParams,
    ) -> ChatCompletion:
        """Create a completion based on the given message and context.

        Args:
            message (Message): The message to process.
            context (Optional[List[Message]]): The context of messages.
            model (Optional[Any]): The model used for processing the messages.
            **kwargs: Additional keyword arguments.

        Returns:
            str: The result of the completion.
        """
        context = context or self.INITIAL_CONTEXT
        model = model or self.DEFAULT_MODEL
        from rich.pretty import pprint 
        pprint(message)
        serialized_messages = [self.serialized(msg).serialize() for msg in context + [message]]

        completion: ChatCompletion = self.client.chat.completions.create(
            model=model,
            messages=serialized_messages,
            temperature=0,
            max_tokens=1000,
            **kwargs,
        )
        return completion

    def stream(self, message: Message, context: List[Message] = None, model: str = "gpt-4o", **kwargs: ChatCompletionParams) ->  Iterator[ChatCompletionChunk]:
        """Streams a completion for the given messages using the OpenAI API standard.

        Args:
            message: Message to be sent to the completion API.
            context: The context of the messages.
            model: The model to be used for the completion.
            **kwargs: Additional keyword arguments.
        """
        print(f"Streaming completion for message: {message}")
        model = model or self.DEFAULT_MODEL
        context = context or self.INITIAL_CONTEXT
        serialized_messages = [self.serialized(msg).serialize() for msg in context + [message]]
        yield from self.client.chat.completions.create(
            messages=serialized_messages,
            model=model,
            temperature=0,
            stream=True,
            **kwargs,
        )
        

    async def astream(self, message: Message, context: List[Message] = None, model: str = "gpt-4o", **kwargs: ChatCompletionParams)-> Iterator[ChatCompletionChunk]:
        """Streams a completion asynchronously for the given messages using the OpenAI API standard.

        Args:
            message: Message to be sent to the completion API.
            context: The context of the messages.
            model: The model to be used for the completion.
            **kwargs: Additional keyword arguments.
        """
        if not hasattr(self, "aclient"):
            raise AttributeError("AsyncOpenAI client not initialized. Pass in aclient=True to the constructor.")
        model = model or self.DEFAULT_MODEL
        context = context or self.INITIAL_CONTEXT
        serialized_messages = [self.serialized(msg).serialize() for msg in context + [message]]
        chunks: Iterator[ChatCompletionChunk] = await self.aclient.chat.completions.create(
            messages=serialized_messages,
            model=model,
            temperature=0,
            stream=True,
            **kwargs,
        )
        return chunks

