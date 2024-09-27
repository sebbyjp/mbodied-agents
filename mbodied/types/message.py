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

from inspect import signature
from typing import Mapping, MutableMapping, Protocol

from openai.types.beta.assistant_tool_choice_function import AssistantToolChoiceFunction
from openai.types.beta.function_tool_param import FunctionToolParam
from openai.types.chat.chat_completion import ChatCompletion, ChoiceLogprobs
from openai.types.chat.chat_completion import Choice as ChatCompletionChoice
from openai.types.chat.chat_completion_assistant_message_param import ChatCompletionAssistantMessageParam
from openai.types.chat.chat_completion_chunk import (
    ChatCompletionChunk,
    ChoiceDelta,
    ChoiceDeltaToolCall,
    ChoiceDeltaToolCallFunction,
)
from openai.types.chat.chat_completion_content_part_image_param import ChatCompletionContentPartImageParam
from openai.types.chat.chat_completion_content_part_text_param import ChatCompletionContentPartTextParam
from openai.types.chat.chat_completion_function_message_param import ChatCompletionFunctionMessageParam
from openai.types.chat.chat_completion_message_tool_call import ChatCompletionMessageToolCall
from openai.types.chat.chat_completion_token_logprob import ChatCompletionTokenLogprob
from openai.types.chat.chat_completion_tool_message_param import ChatCompletionToolMessageParam
from openai.types.shared.function_definition import FunctionDefinition
from openai.types.shared.function_parameters import FunctionParameters
from pydantic import BaseModel, ConfigDict, Field, model_serializer
from pydantic.json_schema import SkipJsonSchema
from typing_extensions import Any, Callable, Generic, Literal, ParamSpec, Self, TypeVar

from mbodied.types.sample import Sample
from mbodied.types.sense.vision import Image

Role = Literal["user", "assistant", "system"]

# How to implement a custom serializer or add a custom endpoint.


class Parameter(Sample):
    name: str
    type: type
    default: Sample | None = None
    resolved: bool = False
    """Whether or not the parameter has been resolved to a concrete value."""

P = ParamSpec("P")
T = TypeVar("T")
class Function(Sample):
    name: str = ""
    parameters: dict[str, Parameter] = {}
    function_object: SkipJsonSchema[Callable[P, T]] | None = None

    @classmethod
    def from_callable(cls, func: Callable[P,T]) -> Self:
        arguments = {}
        for name, value in signature(func).parameters.items():
            arguments[name] = Parameter(name=name, type=value.annotation, default=value.default)
        return cls(name=func.__name__, arguments=arguments)

class FunctionLike(Protocol):
    name: str
    parameters: dict[str, Parameter] | None

P = TypeVar("P")
class ToolCall(Sample, Generic[P]):
    function: Function
    # Can be an abstract parameter or a concrete value
    arguments: dict[str, P] = {}

    def __init__(self, function: FunctionLike | None = None, fn_name: str | None = None, parameters: dict | None = None, **arguments):
        self.function = Function(name=fn_name or getattr(function, "name", "anonymous"), parameters=parameters or getattr(function, "parameters", {}))
        self.arguments = arguments
        super().__init__()


    def resolved(self) -> bool:
        return all(v.resolved for v in self.args.values())

    @model_serializer(when_used="always")
    def serialize(self) -> dict:
        return {
            "function": self.function.model_dump(),
            "arguments": {k: v.model_dump() for k, v in self.arguments.items()},
        }

PT = TypeVar("PT", bound=Parameter | ToolCall)


class Resolved(Generic[PT]):
    def __class__getitem__(cls, p: PT) -> PT:
        # unwrap attributes of class until one of them is resolved
        unresolved = p
        while not hasattr(unresolved, "resolved"):
            for attr, value in unresolved.items():
                if attr == "resolved":
                    break
        if attr == "resolved" and not isinstance(value, Callable):
            setattr(unresolved, "resolved", True)
        return unresolved

class TaskCompletion(Sample):
    tool_calls: list[Resolved[ToolCall]] = Field(default_factory=list)
    image: Image | None = None
    text: str = ""
    code: str = ""

class ToolCalls(BaseModel):
    pass
    
class _Choice(BaseModel):
    model_config = ConfigDict(use_attribute_docstrings=True)
    tool_calls: ToolCalls = Field(default_factory=ToolCalls)
    image: Image | None = Field(default=None)
    text: str = Field(default="")
    code: str = Field(default="", description="The code to be executed.")
    """The code to be executed."""

    # def __init__(self, tool_calls: ChatCompletionChoice | ChoiceDelta | dict | None, **kwargs):
    #     choice = tool_calls
    #     if isinstance(choice, ChatCompletionChoice):
    #         tool_calls = {tc.function.name: tc.function.arguments for tc in choice.message.tool_calls}
    #     elif isinstance(choice, ChoiceDelta):
    #         tool_calls = {tc.function.name: tc.function.arguments for tc in choice.tool_calls}
    #     elif isinstance(choice, ChatCompletion):
    #         tool_calls = {tc.function.name: tc.function.arguments for tc in choice.choices[0].message.tool_calls}
    #     else:
    #         tool_calls = choice
    #     super().__init__(tool_calls=tool_calls, **kwargs)
    
c = _Choice()

from pydantic._internal._model_construction import ModelMetaclass

class ChoiceMeta(ModelMetaclass):
    @classmethod
    def __new__(cls,mcs, name, bases, namespace, **kwargs):
        return super().__new__(mcs, name, bases, namespace, **kwargs)


class Choice(ToolCall, metaclass=type):
    __signature__ = signature(_Choice)

c = Choice()

class Message(Sample):
    """Single completion sample space.

    Message can be text, image, list of text/images, Sample, or other modality.

    Attributes:
        role: The role of the message sender (user, assistant, or system).
        content: The content of the message, which can be of various types.
    """
    role: Role = "user"
    content: Any | str | Choice | ToolCall | list
    choices: list[Choice] | None = None
    """This will only be used if multiple responses are requested."""


    @classmethod
    def supports(cls, arg: Any) -> bool:
        """Checks if the argument type is supported by the Message class.

        Args:
            arg: The argument to be checked.

        Returns:
            True if the argument type is supported, False otherwise.
        """
        return Image.supports(arg) or isinstance(arg, str | list | Sample | tuple | dict)

    def __init__(
        self,
        content: Any | None = None,
        role: Role = "user",
        choices: list[Choice] | None = None,
    ):
        """Initializes a Message instance.

        Args:
            content: The content of the message, which can be of various types.
            role: The role of the message sender (default is "user").
        """
        data = {"role": role}
        if content is not None and not isinstance(content, list):
            content = [content]
        if choices is not None and not isinstance(choices, list):
            choices = [choices]
        if choices is not None:
            data["choices"] = choices
        data["content"] = content
        super().__init__(**data)
