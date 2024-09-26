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

from typing_extensions import Any, Literal, Generic, TypeVar, Callable, ParamSpec, Self

from pydantic import Field, model_serializer
from pydantic.json_schema import SkipJsonSchema

from mbodied.types.sample import Sample
from mbodied.types.sense.vision import Image
from inspect import signature


Role = Literal["user", "assistant", "system"]


class Parameter(Sample):
    name: str
    type: type
    default: Sample | None = None
    resolved: bool = False
    """Whether or not the parameter has been resolved to a concrete value."""

P = ParamSpec("P")
T = TypeVar("T")
class Function(Sample, Generic[P, T]):
    name: str = ""
    arguments: dict[str, Parameter] = {}
    function_object: SkipJsonSchema[Callable[P, T]] | None = None

    @classmethod
    def from_callable(cls, func: Callable[P,T]) -> Self:
        arguments = {}
        for name, value in signature(func).parameters.items():
            arguments[name] = Parameter(name=name, type=value.annotation, default=value.default)
        return cls(name=func.__name__, arguments=arguments)

P = TypeVar("P", bound=Parameter)
class ToolCall(Sample, Generic[P]):
    function: Function
    # Can be an abstract parameter or a concrete value
    arguments: dict[str, P] = {}


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


class Choice(Sample):
    tool_calls: dict[str, ToolCall] | Any = Field(default_factory=dict)
    image: Image | None = None
    text: str = ""
    code: str = ""
    

class Message(Sample):
    """Single completion sample space.

    Message can be text, image, list of text/images, Sample, or other modality.

    Attributes:
        role: The role of the message sender (user, assistant, or system).
        content: The content of the message, which can be of various types.
    """
    role: Role = "user"
    content: Any | list[Choice] | list = Field(default_factory=list)
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
