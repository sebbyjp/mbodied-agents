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

"""Run a LanguageAgent with memory, optional remote acting, and optional automatic dataset creation capabilities.

While it is always recommended to explicitly define your observation and action spaces,
which can be set with a gym.Space object or any python object using the Sample class
(see examples/using_sample.py for a tutorial), you can have the recorder infer the spaces
by setting recorder="default" for automatic dataset recording.

Examples:
    >>> agent = LanguageAgent(context=SYSTEM_PROMPT, model_src=backend, recorder="default")
    >>> agent.act_and_record("pick up the fork", image)

Alternatively, you can define the recorder separately to record the space you want.
For example, to record the dataset with the image and instruction observation and AnswerAndActionsList as action.

Examples:
    >>> observation_space = spaces.Dict({"image": Image(size=(224, 224)).space(), "instruction": spaces.Text(1000)})
    >>> action_space = AnswerAndActionsList(actions=[HandControl()] * 6).space()
    >>> recorder = Recorder(
    ...     'example_recorder',
    ...     out_dir='saved_datasets',
    ...     observation_space=observation_space,
    ...     action_space=action_space

To record the dataset, you can use the record method of the recorder object.

Examples:
    >>> recorder.record(
    ...     observation={
    ...         "image": image,
    ...         "instruction": instruction,
    ...     },
    ...     action=answer_actions,
    ... )
"""

import asyncio
import inspect
import json
import logging
import os
from dataclasses import dataclass

from art import text2art
from more_itertools import always_iterable as alwaysiter
from pydantic import AnyUrl, DirectoryPath, FilePath, NewPath
from typing_extensions import AsyncGenerator, Generator, Iterator, List, Literal, TypeAlias

from mbodied.agents import Agent
from mbodied.agents.backends import OpenAIBackend
from mbodied.agents.backends.gradio_backend import GradioParams
from mbodied.agents.backends.openai_backend import ChatCompletion, ChatCompletionChunk, ChatCompletionParams
from mbodied.data.recording import RecorderParams
from mbodied.types.message import Choice, Message, ToolCall
from mbodied.types.message import FunctionDefinition as Function
from mbodied.types.sample import Sample
from mbodied.types.sense.vision import Image

SupportsOpenAI: TypeAlias = OpenAIBackend


@dataclass
class Reminder:
    """A reminder to show the agent a prompt every n messages."""

    prompt: str | Image | Message
    n: int

    def __iter__(self):
        yield self.prompt
        yield self.n

    def __getitem__(self, key):
        if key == 0:
            return self.prompt
        elif key == 1:
            return self.n
        else:
            raise IndexError("Invalid index")


def make_context_list(context: list[str | Image | Message] | Image | str | Message | None) -> List[Message]:
    """Convert the context to a list of messages."""
    if isinstance(context, list):
        return [Message(content=c) if not isinstance(c, Message) else c for c in context]
    if isinstance(context, Message):
        return [context]
    if isinstance(context, str | Image):
        return [Message(role="user", content=[context]), Message(role="assistant", content="Understood.")]
    return []


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

    _art_printed = False

    def __init__(
        self,
        model_src: Literal["openai", "anthropic", "gradio", "ollama", "http"]
        | AnyUrl
        | FilePath
        | DirectoryPath
        | NewPath = "openai",
        context: list | Image | str | Message = None,
        api_key: str | None = os.getenv("OPENAI_API_KEY"),
        recorder_kwargs: RecorderParams = None,
        **model_kwargs: ChatCompletionParams | GradioParams | dict,
    ) -> None:
        """Agent with memory,  asynchronous remote acting, and automatic dataset recording.

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
                - FilePath: A local path to the model's weights.
                - DirectoryPath: A local directory containing the model's weights.
                - NewPath: A new path object representing the model source.
            context (Union[list, Image, str, Message], optional): The starting context to use for the conversation.
                    It can be a list of messages, an image, a string, or a message.
                    If a string is provided, it will be interpreted as a user message. Defaults to None.
            api_key (str, optional): The API key to use for the remote actor (if applicable).
                 Defaults to the value of the OPENAI_API_KEY environment variable.
            model_kwargs (dict, optional): Additional keyword arguments to pass to the model source such as default model parameters.
                See the documentation of the specific backend for more details. Defaults to None.
            recorder (Union[str, Literal["default", "omit"]], optional):
                The recorder configuration or name or action. Defaults to "omit".
            recorder_kwargs (dict, optional): Additional keyword arguments to pass to the recorder. Defaults to None.
        """
        if not globals().get("LanguageAgent._art_printed", False):
            print("Welcome to")  # noqa: T201
            print(text2art("mbodi"))  # noqa: T201
            print("A platform for intelligent embodied agents.\n\n")  # noqa: T201
            globals()["LanguageAgent._art_printed"] = True  # Don't print the art again
        self.reminders: List[Reminder] = []
        print(f"Initializing language agent for robot using : {model_src}")  # noqa: T201
        model_kwargs.update({"api_key": api_key or os.getenv("OPENAI_API_KEY")})
        super().__init__(
            model_src=model_src,
            recorder_kwargs=recorder_kwargs,
            **model_kwargs,
        )
        self.default_model_kwargs = {
            k: v for k, v in model_kwargs.items() if k in inspect.signature(self.actor.predict).parameters
        }
        self.context = make_context_list(context)

    def forget_last(self) -> Message:
        """Forget the last message in the context."""
        try:
            return self.context.pop(-1)
        except IndexError:
            logging.warning("No message to forget in the context")

    def forget_after(self, first_n: int) -> None:
        """Forget after the first n messages in the context.

        Args:
            first_n: The number of messages to keep.
        """
        self.context = self.context[:first_n]

    def forget(self, everything=False, last_n: int = -1) -> List[Message]:
        """Forget the last n messages in the context.

        Args:
            everything: Whether to forget everything.
            last_n: The number of messages to forget.
        """
        if everything:
            context = self.context
            self.context = []
            return context
        forgotten = []
        for _ in range(last_n):
            last = self.forget_last()
            if last:
                forgotten.append(last)
        return forgotten

    def history(self) -> List[Message]:
        """Return the conversation history."""
        return self.context

    def remind_every(self, prompt: str | Image | Message, n: int) -> None:
        """Remind the agent of the prompt every n messages.

        Args:
            prompt: The prompt to remind the agent of.
            n: The frequency of the reminder.
        """
        message = Message([prompt]) if not isinstance(prompt, Message) else prompt
        self.reminders.append(Reminder(message, n))

    def _check_for_reminders(self) -> None:
        """Check if there are any reminders to show."""
        for reminder, n in self.reminders:
            if len(self.context) % n == 0:
                self.context.append(reminder)

    def act_and_parse(
        self,
        instruction: str,
        image: Image = None,
        parse_target: type[Sample] = Sample,
        context: list | str | Image | Message = None,
        model=None,
        max_retries: int = 1,
        record: bool = False,
        **model_kwargs: ChatCompletionParams | GradioParams,
    ) -> Sample:
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
        original_instruction = instruction
        kwargs = {**self.default_model_kwargs, **model_kwargs}
        for attempt in range(max_retries + 1):
            if record:
                response = self.act_and_record(instruction, image, context, model, **kwargs)
            else:
                response = self.act(instruction, image, context, model, **kwargs)
            response = response[response.find("{") : response.rfind("}") + 1]
            try:
                return parse_target.model_validate_json(response)
            except Exception as e:
                if attempt == max_retries:
                    raise ValueError(f"Failed to parse response after {max_retries + 1} attempts") from e
                error = f"Error parsing response: {e}"
                instruction = original_instruction + f". Avoid the following error: {error}"
                self.forget(last_n=2)
                logging.warning(f"\nReceived response: {response}.\n Retrying with error message: {instruction}")
        raise ValueError(f"Failed to parse response after {max_retries + 1} attempts")

    async def async_act_and_parse(
        self,
        instruction: str,
        image: Image = None,
        parse_target: Sample = Sample,
        context: list | str | Image | Message = None,
        model=None,
        max_retries: int = 1,
        **model_kwargs: ChatCompletionParams | GradioParams,
    ) -> Sample:
        """Responds to the given instruction, image, and context asynchronously and parses the response into a Sample object."""
        return await asyncio.to_thread(
            self.act_and_parse,
            instruction,
            image,
            parse_target,
            context,
            model=model,
            max_retries=max_retries,
            **model_kwargs,
        )

    def prepare_inputs(
        self, instruction: str, image: Image = None, context: list | str | Image | Message = None
    ) -> tuple[Message, list[Message]]:
        """Helper method to prepare the inputs for the agent.

        Args:
            instruction: The instruction to be processed.
            image: The image to be processed.
            context: Additonal context to include in the response. If context is a list of messages, it will be interpreted
                as new memory.
        """
        self._check_for_reminders()
        memory = self.context
        if context and all(isinstance(c, Message) for c in context):
            memory += context
            context = []

        # Prepare the inputs
        inputs = [instruction]
        if image is not None:
            inputs.append(image)
        if context:
            inputs.extend(context if isinstance(context, list) else [context])
        message = Message(role="user", content=inputs)

        return message, memory

    def on_completion_finish(
        self, response: ChatCompletion, message: Message, memory: list[Message], **kwargs: ChatCompletionParams
    ) -> str | list[str] | tuple[str, ToolCall] | ToolCall:
        """Postprocess the response."""
        from rich.pretty import pprint

        pprint(response)
        pprint("streaming_response")
        pprint(getattr(self, "streaming_response", None))
        if hasattr(self, "streaming_response") and self.streaming_response is not None:
            response: Message = self.streaming_response
        else:
            if kwargs.get("n", 1) == 1:
                response = Message(role="assistant", choices=response.choices[0].message.content)
            elif not response.choices:
                raise ValueError("No response choices found.")
            elif "tools" in kwargs and kwargs.get("tool_choice") == "required":
                response = Message(role="assistant", content=[ToolCall(c.message.tool_calls) for c in response.choices])

        if not response.choices and hasattr(self, "streaming_response"):
            content = response.content
            self.context.extend([message, Message(role="assistant", content=content)])
            return self.streaming_response.content

        text, response_message = (
            response.content,
            Message(
                role="assistant",
                content=response.content or "Tool call.",
                choices=[Choice(choice) for choice in response.choices],
            ),
        )
        # print(f"Response: {response_message}")
        self.context.extend([message, response_message])
        # print(f"Tool calls: {response_message.choices[0].tool_calls}")
        del self.streaming_response
        if kwargs.get("tools") and text:
            return text, response_message.choices[0].tool_calls
        if kwargs.get("tools"):
            return response_message.choices[0].tool_calls

        return text
        # choices = None
        # if hasattr(self, "streaming_response"):
        #     if hasattr(self.streaming_response, "choices"):
        #         if self.streaming_response.choices and isinstance(self.streaming_response.choices[0], str):
        #             self.streaming_response.choices = json.loads(self.streaming_response.choices[0])
        #           choices = self.streaming_response.choices

        if isinstance(response, str):
            self.context.extend([message, Message(role="assistant", content=response)])
            return response

        # if hasattr(self, "streaming_response"):
        #     # content = self.streaming_response.content if  hasattr(self.streaming_response, "content") else response.choices[0].message.content
        #     self.context.extend(
        #         [
        #             message,
        #             Message(
        #                 role="assistant",
        #                 content=self.streaming_response.content,
        #                 choices=self.streaming_response.choices,
        #             ),
        #         ]
        #     )
        # else:
        #     content = response.choices[0].message.content
        #     self.context.extend([message, Message(role="assistant", content=content)])
        # self.context[-1] = Message(role="assistant", content=response.choices[0].message.content)
        # if self.context[-1].choices:
        #     pprint(f"response:")
        #     pprint(response)
        #     self.context[-1].choices = [Choice(tool_calls=c.message.tool_calls) for c in response.choices]
        # pprint(f"len of context: {len(self.context)}")
        # pprint(self.context)

        # pprint(self.context)
        # # # Clean up the streaming response if appropriate.
        # if hasattr(self, "streaming_response"):
        #     del self.streaming_response

        if kwargs.get("n", 1) == 1:
            text, response_message = (
                response.choices[0].message.content,
                Message(
                    role="assistant",
                    content="Tool call.",
                    choices=[
                        Choice(
                            tool_calls={
                                tc.function.name: json.loads(tc.function.arguments) for tc in c.message.tool_calls
                            }
                        )
                        for c in response.choices
                    ]
                    if response.choices[0].message.tool_calls is not None
                    else None,
                ),
            )
            self.context.extend([message, response_message])
            if kwargs.get("tools"):
                if text:
                    return text, response_message.choices[0].tool_calls
                return response_message.choices[0].tool_calls
            return text

        # pprint(self.context)
        # print(f"reponse.choices: ")
        # from rich.pretty import pprint
        # pprint("CONTEXT")
        # pprint(self.context)
        # pprint("RESPONSE")
        # pprint(response)
        text, tool_calls = (
            [c.message.content for c in response.choices],
            Message(
                role="assistant",
                content="Tool call.",
                choices=[
                    Choice(tool_calls={tc.function.name: tc.function.arguments for tc in c.message.tool_calls})
                    for c in response.choices
                ],
            ),
        )
        self.context.extend([message, tool_calls])
        if kwargs.get("tools") and text:
            return text, tool_calls.choices
        if kwargs.get("tools"):
            return tool_calls.choices
        return text

    # globals()["choice_count"] = 0

    def on_stream_yield(
        self,
        response_chunk: ChatCompletionChunk | Iterator[ChatCompletionChunk],
        message: Message,
        memory: list[Message],
        *,
        accumulate=False,
        **kwargs: ChatCompletionParams,
    ) -> str | Message:
        """Postprocess the response, accumulating both content and tool calls."""
        # Ensure the streaming_response is initialized
        if not hasattr(self, "streaming_response"):
            if "tools" in kwargs and kwargs.get("tool_choice") == "required":
                self.streaming_response = Message(role="assistant", content="", choices=[Choice(tool_calls={})])
            else:
                self.streaming_response = Message(role="assistant", content="")

            self.last_tool = None
        # If n=1, handle a single response
        if kwargs.get("n", 1) == 1:
            # Check if tools are being used
            if "tools" in kwargs:
                # If a tool choice is required and tool calls are present
                if (
                    kwargs.get("tool_choice") == "required"
                    and response_chunk.choices
                    and response_chunk.choices[0].delta.tool_calls
                ):
                    self.last_tool = response_chunk.choices[0].delta.tool_calls[0].function.name or getattr(
                        self, "last_tool", "toolfallback"
                    )
                    chunks = self.streaming_response.choices[0].tool_calls.get(self.last_tool, "")
                    # pprint(f"Tool call arguments: {chunks}")
                    # pprint(f"Tool calls: {self.streaming_response.choices[0].tool_calls}")
                    if accumulate:
                        self.streaming_response.choices[0].tool_calls[self.last_tool] = (
                            chunks + response_chunk.choices[0].delta.tool_calls[0].function.arguments
                        )
                    else:
                        self.streaming_response.choices[0].tool_calls[self.last_tool] = (
                            response_chunk.choices[0].delta.tool_calls[0].function.arguments
                        )

                    # pprint(self.streaming_response.content)
                    # pprint(self.streaming_response.choices)
                # globals()["choice_count"] += 1
                # if globals()["choice_count"] == 2:
                #     exit()
                # If there is content (non-tool case), accumulate it
                if response_chunk.choices and response_chunk.choices[0].delta.content:
                    chunks = self.streaming_response.content or ""
                    if accumulate:
                        self.streaming_response.content = chunks + response_chunk.choices[0].delta.content
                    else:
                        self.streaming_response.content = response_chunk.choices[0].delta.content
                yield self.streaming_response
            else:
                # from rich.pretty import pprint
                # pprint(self.streaming_response.content)
                chunk = response_chunk.choices[0].delta.content

                if accumulate:
                    self.streaming_response.content += chunk
                else:
                    self.streaming_response.content = chunk
                yield self.streaming_response.content
                # pprint(self.streaming_response.content)
                # pprint(self.streaming_response.choices)
            # yield self.streaming_response
        else:
            # Handle the multiple choices (n > 1) case
            if "tools" in kwargs:
                self.streaming_response.choices.append([c.delta.tool_calls for c in response_chunk.choices])
            # Accumulate content from all choices
            self.streaming_response.choices.append([c.delta.content for c in response_chunk.choices])
            yield self.streaming_response
        # pprint(self.streaming_response)

        # return self.streaming_response.content if hasattr(self.streaming_response, "content") else self.streaming_response

    def maybe_override_defaults(
        self, model_arg: str | None, **model_kwargs: ChatCompletionParams
    ) -> ChatCompletionParams | GradioParams | dict:
        """Override the default model if a model is provided."""
        model_kwargs = {**self.default_model_kwargs, **model_kwargs}
        if model_arg:
            model_kwargs["model"] = model_arg
        return model_kwargs

    def act(
        self,
        instruction: str,
        image: Image = None,
        context: list | str | Image | Message = None,
        model=None,
        **model_kwargs: ChatCompletionParams | GradioParams,
    ) -> str | Message:
        """Responds to the given instruction, image, and context.

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

        Examples:
            >>> agent.act("Hello, world!", Image("scene.jpeg"))
            "Hello! What can I do for you today?"
            >>> agent.act("Return a plan to pickup the object as a python list.", Image("scene.jpeg"))
            "['Move left arm to the object', 'Move right arm to the object']"
        """
        message, memory = self.prepare_inputs(instruction, image, context)
        model_kwargs = self.maybe_override_defaults(model, **model_kwargs)
        response = self.actor.predict(message, memory, **model_kwargs)
        return self.on_completion_finish(response, message, memory, **model_kwargs)

    def act_and_stream(
        self,
        instruction: str,
        image: Image = None,
        context: list | str | Image | Message = None,
        model=None,
        *,
        accumulate: bool = False,
        **model_kwargs: ChatCompletionParams | GradioParams | dict,
    ) -> Generator[str | ChatCompletionChunk, None, str]:
        """Responds to the given instruction, image, and context and streams the response."""
        message, memory = self.prepare_inputs(instruction, image, context)

        model_kwargs = self.maybe_override_defaults(model, **model_kwargs)

        for chunk in self.actor.stream(message, memory, **model_kwargs):
            yield from self.on_stream_yield(chunk, message, memory, accumulate=accumulate, **model_kwargs)

        if "tools" not in model_kwargs:
            return "".join(
                alwaysiter(self.on_completion_finish(self.streaming_response, message, memory, **model_kwargs))
            )

        return self.on_completion_finish(self.streaming_response, message, memory, **model_kwargs)

    async def async_act_and_stream(
        self,
        instruction: str,
        image: Image = None,
        context: list | str | Image | Message = None,
        model=None,
        **model_kwargs: ChatCompletionParams | GradioParams | dict,
    ) -> AsyncGenerator[str, None]:
        """Responds to the given instruction, image, and context asynchronously and streams the response."""
        message, memory = self.prepare_inputs(instruction, image, context)
        model_kwargs = self.maybe_override_defaults(
            model or model_kwargs.get("model", self.actor.DEFAULT_MODEL), **model_kwargs
        )
        async for chunk in map(self.on_stream_yield, self.actor.astream(message, memory, **model_kwargs)):
            yield chunk


def main():
    agent = LanguageAgent(model_src="openai")
    resp = ""
    for chunk in agent.act_and_stream("Hello, world!"):
        resp += chunk
        print(resp)


async def async_main():
    agent = LanguageAgent(model_src="openai", model_kwargs={"aclient": True})
    resp = ""
    async for chunk in agent.async_act_and_stream("Hello, world!"):
        resp += chunk
        print(resp)


if __name__ == "__main__":
    main()
    asyncio.run(async_main())
