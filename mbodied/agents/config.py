import os

from gradio.components import Component
from pydantic import Field
from pydantic.json_schema import JsonSchemaValue
from pydantic.types import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing_extensions import Callable, Dict, TypedDict, dataclass_transform  # noqa: UP035

WEBSOCKET_URI="http://localhost:7543/v1/audio/transcriptions"


"""
The logic of Embodied Agents V2 is not unsimilar to general concurrent programming.Every agent has a responsibility, a local state, 
a shared state, and completion configuration for querying an endpoint. The novelty of this framework,
is that emphasis is placed on synchronizing a shared state with the latest and most accurate information. This paradigm is
similar to Behavior Trees but requires only 


One can immediately notice that the problem of concurrency is quite ubiquitous. From low level multiprocessing with
and context management, to handling user sessions, to desiging a RAG system, at its core, the problem is just safely
updating a **State** with multiple operators and passing as little context as possible through  **State Transition**s.

The true challenge is again, as it always is, scale. Only in robotics, one needs to reach truly massive levels
of synchronized concurrency for just a single embodiment. Image scheduling a zoom call between just 10
people who all need to do a certain task based on the results of everyone else's task and if one person fails,
the whole system fails. Oh and the whole group needs to do this at least 10 times a second.

You may wonder, isn't that just what normal websites are doing which is even faster and handles more processes? Yes... and no. 
The difference is the complexity or richness of the data that is being sent and operated on. It must be sent

1. Exactly correctly.
2. Fast Enough to make a cognitive decision.

A cell network doesnt use much brain power to decide what phone to send a text to. But
an embodied agent needs to think about what to do next for each individual part of the outside world,
its internal state, monitor its progress, have fail safes, and all around as fast as you are able to move your hand.

For example, a web server may have no need for a real-time shared state but an embodied collective of agents
MUST stop moving its hand before it hits the table. So instead of considering topologies or specific communication patterns like
server/client or pub/sub, we define only the FSM and the transitions. Any message passing or network protocol can
be hooked up with currently http, grpc, websockets, and ROS currently supported.

The benefit of using AgentsV2 is that ever piece of data is backed by a pydantic schema and conversion methods to any
other data type you will likely need. Some of the most common include:

- JSON 
- Dict
- Gym Space Sample
- RLDS Dataset
- ROS Message
- Apache Arrow Compatible GRPC
- PyArrow Table and HuggingFace Dataset Directly
- Numpy, Torch, and TF Tensors
- VLLM MultiModal Data Input
- Furthermore, 3D geometry and trajectory data operations are supported out of the box.




"""
@dataclass_transform(order_default=True)
class State(TypedDict, total=False):
    pass

@dataclass_transform(order_default=True)
class Guidance(TypedDict, total=False):
    choices: list[str] | None = Field(default=None, examples=[lambda: ["Yes", "No"]])
    json: str | Dict| JsonSchemaValue | None = Field(
      default=None,
      description="The json schema as a dict or string.",
      examples=[{"type": "object", "properties": {"key": {"type": "string"}}}],
    )


class BaseAgentConfig(BaseSettings):
    model_config = SettingsConfigDict(cli_parse_args=False)
    base_url: str = "https://api.mbodi.ai/v1"
    auth_token: SecretStr = os.getenv("MBODI_API_KEY", "mbodi-demo-1")


@dataclass_transform(order_default=True)
class CompletionConfig(TypedDict, total=False):
    guidance: Guidance | None = Field(default=None, examples=[Guidance(choices=["Yes", "No"])])
    pre_process: Callable[[str, State], str] | BaseAgentConfig | None = Field(default=None, description="A callable or agent that takes the prompt and state and returns a modified prompt.")
    post_process: Callable[[str, str, State], str] | BaseAgentConfig| None = Field(default=None, examples=[lambda prompt, response: prompt if response == "yes" else ""],
        description="A callable or agent that takes the prompt, response, and state and returns a modified prompt.")
    prompt: str | Callable | None = Field(
      default="Give a command for how the robot should move in the following json format:",
      examples=[lambda x, y: f"Translate the following text to {x} if it is {y} and vice versa. Respond with only the translated text and nothing else."],
      description="The prompt to be used in the completion. Use a callable if you want to add extra information at the time of agent action."
    )
    reminder: str | None = Field(default=None, examples=["Remember to respond with only the translated text and nothing else."])


class AgentConfig(BaseAgentConfig):
    base_url: str = "https://api.mbodi.ai/v1"
    auth_token: str = "mbodi-demo-1"
    system_prompt: str | None = Field(default=None)
    completion: CompletionConfig = Field(default_factory=CompletionConfig)
    stream: CompletionConfig | None = Field(default_factory=CompletionConfig)
    sub_agents: list["AgentConfig"] | None = Field(default=None)
    state: State = Field(default_factory=State)
    gradio_io: tuple[Component, Component] | None = Field(default=None, description="The input and output components for the Gradio interface.")


class STTConfig(AgentConfig):
    agent_base_url: str = "http://localhost:3389/v1"
    agent_token: str = "mbodi-demo-1"
    transcription_endpoint: str = "/audio/transcriptions"
    translation_endpoint: str = "/audio/translations"
    websocket_uri: str = "wss://api.mbodi.ai/audio/v1/transcriptions"
    place_holder: str = "Loading can take 30 seconds if a new model is selected..."
    


"""
Shared State is a dictionary that is shared between all the agents in the pipeline. It is used to store information that is needed by multiple agents.
An example of shared state is the `clear` key that is used to clear the state of all the agents in the pipeline.
"""
def persist_post_process(_prompt:str, response:str, state:State, shared_state:State | None = None) -> str:
    """This is useful to stabalize the response of the agent."""
    if shared_state.get("clear"):
        state.clear()
        return ""
    persist = state.get("persist", response)
    return response if persist in response or persist in ("No audio yet...", "Not a complete instruction") else persist



def actor_pre_process(prompt: str, state: State, shared_state: State | None = None) -> str:
    if shared_state.get("clear"):
        state.clear()
        return ""
    if shared_state.get("actor_status") == "wait":
        return ""
    return prompt

def actor_post_process(prompt: str, response: str, state: State, shared_state: State | None = None) -> str:
    if shared_state.get("clear"):
        state.clear()
        return ""
    shared_state["speaker_status"] = "ready"
    return persist_post_process(prompt, response, state, shared_state)


def speaker_post_process(prompt: str, response: str, state: State, shared_state: State | None = None) -> str:
    if shared_state.get("clear"):
        state.clear()
        return ""
    shared_state["speaker_status"] = "done"
    shared_state["actor_status"] = "wait"
    shared_state["instruct_status"] = "ready"
    state[""]
    return persist_post_process(prompt, response, state, shared_state), 


def instruct_pre_process(prompt: str, state: State, shared_state: State | None = None) -> str:
    """Either wait or forward the prompt to the agent."""
    if shared_state.get("clear"):
        state.clear()
        return ""
    if shared_state.get("instruct_status") == "wait":
        return ""
    return prompt

def instruct_post_process(prompt: str, response: str, state: State, shared_state: State | None = None) -> str:
    """Signal movement if the response is movement."""
    if shared_state.get("clear"):
        state.clear()
        return ""
    if response in ["incomplete", "noise"]:
        shared_state["actor_status"] = "wait"
        return state.get("persist", "Not a complete instruction")
    if response == "movement":
        shared_state["moving"] = True
    shared_state.instruct_status = "repeat"
    shared_state.actor_status = "ready"
    return persist_post_process(prompt, response, state, shared_state)

class TranslateConfig(CompletionConfig):
    source_language: str = "en"
    target_language: str = "en"
    prompt: Callable = lambda x, y: f"Translate the following text to {x} if it is {y} and vice versa. Respond with only the translated text and nothing else."
    reminder: str = "Remember to respond with only the translated text and nothing else."

class InstructConfig(CompletionConfig):
    prompt: str = "Determine whether the following text is a command for physical movement,other actionable command, question, incomplete statement, or background noise. Respond with only ['movement', 'command', 'question', 'incomplete', 'noise']"
    reminder: str = "Remember that you should be very confident to label a command as movement. If you are not sure, label it as noise. You should be very eager to label a question as a question and a command as a command. If you are not sure, label it as incomplete."
    guidance: Guidance = Guidance(choices=["movement","command", "question", "incomplete", "noise"])
    post_process: Callable[[str, str, State], str] = instruct_post_process
