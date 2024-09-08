from _typeshed import Incomplete
from gradio.components import Component as Component
from pydantic.json_schema import JsonSchemaValue as JsonSchemaValue
from pydantic.types import SecretStr as SecretStr
from pydantic_settings import BaseSettings
from typing_extensions import Callable, TypedDict

WEBSOCKET_URI: str

class State(TypedDict, total=False): ...

class Guidance(TypedDict, total=False):
    choices: list[str] | None
    json: str | dict | JsonSchemaValue | None

class BaseAgentConfig(BaseSettings):
    model_config: Incomplete
    base_url: str
    auth_token: SecretStr

class CompletionConfig(TypedDict, total=False):
    guidance: Guidance | None
    pre_process: Callable[[str, State], str] | BaseAgentConfig | None
    post_process: Callable[[str, str, State], str] | BaseAgentConfig | None
    prompt: str | Callable | None
    reminder: str | None

class AgentConfig(BaseAgentConfig):
    base_url: str
    auth_token: str
    system_prompt: str | None
    completion: CompletionConfig
    stream: CompletionConfig | None
    sub_agents: list['AgentConfig'] | None
    state: State
    gradio_io: tuple[Component, Component] | None

class STTConfig(AgentConfig):
    agent_base_url: str
    agent_token: str
    transcription_endpoint: str
    translation_endpoint: str
    websocket_uri: str
    place_holder: str

def persist_post_process(_prompt: str, response: str, state: State, shared_state: State | None = None) -> str:
    """This is useful to stabalize the response of the agent."""
def actor_pre_process(prompt: str, state: State, shared_state: State | None = None) -> str: ...
def actor_post_process(prompt: str, response: str, state: State, shared_state: State | None = None) -> str: ...
def speaker_post_process(prompt: str, response: str, state: State, shared_state: State | None = None) -> str: ...
def instruct_pre_process(prompt: str, state: State, shared_state: State | None = None) -> str:
    """Either wait or forward the prompt to the agent."""
def instruct_post_process(prompt: str, response: str, state: State, shared_state: State | None = None) -> str:
    """Signal movement if the response is movement."""

class TranslateConfig(CompletionConfig):
    source_language: str
    target_language: str
    prompt: Callable
    reminder: str

class InstructConfig(CompletionConfig):
    prompt: str
    reminder: str
    guidance: Guidance
    post_process: Callable[[str, str, State], str]
