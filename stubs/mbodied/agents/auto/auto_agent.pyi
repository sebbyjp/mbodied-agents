import mbodied.agents.agent
import typing
from mbodied.agents.agent import Agent as Agent
from mbodied.agents.language.language_agent import LanguageAgent as LanguageAgent
from mbodied.agents.motion.openvla_agent import OpenVlaAgent as OpenVlaAgent
from mbodied.agents.sense.depth_estimation_agent import DepthEstimationAgent as DepthEstimationAgent
from mbodied.agents.sense.object_detection_agent import ObjectDetectionAgent as ObjectDetectionAgent
from mbodied.agents.sense.segmentation_agent import SegmentationAgent as SegmentationAgent
from mbodied.types.sense.vision import Image as Image
from typing import Any, ClassVar

class AutoAgent(mbodied.agents.agent.Agent):
    TASK_TO_AGENT_MAP: ClassVar[dict] = ...
    def __init__(self, task=..., model_src=..., model_kwargs=..., **kwargs) -> None:
        """Initialize the AutoAgent with the specified task and model."""
    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access to the agent if not found in AutoAgent."""
    def act(self, *args, **kwargs) -> Any:
        """Invoke the agent's act method without reinitializing the agent."""
    @staticmethod
    def available_tasks() -> None:
        """Print available tasks that can be used with AutoAgent."""
def get_agent(task: typing.Literal, model_src: str, model_kwargs=..., **kwargs) -> mbodied.agents.agent.Agent:
    '''Initialize the AutoAgent with the specified task and model.

        This is an alternative to using the AutoAgent class directly. It returns the corresponding agent instance directly.

        Usage:
        ```python
        # Get LanguageAgent instance
        language_agent = get_agent(task="language", model_src="openai")
        response = language_agent.act("What is the capital of France?")

        # Get OpenVlaAgent instance
        openvla_agent = get_agent(task="motion-openvla", model_src="https://api.mbodi.ai/community-models/")
        action = openvla_agent.act("move hand forward", Image(size=(224, 224)))

        # Get DepthEstimationAgent instance
        depth_agent = get_agent(task="sense-depth-estimation", model_src="https://api.mbodi.ai/sense/")
        depth = depth_agent.act(image=Image("resources/bridge_example.jpeg", size=(224, 224)))
        ```
    '''
