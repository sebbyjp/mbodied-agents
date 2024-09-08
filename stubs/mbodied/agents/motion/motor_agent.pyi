import abc
from abc import abstractmethod
from mbodied.agents import Agent as Agent
from mbodied.types.motion import Motion as Motion

class MotorAgent(Agent, metaclass=abc.ABCMeta):
    """Abstract base class for motor agents.

    Subclassed from Agent, thus possessing the ability to make remote calls, etc.
    """
    @abstractmethod
    def act(self, **kwargs) -> Motion:
        """Generate a Motion based on given parameters.

        Args:
            **kwargs: Arbitrary keyword arguments for motor agent to act on.

        Returns:
            Motion: A Motion object based on the provided arguments.
        """
