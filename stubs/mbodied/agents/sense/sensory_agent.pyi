from mbodied.agents import Agent as Agent
from mbodied.types.sense.sensor_reading import SensorReading as SensorReading

class SensoryAgent(Agent):
    """Abstract base class for sensory agents.

    This class provides a template for creating agents that can sense the environment.

    Attributes:
        kwargs (dict): Additional arguments to pass to the recorder.
    """
    def __init__(self, **kwargs) -> None:
        """Initialize the agent.

        Args:
            **kwargs: Additional arguments to pass to the recorder.
        """
    def act(self, **kwargs) -> SensorReading:
        """Abstract method to define the sensing mechanism of the agent.

        Args:
            **kwargs: Additional arguments to pass to the `sense` method.

        Returns:
            Sample: The sensory sample created by the agent.
        """
    def sense(self, **kwargs) -> SensorReading:
        """Generate a SensorReading based on given parameters.

        Args:
            **kwargs: Arbitrary keyword arguments for sensory agent to sense on.

        Returns:
            SensorReading: A SensorReading object based on the provided arguments.
        """
