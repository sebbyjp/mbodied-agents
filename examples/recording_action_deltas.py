from pathlib import Path
from typing import TypeVar

from mbodied.robots import Robot
from mbodied.types.motion.control import HandControl
from mbodied.types.sense.vision import Image


class MockImageHandRobot(Robot[Image, HandControl, HandControl, None]):
    """A simulated robot interface for testing and validating purposes.

    This class simulates the interface between the robot arm and the control system.
    do() simulates the execution of HandControl motions that got executed in execution_time.

    Attributes:
        home_state: The default state of the robot arm.
        current_state: The current state of the robot arm.
    """

    def __init__(self, execution_time: float = 1.0):
        """Initializes the SimRobot and sets up the robot arm.

        Args:
            execution_time: The time it takes to execute a motion.
        """
        super().__init__(execution_time)

    def capture(self, **_) -> ObsT:
        """Captures an image."""
        resource = Path("resources") / "xarm.jpeg"
        return Image(resource, size=(224, 224))

    def prepare_action(self, old_pose: HandControl, new_pose: HandControl) -> HandControl:
        """Calculates the action between two poses."""
        # Calculate the difference between the old and new poses. Use absolute value for grasp.
        old = list(old_pose.flatten())
        new = list(new_pose.flatten())
        result = [(new[i] - old[i]) for i in range(len(new) - 1)] + [new[-1]]
        return HandControl.unflatten(result)
