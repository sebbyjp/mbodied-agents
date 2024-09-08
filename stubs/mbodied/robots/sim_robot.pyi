from _typeshed import Incomplete
from mbodied.robots import Robot as Robot
from mbodied.types.motion.control import HandControl as HandControl
from mbodied.types.sense.vision import Image as Image

class SimRobot(Robot):
    """A simulated robot interface for testing and validating purposes.

    This class simulates the interface between the robot arm and the control system.
    do() simulates the execution of HandControl motions that got executed in execution_time.

    Attributes:
        home_pos: The home position of the robot arm.
        current_pos: The current position of the robot arm.
    """
    execution_time: Incomplete
    home_pos: Incomplete
    current_pos: Incomplete
    def __init__(self, execution_time: float = 1.0) -> None:
        """Initializes the SimRobot and sets up the robot arm.

        Args:
            execution_time: The time it takes to execute a motion.

        position: [x, y, z, r, p, y, grasp]
        """
    def do(self, motion: HandControl | list[HandControl]) -> list[float]:
        """Executes HandControl motions and returns the new position of the robot arm.

        This simulates the execution of each motion for self.execution_time. It divides the motion into 10 steps.

        Args:
            motion: The HandControl motion to be executed.
        """
    def capture(self, **_) -> Image:
        """Captures an image."""
    def get_observation(self) -> Image:
        """Alias of capture for recording."""
    def get_state(self) -> HandControl:
        """Gets the current pose of the robot arm.

        Returns:
            list[float]: A list of the current pose values [x, y, z, r, p, y, grasp].
        """
    def prepare_action(self, old_pose: HandControl, new_pose: HandControl) -> HandControl:
        """Calculates the action between two poses."""
