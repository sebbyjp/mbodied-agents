from _typeshed import Incomplete

from mbodied.hardware.interface import HardwareInterface as HardwareInterface
from mbodied.types.motion.control import HandControl as HandControl
from mbodied.types.sense.vision import Image as Image

class SimInterface(HardwareInterface):
    """A simulated interface for testing and validating purposes.

    This class simulates the interface between the robot arm and the control system.

    Attributes:
        home_pos: The home position of the robot arm.
        current_pos: The current position of the robot arm.
    """
    home_pos: Incomplete
    current_pos: Incomplete
    def __init__(self) -> None:
        """Initializes the SimInterface and sets up the robot arm.

        position: [x, y, z, r, p, y, grasp]
        """
    def do(self, motion: HandControl) -> list[float]:
        """Executes a given HandControl motion and returns the new position of the robot arm.

        Args:
            motion: The HandControl motion to be executed.
        """
    def get_pose(self) -> list[float]:
        """Gets the current pose of the robot arm.

        Returns:
            list[float]: A list of the current pose values [x, y, z, r, p, y, grasp].
        """
    def capture(self, **_) -> Image:
        """Captures an image."""
