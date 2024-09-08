from _typeshed import Incomplete
from gymnasium import spaces as spaces
from mbodied.robots import Robot as Robot
from mbodied.types.motion.control import HandControl as HandControl
from mbodied.types.sense.vision import Image as Image

class XarmRobot(Robot):
    """Control the xArm robot arm with SDK.

    Usage:
        xarm = XarmRobot()
        xarm.do(HandControl(...))

    Attributes:
        ip: The IP address of the xArm robot.
        arm: The XArmAPI instance for controlling the robot.
        home_pos: The home position of the robot arm.
    """
    ip: Incomplete
    arm: Incomplete
    home_pos: Incomplete
    arm_speed: int
    use_realsense: bool
    realsense_camera: Incomplete
    def __init__(self, ip: str = '192.168.1.228', use_realsense: bool = False) -> None:
        """Initializes the XarmRobot and sets up the robot arm.

        Args:
            ip: The IP address of the xArm robot.
            use_realsense: Whether to use a RealSense camera for capturing images
        """
    def do(self, motion: HandControl | list[HandControl]) -> None:
        """Executes HandControl(s).

        HandControl is in meters and radians.

        Args:
            motion: The HandControl motion(s) to be executed.
        """
    def get_state(self) -> HandControl:
        """Gets the current pose (absolute HandControl) of the robot arm.

        Returns:
            The current pose of the robot arm.
        """
    def prepare_action(self, old_pose: HandControl, new_pose: HandControl) -> HandControl:
        """Calculates the action between two poses.

        Args:
            old_pose: The old pose(state) of the hardware.
            new_pose: The new pose(state) of the hardware.

        Returns:
            The action to be taken between the old and new poses.
        """
    def capture(self) -> Image:
        """Captures an image from the robot camera."""
    def get_observation(self) -> Image:
        """Captures an image for recording."""
